#include "ParallelCollisionOptimizer.h"
#include <ll/api/memory/Hook.h>
#include <ll/api/mod/RegisterHelper.h>
#include <ll/api/io/Logger.h>
#include <ll/api/io/LoggerRegistry.h>
#include <ll/api/thread/ServerThreadExecutor.h>
#include <ll/api/coro/CoroTask.h>
#include <mc/world/level/Level.h>
#include <mc/world/actor/Actor.h>
#include <mc/world/actor/ActorCategory.h>
#include <mc/world/phys/AABB.h>
#include <mc/legacy/ActorUniqueID.h>
#include <mc/entity/components_json_legacy/PushableComponent.h>
#include <entt/entt.hpp>
#include <filesystem>
#include <chrono>
#include <vector>
#include <algorithm>
#include <thread>
#include <future>
#include <mutex>
#include <unordered_map>

namespace parallel_collision {

static Config config;
static std::shared_ptr<ll::io::Logger> log;
static bool debugTaskRunning = false;

static size_t totalDetected = 0;
static size_t totalTicks = 0;
static size_t lastEntityCount = 0;

static bool g_processingParallel = false;

// 网格大小（每个单元格边长，单位：方块）
constexpr int GRID_CELL_SIZE = 16;

static ll::io::Logger& getLogger() {
    if (!log) {
        log = ll::io::LoggerRegistry::getInstance().getOrCreate("ParallelCollision");
    }
    return *log;
}

Config& getConfig() { return config; }

bool loadConfig() {
    auto path = ParallelCollisionOptimizer::getInstance().getSelf().getConfigDir() / "config.json";
    return ll::config::loadConfig(config, path);
}

bool saveConfig() {
    auto path = ParallelCollisionOptimizer::getInstance().getSelf().getConfigDir() / "config.json";
    return ll::config::saveConfig(config, path);
}

static void resetStats() {
    totalDetected = 0;
    totalTicks = 0;
}

static void startDebugTask() {
    if (debugTaskRunning) return;
    debugTaskRunning = true;

    ll::coro::keepThis([]() -> ll::coro::CoroTask<> {
        while (debugTaskRunning) {
            co_await std::chrono::seconds(5);
            ll::thread::ServerThreadExecutor::getDefault().execute([] {
                if (!config.debug) return;
                size_t avgPerTick = totalTicks > 0 ? totalDetected / totalTicks : 0;
                getLogger().info(
                    "Collision stats (5s): total events={}, avg per tick={}, last entity count={}",
                    totalDetected, avgPerTick, lastEntityCount
                );
                resetStats();
            });
        }
        debugTaskRunning = false;
    }).launch(ll::thread::ServerThreadExecutor::getDefault());
}

static void stopDebugTask() { debugTaskRunning = false; }

struct EntitySnapshot {
    ActorUniqueID uid;
    AABB aabb;
    Vec3 pos;           // 用于网格定位
    bool isPlayer;
};

struct CollisionEvent {
    ActorUniqueID a;
    ActorUniqueID b;
};

static PushableComponent* tryGetPushableComponent(Actor& actor) {
    auto& ctx = actor.getEntityContext();
    auto comp = ctx.tryGetComponent<PushableComponent>();
    return comp.has_value() ? &comp.value() : nullptr;
}

// 网格坐标
struct GridCoord {
    int x, y;
    bool operator==(const GridCoord& other) const { return x == other.x && y == other.y; }
};
struct GridCoordHash {
    std::size_t operator()(const GridCoord& c) const {
        return std::hash<int>()(c.x) ^ (std::hash<int>()(c.y) << 1);
    }
};

// 网格：每个单元格存储实体索引列表
using Grid = std::unordered_map<GridCoord, std::vector<size_t>, GridCoordHash>;

// 从位置获取网格坐标
static GridCoord getGridCoord(const Vec3& pos) {
    return {
        static_cast<int>(std::floor(pos.x / GRID_CELL_SIZE)),
        static_cast<int>(std::floor(pos.z / GRID_CELL_SIZE)) // 使用 x,z 平面
    };
}

// 构建网格索引
static Grid buildGrid(const std::vector<EntitySnapshot>& snapshots) {
    Grid grid;
    for (size_t i = 0; i < snapshots.size(); ++i) {
        auto coord = getGridCoord(snapshots[i].pos);
        grid[coord].push_back(i);
    }
    return grid;
}

// 返回检测耗时（微秒）
static std::pair<std::vector<CollisionEvent>, long long> detectCollisionsParallel(
    const std::vector<EntitySnapshot>& snapshots,
    const Grid& grid,
    size_t numThreads
) {
    auto detectStart = std::chrono::steady_clock::now();
    std::vector<CollisionEvent> events;
    if (snapshots.empty()) return {events, 0};

    const size_t N = snapshots.size();
    std::mutex eventsMutex;
    std::vector<std::future<void>> futures;
    futures.reserve(numThreads);

    size_t chunkSize = (N + numThreads - 1) / numThreads;

    for (size_t t = 0; t < numThreads; ++t) {
        size_t start = t * chunkSize;
        size_t end = std::min(start + chunkSize, N);
        if (start >= end) break;

        futures.push_back(std::async(std::launch::async, [&snapshots, &grid, start, end, &events, &eventsMutex]() {
            std::vector<CollisionEvent> localEvents;
            for (size_t i = start; i < end; ++i) {
                const auto& si = snapshots[i];
                GridCoord center = getGridCoord(si.pos);
                // 遍历相邻 9 个单元格
                for (int dx = -1; dx <= 1; ++dx) {
                    for (int dy = -1; dy <= 1; ++dy) {
                        GridCoord neighbor = {center.x + dx, center.y + dy};
                        auto it = grid.find(neighbor);
                        if (it == grid.end()) continue;
                        for (size_t j : it->second) {
                            if (j <= i) continue; // 只检查 i < j，避免重复
                            const auto& sj = snapshots[j];
                            if (config.skipPlayers && (si.isPlayer || sj.isPlayer)) {
                                continue;
                            }
                            if (si.aabb.intersects(sj.aabb)) {
                                localEvents.push_back({si.uid, sj.uid});
                            }
                        }
                    }
                }
            }
            if (!localEvents.empty()) {
                std::lock_guard<std::mutex> lock(eventsMutex);
                events.insert(events.end(), localEvents.begin(), localEvents.end());
            }
        }));
    }

    for (auto& f : futures) {
        f.get();
    }

    auto detectEnd = std::chrono::steady_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(detectEnd - detectStart).count();
    return {std::move(events), elapsed};
}

static long long processCollisionEvents(Level& level, const std::vector<CollisionEvent>& events) {
    auto processStart = std::chrono::steady_clock::now();
    for (const auto& e : events) {
        Actor* actorA = level.fetchEntity(e.a, false);
        Actor* actorB = level.fetchEntity(e.b, false);
        if (!actorA || !actorB) continue;

        PushableComponent* pushA = tryGetPushableComponent(*actorA);
        PushableComponent* pushB = tryGetPushableComponent(*actorB);
        if (!pushA && !pushB) continue;

        g_processingParallel = true;

        if (pushA) pushA->push(*actorA, *actorB, false);
        if (pushB) pushB->push(*actorB, *actorA, false);

        g_processingParallel = false;
    }
    auto processEnd = std::chrono::steady_clock::now();
    return std::chrono::duration_cast<std::chrono::microseconds>(processEnd - processStart).count();
}

using PushFunc = void(PushableComponent::*)(Actor&, Actor&, bool);

LL_AUTO_TYPE_INSTANCE_HOOK(
    PushableComponentPushHook,
    ll::memory::HookPriority::Normal,
    PushableComponent,
    static_cast<PushFunc>(&PushableComponent::push),
    void,
    class Actor& owner,
    class Actor& other,
    bool pushSelfOnly
) {
    if (g_processingParallel) {
        origin(owner, other, pushSelfOnly);
        return;
    }
    if (config.skipPlayers && (owner.isPlayer() || other.isPlayer())) {
        origin(owner, other, pushSelfOnly);
        return;
    }
    // 否则跳过
}

LL_AUTO_TYPE_INSTANCE_HOOK(
    LevelTickHook,
    ll::memory::HookPriority::Normal,
    Level,
    &Level::$tick,
    void
) {
    auto tickStart = std::chrono::steady_clock::now();

    if (config.enabled) {
        auto actors = this->getRuntimeActorList();
        lastEntityCount = actors.size();

        auto snapshotStart = std::chrono::steady_clock::now();
        std::vector<EntitySnapshot> snapshots;
        snapshots.reserve(actors.size());
        for (Actor* actor : actors) {
            if (!actor) continue;
            snapshots.push_back({
                .uid = actor->getOrCreateUniqueID(),
                .aabb = actor->getAABB(),
                .pos = actor->getPosition(),
                .isPlayer = actor->isPlayer()
            });
        }
        auto snapshotEnd = std::chrono::steady_clock::now();
        auto snapshotElapsed = std::chrono::duration_cast<std::chrono::microseconds>(snapshotEnd - snapshotStart).count();

        auto gridStart = std::chrono::steady_clock::now();
        Grid grid = buildGrid(snapshots);
        auto gridEnd = std::chrono::steady_clock::now();
        auto gridElapsed = std::chrono::duration_cast<std::chrono::microseconds>(gridEnd - gridStart).count();

        size_t numThreads = std::max(1u, std::thread::hardware_concurrency());
        auto [events, detectElapsed] = detectCollisionsParallel(snapshots, grid, numThreads);
        totalDetected += events.size();
        totalTicks++;

        auto processElapsed = processCollisionEvents(*this, events);

        auto tickEnd = std::chrono::steady_clock::now();
        auto totalElapsed = std::chrono::duration_cast<std::chrono::microseconds>(tickEnd - tickStart).count();

        if (config.debug) {
            getLogger().info(
                "Tick debug: entities={}, events={}, threads={}, snapshot={}us, grid={}us, detect={}us, process={}us, total={}us",
                actors.size(), events.size(), numThreads,
                snapshotElapsed, gridElapsed, detectElapsed, processElapsed, totalElapsed
            );
        }
    }

    origin();
}

ParallelCollisionOptimizer& ParallelCollisionOptimizer::getInstance() {
    static ParallelCollisionOptimizer instance;
    return instance;
}

bool ParallelCollisionOptimizer::load() {
    std::filesystem::create_directories(getSelf().getConfigDir());
    if (!loadConfig()) {
        getLogger().warn("Failed to load config, using defaults and saving");
        saveConfig();
    }
    getLogger().info("Loaded. enabled={}, debug={}, skipPlayers={}",
                     config.enabled, config.debug, config.skipPlayers);
    return true;
}

bool ParallelCollisionOptimizer::enable() {
    if (config.debug) startDebugTask();
    getLogger().info("Enabled.");
    return true;
}

bool ParallelCollisionOptimizer::disable() {
    stopDebugTask();
    resetStats();
    getLogger().info("Disabled");
    return true;
}

} // namespace parallel_collision

LL_REGISTER_MOD(parallel_collision::ParallelCollisionOptimizer, parallel_collision::ParallelCollisionOptimizer::getInstance());
