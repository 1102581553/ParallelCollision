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
#include <unordered_set>

namespace parallel_collision {

static Config config;
static std::shared_ptr<ll::io::Logger> log;
static bool debugTaskRunning = false;

static size_t totalDetected = 0;
static size_t totalTicks = 0;
static size_t lastEntityCount = 0;

static bool g_processingParallel = false;

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
    Vec3 pos;
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

struct GridCoord {
    int x, y;
    bool operator==(const GridCoord& other) const { return x == other.x && y == other.y; }
};
struct GridCoordHash {
    std::size_t operator()(const GridCoord& c) const {
        return std::hash<int>()(c.x) ^ (std::hash<int>()(c.y) << 1);
    }
};
using Grid = std::unordered_map<GridCoord, std::vector<size_t>, GridCoordHash>;

static GridCoord getGridCoord(const Vec3& pos) {
    return {
        static_cast<int>(std::floor(pos.x / GRID_CELL_SIZE)),
        static_cast<int>(std::floor(pos.z / GRID_CELL_SIZE))
    };
}

static Grid buildGrid(const std::vector<EntitySnapshot>& snapshots) {
    Grid grid;
    for (size_t i = 0; i < snapshots.size(); ++i) {
        auto coord = getGridCoord(snapshots[i].pos);
        grid[coord].push_back(i);
    }
    return grid;
}

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
                for (int dx = -1; dx <= 1; ++dx) {
                    for (int dy = -1; dy <= 1; ++dy) {
                        GridCoord neighbor = {center.x + dx, center.y + dy};
                        auto it = grid.find(neighbor);
                        if (it == grid.end()) continue;
                        for (size_t j : it->second) {
                            if (j <= i) continue;
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

// 贪心分组：将事件分成多个互不相交的批次（每个批次内所有事件的实体集无重叠）
static std::vector<std::vector<CollisionEvent>> partitionEvents(const std::vector<CollisionEvent>& events) {
    std::vector<std::vector<CollisionEvent>> batches;
    if (events.empty()) return batches;

    // 使用贪心算法：对于每个事件，尝试放入已有的批次，如果与批次内任何实体的ID冲突，则创建新批次
    // 为了加速，我们维护每个批次当前占用的实体集合
    std::vector<std::unordered_set<ActorUniqueID>> batchOccupied;
    for (const auto& ev : events) {
        bool placed = false;
        for (size_t i = 0; i < batches.size(); ++i) {
            // 检查冲突
            if (batchOccupied[i].count(ev.a) == 0 && batchOccupied[i].count(ev.b) == 0) {
                batches[i].push_back(ev);
                batchOccupied[i].insert(ev.a);
                batchOccupied[i].insert(ev.b);
                placed = true;
                break;
            }
        }
        if (!placed) {
            // 创建新批次
            batches.push_back({ev});
            batchOccupied.push_back({ev.a, ev.b});
        }
    }
    return batches;
}

// 处理单个批次的事件（在子线程中调用）
static void processBatch(Level& level, const std::vector<CollisionEvent>& batch) {
    for (const auto& e : batch) {
        Actor* actorA = level.fetchEntity(e.a, false);
        Actor* actorB = level.fetchEntity(e.b, false);
        if (!actorA || !actorB) continue;

        PushableComponent* pushA = tryGetPushableComponent(*actorA);
        PushableComponent* pushB = tryGetPushableComponent(*actorB);
        if (!pushA && !pushB) continue;

        // 注意：这里不再设置全局标志，因为我们在并行处理批次，原版推动已被钩子禁用，除非玩家放行。
        // 但批次处理时，我们仍然需要允许推动，所以应设置 g_processingParallel = true。
        // 然而 g_processingParallel 是全局布尔，如果多个线程同时设置，可能互相覆盖。
        // 改为在每个线程内临时设置为 true，但需要考虑并发。由于原版推动钩子只读该标志，多个线程同时写可能导致数据竞争（非原子）。
        // 我们改为使用 thread_local 变量，但钩子中无法访问线程局部变量。
        // 另一种方法：在钩子中检查当前线程是否正在处理并行事件。我们可以通过一个线程局部标志来实现。
        // 这里为了简化，我们暂时不在钩子中区分，而是依赖 skipPlayers 配置和玩家放行逻辑。
        // 实际上，批次处理时，我们希望推动被执行，但原版推动钩子会拦截非玩家推动（因为 g_processingParallel 为 false 且非玩家会跳过）。
        // 因此我们必须设置一个标志。为了线程安全，我们可以使用原子布尔，但每个线程需要自己的标志。最好用 thread_local。
        // 由于我们无法修改钩子来读取 thread_local，我们可以改为在钩子中检查另一个全局原子计数器，表示当前正在处理批次的线程数。但这也不可靠。
        // 简单方案：在 processBatch 中直接调用 push，不依赖钩子，因为我们已经禁用了原版推动？实际上原版推动仍在钩子中被拦截，我们需要确保推动被执行。
        // 我们可以临时设置一个 thread_local 标志，然后在钩子中判断该标志。但钩子函数是静态的，无法访问 thread_local 除非声明为线程局部。
        // 我们将声明一个 thread_local bool t_processingBatch = false，在 processBatch 中设置为 true，钩子中检查它。
        // 注意：钩子中需要访问 thread_local 变量，必须在文件作用域声明。
    }
}

// 线程局部标志，用于钩子判断
thread_local bool t_processingBatch = false;

// 处理批次（实际执行推动）
static void processBatchImpl(Level& level, const std::vector<CollisionEvent>& batch) {
    t_processingBatch = true;
    for (const auto& e : batch) {
        Actor* actorA = level.fetchEntity(e.a, false);
        Actor* actorB = level.fetchEntity(e.b, false);
        if (!actorA || !actorB) continue;

        PushableComponent* pushA = tryGetPushableComponent(*actorA);
        PushableComponent* pushB = tryGetPushableComponent(*actorB);
        if (!pushA && !pushB) continue;

        if (pushA) pushA->push(*actorA, *actorB, false);
        if (pushB) pushB->push(*actorB, *actorA, false);
    }
    t_processingBatch = false;
}

// 并行处理所有批次
static long long processCollisionEventsParallel(Level& level, const std::vector<CollisionEvent>& events) {
    auto processStart = std::chrono::steady_clock::now();

    // 分组
    auto groupStart = std::chrono::steady_clock::now();
    auto batches = partitionEvents(events);
    auto groupEnd = std::chrono::steady_clock::now();
    auto groupElapsed = std::chrono::duration_cast<std::chrono::microseconds>(groupEnd - groupStart).count();

    if (config.debug) {
        getLogger().info("Partitioned {} events into {} batches, took {} us", events.size(), batches.size(), groupElapsed);
    }

    // 并行处理每个批次
    std::vector<std::future<void>> futures;
    futures.reserve(batches.size());
    for (const auto& batch : batches) {
        futures.push_back(std::async(std::launch::async, [&level, &batch]() {
            processBatchImpl(level, batch);
        }));
    }

    for (auto& f : futures) {
        f.get();
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
    // 如果当前线程正在处理并行批次，则允许执行
    if (t_processingBatch) {
        origin(owner, other, pushSelfOnly);
        return;
    }
    // 或者如果是玩家且配置跳过玩家，则允许原版推动
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

        auto processElapsed = processCollisionEventsParallel(*this, events);

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
