#pragma once
#include <ll/api/Config.h>
#include <ll/api/mod/NativeMod.h>

namespace parallel_collision {

struct Config {
    int  version = 1;
    bool enabled = true;
    bool debug   = false;
    bool skipPlayers = true; // 玩家碰撞仍由原版处理（避免延迟）
};

Config& getConfig();
bool    loadConfig();
bool    saveConfig();

class ParallelCollisionOptimizer {
public:
    static ParallelCollisionOptimizer& getInstance();

    ParallelCollisionOptimizer() : mSelf(*ll::mod::NativeMod::current()) {}

    [[nodiscard]] ll::mod::NativeMod& getSelf() const { return mSelf; }

    bool load();
    bool enable();
    bool disable();

private:
    ll::mod::NativeMod& mSelf;
};

} // namespace parallel_collision
