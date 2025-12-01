#ifndef GOGAME_AI_PLAYER_H
#define GOGAME_AI_PLAYER_H

#include "../game/board.h"
#include "minimax/minimax_engine.h"
#include "mcts/mcts_engine.h"
#include <map>
#include <optional>
#include <string>

class AIPlayer {
public:
    enum class Algorithm {
        Minimax,
        MCTS,
    };

    struct LevelConfig {
        Algorithm algorithm;
        MinimaxEngine::Config minimax{};
        MCTSEngine::Config mcts{};
    };

    AIPlayer();

    [[nodiscard]] Move select_move(const Board &board, int level) const;
    [[nodiscard]] std::optional<MinimaxEngine::SearchResult> minimax_result(const Board &board, int level) const;
    [[nodiscard]] std::optional<MCTSEngine::SearchResult> mcts_result(const Board &board, int level) const;

    void set_level_config(int level, LevelConfig config);
    [[nodiscard]] const LevelConfig &get_level_config(int level) const;

private:
    std::map<int, LevelConfig> level_configs_;
};

#endif // GOGAME_AI_PLAYER_H

