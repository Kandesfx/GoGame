#include "../src/ai/minimax/minimax_engine.h"

#include <cassert>

int main() {
    Board board(3);

    MinimaxEngine::Config config{};
    config.max_depth = 1;
    config.use_alpha_beta = true;
    config.use_move_ordering = true;
    config.use_transposition = false;
    config.time_limit_seconds = 0.0;
    config.board_size = 3;

    MinimaxEngine engine(config);
    auto result = engine.search(board, Color::Black);
    assert(result.best_move.is_valid());
    assert(!result.best_move.is_pass());
    assert(result.best_move.color() == Color::Black);

    return 0;
}

