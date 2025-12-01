#include "../src/ai/mcts/mcts_engine.h"

#include <cassert>

int main() {
    Board board(3);

    MCTSEngine::Config config(1000, 0.0, 1.41421356237, true, false, 1);
    MCTSEngine engine(config);
    auto result = engine.search(board, Color::Black);

    assert(result.total_visits > 0);
    assert(result.best_move.is_valid());
    assert(!result.best_move.is_pass());

    return 0;
}

