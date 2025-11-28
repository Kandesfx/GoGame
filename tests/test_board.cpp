#include "../src/game/board.h"

#include <cassert>
#include <vector>

int main() {
    Board board(9);
    std::vector<Board::UndoInfo> history;

    auto play = [&](const Move &move) {
        history.push_back(board.make_move(move));
    };

    // Setup: white stone at (1,1) in atari, black to move.
    play(Move(1, 0, Color::Black));
    play(Move(1, 1, Color::White));
    play(Move(0, 1, Color::Black));
    play(Move(2, 2, Color::White));
    play(Move(1, 2, Color::Black));
    play(Move(2, 0, Color::White));

    Board::UndoInfo capture = board.make_move(Move(2, 1, Color::Black));

    assert(board.at(1, 1) == Stone::Empty);
    assert(board.get_prisoners(Color::Black) == 1);

    board.undo_move(capture);
    assert(board.at(1, 1) == Stone::White);
    assert(board.get_prisoners(Color::Black) == 0);

    while (!history.empty()) {
        board.undo_move(history.back());
        history.pop_back();
    }

    assert(board.at(1, 1) == Stone::Empty);
    assert(board.get_prisoners(Color::Black) == 0);
    return 0;
}

