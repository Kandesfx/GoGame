#ifndef MINIMAX_MOVE_ORDERING_H
#define MINIMAX_MOVE_ORDERING_H

#include <vector>

#include "../../game/board.h"

class MoveOrdering {
public:
    static void order_moves(std::vector<Move> &moves, const Board &board, Color player);

private:
    static float score_move(const Move &move, const Board &board, Color player);
    static bool is_capturing_move(const Move &move, const Board &board);
    static bool saves_atari(const Move &move, const Board &board, Color player);
    static float position_value(const Move &move, int board_size);
};

#endif // MINIMAX_MOVE_ORDERING_H

