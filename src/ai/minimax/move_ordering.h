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
    static float connection_bonus(const Move &move, const Board &board, Color player);
    static float influence_bonus(const Move &move, const Board &board, Color player);
    static float opening_territory_bonus(const Move &move, const Board &board, Color player);
    static float endgame_territory_bonus(const Move &move, const Board &board, Color player);
    static float endgame_eye_bonus(const Move &move, const Board &board, Color player);
    
    // Các hàm mới để cải thiện AI
    static float expansion_bonus(const Move &move, const Board &board, Color player);
    static float opponent_response_bonus(const Move &move, const Board &board, Color player);
    
    // Các hàm mới để cải thiện khai cuộc
    static float star_point_bonus(const Move &move, int board_size);
    static float clustering_penalty(const Move &move, const Board &board, Color player);
    static float territory_diversification_bonus(const Move &move, const Board &board, Color player);
};

#endif // MINIMAX_MOVE_ORDERING_H

