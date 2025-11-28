#ifndef MINIMAX_ENGINE_H
#define MINIMAX_ENGINE_H

#include <limits>
#include <vector>

#include "../../game/board.h"
#include "game_tree.h"
#include "evaluator.h"
#include "transposition_table.h"

class MinimaxEngine {
public:
    struct SearchResult {
        Move best_move;
        float evaluation;
        int nodes_searched;
        int nodes_pruned;
        double search_time;
        std::vector<Move> principal_variation;
    };

    struct Config {
        int max_depth;
        bool use_alpha_beta;
        bool use_move_ordering;
        bool use_transposition;
        double time_limit_seconds;
        int board_size;
    };

private:
    Config config_;
    Evaluator evaluator_;
    TranspositionTable transposition_table_;

    int nodes_searched_;
    int nodes_pruned_;

    static constexpr float INFINITY_VALUE =
        std::numeric_limits<float>::infinity();
    static constexpr float MATE_SCORE = 100000.0f;

public:
    explicit MinimaxEngine(const Config &config);

    SearchResult search(const Board &board, Color to_move);

    GameTree build_game_tree(const Board &board, int depth);

private:
    float minimax(Board &board,
                  int depth,
                  float alpha,
                  float beta,
                  Color maximizing_player,
                  Move *best_move_out);

    std::vector<Move> get_ordered_moves(const Board &board, Color player);

    bool is_cutoff(const Board &board, int depth) const;

    float evaluate_position(const Board &board, Color player);

    float build_tree_recursive(Board &board,
                               int depth,
                               float alpha,
                               float beta,
                               Color maximizing_player,
                               GameTreeNode &node);
};

#endif // MINIMAX_ENGINE_H

