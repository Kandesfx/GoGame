#ifndef MCTS_ENGINE_H
#define MCTS_ENGINE_H

#include "../../game/board.h"
#include "mcts_node.h"
#include <memory>
#include <vector>

class MCTSEngine {
public:
    struct Config {
        int num_playouts;
        double time_limit_seconds;
        double ucb_constant;
        bool use_heuristics;
        bool parallel;
        int num_threads;

        constexpr Config(int playouts = 3000,
                         double time_limit = 0.0,
                         double ucb = 1.41421356237,
                         bool heuristics = false,
                         bool parallel_search = false,
                         int threads = 1)
            : num_playouts(playouts),
              time_limit_seconds(time_limit),
              ucb_constant(ucb),
              use_heuristics(heuristics),
              parallel(parallel_search),
              num_threads(threads) {}
    };

    struct MoveStats {
        Move move;
        int visits = 0;
        double win_rate = 0.0;
        double ucb_value = 0.0;
    };

    struct SearchResult {
        Move best_move;
        double win_rate = 0.0;
        int total_visits = 0;
        double search_time = 0.0;
        std::vector<MoveStats> top_moves;
    };

    explicit MCTSEngine(const Config &config = Config{});

    SearchResult search(const Board &board, Color to_move);
    std::vector<MoveStats> get_move_statistics() const;

private:
    Config config_;
    std::unique_ptr<MCTSNode> root_;
    int total_playouts_ = 0;

    MCTSNode *selection(MCTSNode *node, Board &board);
    MCTSNode *expansion(MCTSNode *node, Board &board);
    double simulation(Board &board, Color to_move);
    void backpropagation(MCTSNode *node, double result);

    double calculate_ucb(const MCTSNode *node, int parent_visits) const;
    MCTSNode::Child *best_child_ucb(MCTSNode *node) const;
    const MCTSNode::Child *best_child_robust(const MCTSNode *node) const;

    void parallel_search(const Board &board, Color to_move);

    MoveStats make_move_stats(const MCTSNode::Child &child) const;
};

#endif // MCTS_ENGINE_H

