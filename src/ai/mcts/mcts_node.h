#ifndef MCTS_NODE_H
#define MCTS_NODE_H

#include "../../game/board.h"
#include <memory>
#include <vector>

class MCTSNode {
public:
    struct Child {
        std::unique_ptr<MCTSNode> node;
        Move move;
    };

    explicit MCTSNode(const Board &board_state,
                      Color player_to_move,
                      MCTSNode *parent = nullptr);

    [[nodiscard]] Board board_state() const { return board_state_; }
    [[nodiscard]] Color player_to_move() const { return player_to_move_; }
    [[nodiscard]] int visits() const { return visits_; }
    [[nodiscard]] double wins() const { return wins_; }
    [[nodiscard]] double win_rate() const;
    [[nodiscard]] bool is_fully_expanded() const;
    [[nodiscard]] bool is_terminal() const { return is_terminal_; }

    [[nodiscard]] MCTSNode *parent() const { return parent_; }
    [[nodiscard]] std::vector<Child> &children() { return children_; }
    [[nodiscard]] const std::vector<Child> &children() const { return children_; }

    void add_child(std::unique_ptr<MCTSNode> child_node, const Move &move);
    void increment_visits();
    void add_win(double value);

    void mark_terminal(bool terminal) { is_terminal_ = terminal; }
    void set_untried_moves(std::vector<Move> moves);
    bool has_untried_moves() const;
    Move pop_untried_move();
    const std::vector<Move> &untried_moves() const { return untried_moves_; }
    void remove_untried_move(const Move &move);

private:
    Board board_state_;
    Color player_to_move_;
    MCTSNode *parent_;
    std::vector<Child> children_;
    std::vector<Move> untried_moves_;
    int visits_ = 0;
    double wins_ = 0.0;
    bool is_terminal_ = false;
};

#endif // MCTS_NODE_H

