#include "mcts_node.h"

#include <algorithm>
#include <random>
#include <stdexcept>

using gogame::Board;
using gogame::Color;
using gogame::Move;

namespace {
std::vector<Move> shuffle_moves(std::vector<Move> moves) {
    std::mt19937 rng{std::random_device{}()};
    std::shuffle(moves.begin(), moves.end(), rng);
    return moves;
}
} // namespace

MCTSNode::MCTSNode(const Board &board_state,
                   Color player_to_move,
                   MCTSNode *parent)
    : board_state_(board_state),
      player_to_move_(player_to_move),
      parent_(parent) {
}

double MCTSNode::win_rate() const {
    if (visits_ == 0) {
        return 0.0;
    }
    return wins_ / static_cast<double>(visits_);
}

bool MCTSNode::is_fully_expanded() const {
    return untried_moves_.empty();
}

void MCTSNode::add_child(std::unique_ptr<MCTSNode> child_node, const Move &move) {
    children_.push_back(Child{std::move(child_node), move});
}

void MCTSNode::increment_visits() {
    visits_ += 1;
}

void MCTSNode::add_win(double value) {
    wins_ += value;
}

void MCTSNode::set_untried_moves(std::vector<Move> moves) {
    untried_moves_ = shuffle_moves(std::move(moves));
}

bool MCTSNode::has_untried_moves() const {
    return !untried_moves_.empty();
}

Move MCTSNode::pop_untried_move() {
    if (untried_moves_.empty()) {
        throw std::runtime_error("No more untried moves");
    }
    Move move = untried_moves_.back();
    untried_moves_.pop_back();
    return move;
}

void MCTSNode::remove_untried_move(const Move &move) {
    untried_moves_.erase(
        std::remove_if(untried_moves_.begin(), untried_moves_.end(),
                      [&move](const Move &m) {
                          return m.x() == move.x() && m.y() == move.y() && 
                                 m.color() == move.color() && m.is_pass() == move.is_pass();
                      }),
        untried_moves_.end()
    );
}

