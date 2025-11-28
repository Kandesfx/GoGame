#ifndef MINIMAX_GAME_TREE_H
#define MINIMAX_GAME_TREE_H

#include <string>
#include <vector>

#include "../../game/board.h"

struct GameTreeNode {
    Move move{};
    float evaluation{0.0f};
    int depth{0};
    bool pruned{false};
    std::vector<GameTreeNode> children{};
};

class GameTree {
public:
    GameTreeNode root{};

    [[nodiscard]] int total_nodes() const;
    [[nodiscard]] int pruned_nodes() const;
    [[nodiscard]] std::vector<Move> principal_variation() const;
    [[nodiscard]] std::string to_json() const;

private:
    [[nodiscard]] int count_nodes(const GameTreeNode &node) const;
    [[nodiscard]] int count_pruned_nodes(const GameTreeNode &node) const;
    [[nodiscard]] std::vector<Move> extract_principal_variation(const GameTreeNode &node) const;
    [[nodiscard]] std::string node_to_json(const GameTreeNode &node) const;
};

#endif // MINIMAX_GAME_TREE_H

