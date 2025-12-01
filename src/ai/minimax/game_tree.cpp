#include "game_tree.h"

#include <sstream>

int GameTree::total_nodes() const {
    return count_nodes(root);
}

int GameTree::pruned_nodes() const {
    return count_pruned_nodes(root);
}

std::vector<Move> GameTree::principal_variation() const {
    return extract_principal_variation(root);
}

std::string GameTree::to_json() const {
    std::ostringstream oss;
    oss << node_to_json(root);
    return oss.str();
}

int GameTree::count_nodes(const GameTreeNode &node) const {
    int total = 1;
    for (const auto &child : node.children) {
        total += count_nodes(child);
    }
    return total;
}

int GameTree::count_pruned_nodes(const GameTreeNode &node) const {
    int total = node.pruned ? 1 : 0;
    for (const auto &child : node.children) {
        total += count_pruned_nodes(child);
    }
    return total;
}

std::vector<Move> GameTree::extract_principal_variation(const GameTreeNode &node) const {
    std::vector<Move> pv;
    const GameTreeNode *current = &node;
    while (!current->children.empty()) {
        const GameTreeNode *best_child = &current->children.front();
        for (const auto &child : current->children) {
            if (child.evaluation > best_child->evaluation) {
                best_child = &child;
            }
        }
        pv.push_back(best_child->move);
        current = best_child;
    }
    return pv;
}

std::string GameTree::node_to_json(const GameTreeNode &node) const {
    std::ostringstream oss;
    oss << "{";
    oss << "\"evaluation\":" << node.evaluation << ",";
    oss << "\"depth\":" << node.depth << ",";
    oss << "\"pruned\":" << (node.pruned ? "true" : "false") << ",";
    oss << "\"children\":[";
    for (std::size_t i = 0; i < node.children.size(); ++i) {
        oss << node_to_json(node.children[i]);
        if (i + 1 < node.children.size()) {
            oss << ",";
        }
    }
    oss << "]}";
    return oss.str();
}

