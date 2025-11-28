#include "minimax_engine.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <stdexcept>

#include "move_ordering.h"

MinimaxEngine::MinimaxEngine(const Config &config)
    : config_(config),
      evaluator_(config.board_size),
      transposition_table_(config.use_transposition ? 1'000'000 : 0),
      nodes_searched_(0),
      nodes_pruned_(0) {}

MinimaxEngine::SearchResult MinimaxEngine::search(const Board &board, Color to_move) {
    nodes_searched_ = 0;
    nodes_pruned_ = 0;

    if (config_.use_transposition) {
        transposition_table_.clear();
    }

    auto start_time = std::chrono::steady_clock::now();

    Board working_board = board;
    Move best_move{};
    float evaluation = minimax(working_board,
                               config_.max_depth,
                               -INFINITY_VALUE,
                               INFINITY_VALUE,
                               to_move,
                               &best_move);

    auto end_time = std::chrono::steady_clock::now();
    double elapsed_seconds =
        std::chrono::duration<double>(end_time - start_time).count();

    std::vector<Move> principal_variation{};

    return SearchResult{
        .best_move = best_move,
        .evaluation = evaluation,
        .nodes_searched = nodes_searched_,
        .nodes_pruned = nodes_pruned_,
        .search_time = elapsed_seconds,
        .principal_variation = principal_variation,
    };
}

GameTree MinimaxEngine::build_game_tree(const Board &board, int depth) {
    GameTree tree;
    tree.root.depth = 0;
    tree.root.move = Move{};
    tree.root.pruned = false;

    Board working = board;
    const Color maximizing_player = board.current_player();
    const int actual_depth = std::max(0, depth);

    tree.root.evaluation = build_tree_recursive(working,
                                                actual_depth,
                                                -INFINITY_VALUE,
                                                INFINITY_VALUE,
                                                maximizing_player,
                                                tree.root);
    return tree;
}

float MinimaxEngine::minimax(Board &board,
                             int depth,
                             float alpha,
                             float beta,
                             Color maximizing_player,
                             Move *best_move_out) {
    nodes_searched_++;

    if (is_cutoff(board, depth)) {
        return evaluate_position(board, maximizing_player);
    }

    if (config_.use_transposition) {
        const std::uint64_t hash = board.zobrist_hash();
        const auto entry = transposition_table_.lookup(hash);
        if (entry.valid && entry.depth >= depth) {
            if (best_move_out) {
                *best_move_out = entry.best_move;
            }
            return entry.evaluation;
        }
    }

    const Color current_player = board.current_player();
    std::vector<Move> moves = get_ordered_moves(board, current_player);

    if (moves.empty()) {
        return evaluate_position(board, maximizing_player);
    }

    float best_value =
        (current_player == maximizing_player) ? -INFINITY_VALUE : INFINITY_VALUE;
    Move local_best_move{};

    for (const auto &move : moves) {
        Board::UndoInfo undo_info = board.make_move(move);

        Move child_best{};
        const float value = minimax(board,
                                    depth - 1,
                                    alpha,
                                    beta,
                                    maximizing_player,
                                    &child_best);

        board.undo_move(undo_info);

        if (current_player == maximizing_player) {
            if (value > best_value) {
                best_value = value;
                local_best_move = move;
            }
            if (config_.use_alpha_beta) {
                alpha = std::max(alpha, value);
                if (alpha >= beta) {
                    nodes_pruned_++;
                    break;
                }
            }
        } else {
            if (value < best_value) {
                best_value = value;
                local_best_move = move;
            }
            if (config_.use_alpha_beta) {
                beta = std::min(beta, value);
                if (beta <= alpha) {
                    nodes_pruned_++;
                    break;
                }
            }
        }
    }

    if (config_.use_transposition) {
        const std::uint64_t hash = board.zobrist_hash();
        transposition_table_.store(hash, depth, best_value, local_best_move);
    }

    if (best_move_out) {
        *best_move_out = local_best_move;
    }

    return best_value;
}

std::vector<Move> MinimaxEngine::get_ordered_moves(const Board &board, Color player) {
    std::vector<Move> moves = board.get_legal_moves(player);
    if (config_.use_move_ordering) {
        MoveOrdering::order_moves(moves, board, player);
    }
    return moves;
}

bool MinimaxEngine::is_cutoff(const Board &board, int depth) const {
    if (depth <= 0) {
        return true;
    }
    if (board.is_game_over()) {
        return true;
    }
    return false;
}

float MinimaxEngine::evaluate_position(const Board &board, Color player) {
    return evaluator_.evaluate(board, player);
}

float MinimaxEngine::build_tree_recursive(Board &board,
                                          int depth,
                                          float alpha,
                                          float beta,
                                          Color maximizing_player,
                                          GameTreeNode &node) {
    if (depth == 0 || board.is_game_over()) {
        const float eval = evaluate_position(board, maximizing_player);
        node.evaluation = eval;
        node.children.clear();
        return eval;
    }

    const Color current_player = board.current_player();
    std::vector<Move> moves = get_ordered_moves(board, current_player);

    if (moves.empty()) {
        const float eval = evaluate_position(board, maximizing_player);
        node.evaluation = eval;
        node.children.clear();
        return eval;
    }

    node.children.clear();
    node.children.reserve(moves.size());

    float best_value = (current_player == maximizing_player) ? -INFINITY_VALUE : INFINITY_VALUE;

    for (std::size_t i = 0; i < moves.size(); ++i) {
        const Move &move = moves[i];

        GameTreeNode child;
        child.move = move;
        child.depth = node.depth + 1;
        child.pruned = false;

        Board::UndoInfo undo = board.make_move(move);
        const float child_value = build_tree_recursive(board,
                                                       depth - 1,
                                                       alpha,
                                                       beta,
                                                       maximizing_player,
                                                       child);
        board.undo_move(undo);

        child.evaluation = child_value;
        node.children.push_back(std::move(child));

        if (current_player == maximizing_player) {
            best_value = std::max(best_value, child_value);
            alpha = std::max(alpha, child_value);
        } else {
            best_value = std::min(best_value, child_value);
            beta = std::min(beta, child_value);
        }

        if (config_.use_alpha_beta && beta <= alpha) {
            for (std::size_t remaining = i + 1; remaining < moves.size(); ++remaining) {
                GameTreeNode pruned_node;
                pruned_node.move = moves[remaining];
                pruned_node.depth = node.depth + 1;
                pruned_node.pruned = true;
                pruned_node.evaluation = (current_player == maximizing_player) ? alpha : beta;
                node.children.push_back(std::move(pruned_node));
            }
            break;
        }
    }

    node.evaluation = best_value;
    return best_value;
}

