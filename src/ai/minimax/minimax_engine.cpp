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
      nodes_pruned_(0),
      time_limit_exceeded_(false) {}

MinimaxEngine::SearchResult MinimaxEngine::search(const Board &board, Color to_move) {
    nodes_searched_ = 0;
    nodes_pruned_ = 0;
    time_limit_exceeded_ = false;

    if (config_.use_transposition) {
        transposition_table_.clear();
    }

    search_start_time_ = std::chrono::steady_clock::now();
    auto start_time = search_start_time_;

    Board working_board = board;
    Move best_move{};
    float evaluation = 0.0f;
    
    // Iterative deepening cho board lớn (19x19) để tối ưu thời gian
    // Bắt đầu từ depth 1 và tăng dần đến max_depth
    // Chỉ dùng khi max_depth >= 2 (level 3-4)
    if (config_.board_size >= 19 && config_.max_depth >= 2) {
        // Iterative deepening: tìm best move ở depth thấp trước, sau đó tăng depth
        for (int current_depth = 1; current_depth <= config_.max_depth && !time_limit_exceeded_; ++current_depth) {
            Move current_best{};
            float current_eval = minimax(working_board,
                                         current_depth,
                                         -INFINITY_VALUE,
                                         INFINITY_VALUE,
                                         to_move,
                                         &current_best);
            
            if (!time_limit_exceeded_ && current_best.is_valid()) {
                best_move = current_best;
                evaluation = current_eval;
            } else {
                // Nếu timeout hoặc move không hợp lệ, dùng kết quả từ depth trước đó
                break;
            }
        }
        
        // Fallback: nếu best_move không hợp lệ, chọn move đầu tiên
        if (!best_move.is_valid()) {
            auto legal_moves = working_board.get_legal_moves(to_move);
            if (!legal_moves.empty()) {
                best_move = legal_moves[0];
                evaluation = evaluate_position(working_board, to_move);
            }
        }
    } else {
        // Board nhỏ: dùng depth cố định
        evaluation = minimax(working_board,
                            config_.max_depth,
                            -INFINITY_VALUE,
                            INFINITY_VALUE,
                            to_move,
                            &best_move);
        
        // Fallback: nếu best_move không hợp lệ, chọn move đầu tiên
        if (!best_move.is_valid()) {
            auto legal_moves = working_board.get_legal_moves(to_move);
            if (!legal_moves.empty()) {
                best_move = legal_moves[0];
                evaluation = evaluate_position(working_board, to_move);
            }
        }
    }

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
    // Kiểm tra time limit
    if (config_.time_limit_seconds > 0.0) {
        auto current_time = std::chrono::steady_clock::now();
        double elapsed = std::chrono::duration<double>(current_time - search_start_time_).count();
        if (elapsed >= config_.time_limit_seconds) {
            time_limit_exceeded_ = true;
            // Trả về evaluation hiện tại khi hết thời gian
            return evaluate_position(board, maximizing_player);
        }
    }
    
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
        // Kiểm tra time limit trước mỗi move
        if (time_limit_exceeded_) {
            break;
        }
        
        Board::UndoInfo undo_info = board.make_move(move);

        Move child_best{};
        const float value = minimax(board,
                                    depth - 1,
                                    alpha,
                                    beta,
                                    maximizing_player,
                                    &child_best);

        board.undo_move(undo_info);
        
        // Kiểm tra time limit sau mỗi move
        if (time_limit_exceeded_) {
            break;
        }

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
    float base_eval = evaluator_.evaluate(board, player);
    
    // Thêm Quiescence Search cho endgame để tìm kiếm sâu hơn các nước quan trọng
    int move_count = board.get_move_count();
    int board_size = board.size();
    int max_moves = board_size * board_size * 2;  // Ước tính số nước tối đa
    
    // Trong endgame (sau 70% số nước), dùng quiescence search
    if (move_count >= static_cast<int>(max_moves * 0.7)) {
        // Quiescence search: tìm kiếm sâu hơn các nước quan trọng (capture, atari, etc.)
        float quiescence_eval = quiescence_search(
            const_cast<Board&>(board),
            base_eval - 50.0f,  // Alpha
            base_eval + 50.0f,  // Beta
            player,
            3  // Max depth cho quiescence
        );
        
        // Kết hợp đánh giá cơ bản và quiescence (ưu tiên quiescence trong endgame)
        return quiescence_eval * 0.7f + base_eval * 0.3f;
    }
    
    return base_eval;
}

float MinimaxEngine::quiescence_search(Board &board,
                                       float alpha,
                                       float beta,
                                       Color maximizing_player,
                                       int max_depth) {
    if (max_depth <= 0 || board.is_game_over()) {
        return evaluator_.evaluate(board, maximizing_player);
    }
    
    const Color current_player = board.current_player();
    float stand_pat = evaluator_.evaluate(board, maximizing_player);
    
    // Beta cutoff: nếu stand_pat đã đủ tốt, không cần tìm kiếm sâu hơn
    if (current_player == maximizing_player) {
        if (stand_pat >= beta) {
            return beta;
        }
        if (stand_pat > alpha) {
            alpha = stand_pat;
        }
    } else {
        if (stand_pat <= alpha) {
            return alpha;
        }
        if (stand_pat < beta) {
            beta = stand_pat;
        }
    }
    
    // Chỉ xem xét các nước quan trọng: capture, atari, và các nước gần quân
    std::vector<Move> moves = get_ordered_moves(board, current_player);
    
    // Lọc chỉ lấy các nước quan trọng (capture, atari, hoặc gần quân)
    std::vector<Move> important_moves;
    important_moves.reserve(std::min(10, static_cast<int>(moves.size())));  // Giới hạn số nước
    
    for (const auto &move : moves) {
        if (move.is_pass()) {
            continue;  // Bỏ qua pass trong quiescence
        }
        
        // Chỉ xem xét các nước quan trọng
        bool is_important = false;
        
        // Capture moves
        const Color opponent = opposite_color(move.color());
        for (const auto &dir : std::array<Point, 4>{{{1, 0}, {-1, 0}, {0, 1}, {0, -1}}}) {
            const int nx = move.x() + dir.x;
            const int ny = move.y() + dir.y;
            if (nx >= 0 && ny >= 0 && nx < board.size() && ny < board.size()) {
                if (board.at(nx, ny) == Board::stone_from_color(opponent)) {
                    const auto group = board.group_at(Point{nx, ny});
                    if (group.liberties.size() == 1) {
                        is_important = true;  // Capture move
                        break;
                    }
                }
            }
        }
        
        // Atari moves (cứu nhóm của mình)
        if (!is_important) {
            for (const auto &dir : std::array<Point, 4>{{{1, 0}, {-1, 0}, {0, 1}, {0, -1}}}) {
                const int nx = move.x() + dir.x;
                const int ny = move.y() + dir.y;
                if (nx >= 0 && ny >= 0 && nx < board.size() && ny < board.size()) {
                    if (board.at(nx, ny) == Board::stone_from_color(move.color())) {
                        const auto group = board.group_at(Point{nx, ny});
                        if (group.liberties.size() == 1) {
                            is_important = true;  // Save atari
                            break;
                        }
                    }
                }
            }
        }
        
        // Trong endgame, tất cả nước đều quan trọng (giới hạn số lượng)
        if (!is_important && important_moves.size() < 5) {
            is_important = true;
        }
        
        if (is_important) {
            important_moves.push_back(move);
            if (important_moves.size() >= 10) {
                break;  // Giới hạn số nước để tránh quá chậm
            }
        }
    }
    
    // Nếu không có nước quan trọng nào, trả về stand_pat
    if (important_moves.empty()) {
        return stand_pat;
    }
    
    float best_value = (current_player == maximizing_player) ? -INFINITY_VALUE : INFINITY_VALUE;
    
    for (const auto &move : important_moves) {
        Board::UndoInfo undo_info = board.make_move(move);
        
        const float value = quiescence_search(
            board,
            alpha,
            beta,
            maximizing_player,
            max_depth - 1
        );
        
        board.undo_move(undo_info);
        
        if (current_player == maximizing_player) {
            best_value = std::max(best_value, value);
            alpha = std::max(alpha, value);
            if (alpha >= beta) {
                break;  // Beta cutoff
            }
        } else {
            best_value = std::min(best_value, value);
            beta = std::min(beta, value);
            if (beta <= alpha) {
                break;  // Alpha cutoff
            }
        }
    }
    
    return best_value;
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

