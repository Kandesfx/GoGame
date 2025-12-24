#include "ai_player.h"
#include "opening_book.h"

#include <stdexcept>
#include <random>
#include <vector>
#include <algorithm>

namespace {

MinimaxEngine::Config default_minimax_config(int depth, int board_size = 9, double time_limit = 0.0) {
    MinimaxEngine::Config config{};
    config.max_depth = depth;
    // Bật tính năng bổ trợ cho depth >= 2 (thay vì >= 3)
    config.use_alpha_beta = depth >= 2;
    config.use_move_ordering = depth >= 2;
    config.use_transposition = depth >= 3;  // Transposition chỉ khi depth >= 3
    config.time_limit_seconds = time_limit;
    config.board_size = board_size;
    return config;
}

// Config động dựa trên board size và level
MinimaxEngine::Config adaptive_minimax_config(int level, int board_size, double time_limit) {
    // Giảm depth cho board lớn để tránh timeout
    int depth;
    if (board_size >= 19) {
        // Board 19x19: tăng thêm 1 ply cho siêu khó
        depth = (level == 1) ? 1
              : (level == 2) ? 2
              : (level == 3) ? 3  // khó: +1 so với trước
              : 4;                // siêu khó: +1 depth
    } else if (board_size >= 13) {
        // Board 13x13
        depth = (level == 1) ? 1
              : (level == 2) ? 2
              : (level == 3) ? 2
              : 3;
    } else {
        // Board 9x9: giữ nguyên depth
        depth = (level == 1) ? 1
              : (level == 2) ? 2
              : (level == 3) ? 2
              : 3;
    }
    
    return default_minimax_config(depth, board_size, time_limit);
}

// MCTS config function removed - Level 3-4 now use Minimax only
// MCTSEngine::Config default_mcts_config(int playouts, bool use_heuristics, int threads = 1) {
//     return MCTSEngine::Config{
//         playouts,
//         0.0,
//         1.41421356237,
//         use_heuristics,
//         threads > 1,
//         threads,
//     };
// }

} // namespace

AIPlayer::AIPlayer() {
    // Config mặc định (sẽ được điều chỉnh động trong select_move dựa trên board size)
    // Level 1 (Dễ): Minimax depth 1 (rất yếu) + random + mistake rate
    LevelConfig level1{
        Algorithm::Minimax,
        default_minimax_config(1, 9, 0.0),  // Depth 1, sẽ điều chỉnh theo board size
        {}
    };

    // Level 2 (Trung Bình): Minimax depth 2 (cho bàn cờ 9x9)
    LevelConfig level2{
        Algorithm::Minimax,
        default_minimax_config(2, 9, 0.0),  // Depth 2, sẽ điều chỉnh theo board size
        {}
    };

    // Level 3 (Khó): Minimax depth 2 (nhanh, tránh timeout)
    // Note: Depth sẽ được điều chỉnh động theo board size
    LevelConfig level3{
        Algorithm::Minimax,
        default_minimax_config(2, 9, 0.0),  // Depth 2, sẽ điều chỉnh theo board size
        {}
    };

    // Level 4 (Siêu Khó): Minimax depth 3 (nhanh hơn MCTS, tránh timeout)
    // Note: Depth sẽ được điều chỉnh động theo board size
    LevelConfig level4{
        Algorithm::Minimax,
        default_minimax_config(3, 9, 0.0),  // Depth 3, sẽ điều chỉnh theo board size
        {}
    };

    level_configs_.emplace(1, level1);
    level_configs_.emplace(2, level2);
    level_configs_.emplace(3, level3);
    level_configs_.emplace(4, level4);
}

Move AIPlayer::select_move(const Board &board, int level) const {
    // QUAN TRỌNG: Kiểm tra game đã kết thúc chưa
    if (board.is_game_over()) {
        // Game đã kết thúc (2 consecutive passes), AI phải pass
        return Move::Pass(board.current_player());
    }
    
    const auto &base_config = get_level_config(level);
    int board_size = board.size();
    int move_number = board.get_move_count() + 1;  // +1 vì đây là nước tiếp theo
    
    // Lấy tất cả legal moves để kiểm tra
    std::vector<Move> all_legal_moves = board.get_legal_moves(board.current_player());
    
    // Nếu chỉ còn pass move, AI phải pass
    if (all_legal_moves.size() == 1 && all_legal_moves[0].is_pass()) {
        return Move::Pass(board.current_player());
    }
    
    // Kiểm tra xem có nước hợp lý nào không (không phải pass)
    // Đếm số nước không phải pass
    int non_pass_moves = 0;
    for (const auto &move : all_legal_moves) {
        if (!move.is_pass()) {
            non_pass_moves++;
        }
    }
    
    // Nếu không còn nước hợp lý nào (chỉ còn pass), AI nên pass
    if (non_pass_moves == 0) {
        return Move::Pass(board.current_player());
    }
    
    // Sử dụng Opening Book cho khai cuộc (Level 2-4)
    // Level 1 không dùng opening book để giữ tính ngẫu nhiên
    if (level >= 2 && level <= 4) {
        static OpeningBook opening_book;
        auto opening_move = opening_book.find_move(board, board.current_player(), move_number);
        if (opening_move.has_value()) {
            return opening_move.value();
        }
    }
    
    // Level 1: Thêm randomness và mistake rate để AI yếu hơn (đã giảm từ 50% xuống 20-25%)
    if (level == 1) {
        static std::random_device rd;
        static std::mt19937 gen(rd());
        std::uniform_real_distribution<> dis(0.0, 1.0);
        
        // 15% chance chọn move ngẫu nhiên thay vì best move (giảm từ 30%)
        if (dis(gen) < 0.15) {
            // Lấy tất cả legal moves (đã bao gồm pass move)
            std::vector<Move> legal_moves = board.get_legal_moves(board.current_player());
            
            if (!legal_moves.empty()) {
                std::uniform_int_distribution<> move_dis(0, legal_moves.size() - 1);
                return legal_moves[move_dis(gen)];
            }
        }
        
        // 10% chance chọn move không tối ưu (giảm từ 20%)
        // Tổng mistake rate: 15% + 10% = 25% (giảm từ 50%)
        if (dis(gen) < 0.10) {
            std::vector<Move> legal_moves = board.get_legal_moves(board.current_player());
            
            if (legal_moves.size() > 1) {
                // Chọn random từ tất cả legal moves (không phải best move)
                std::uniform_int_distribution<> move_dis(0, legal_moves.size() - 1);
                return legal_moves[move_dis(gen)];
            }
        }
    }
    
    // Level 3-4: Minimax depth 3-4 (nhanh hơn MCTS)
    // Note: ML integration (PolicyNet) được xử lý ở Python layer (ai_wrapper.py)
    
    switch (base_config.algorithm) {
        case Algorithm::Minimax: {
            // Điều chỉnh config động dựa trên board size
            // Board lớn hơn = giảm depth và thêm time limit
            double time_limit = 0.0;
            if (board_size >= 19) {
                // Board 19x19: nâng depth => tăng time limit một chút
                time_limit = (level == 3) ? 10.0 : (level == 4) ? 14.0 : 0.0;
            } else if (board_size >= 13) {
                time_limit = (level == 3) ? 6.0 : (level == 4) ? 10.0 : 0.0;
            }
            
            MinimaxEngine::Config adaptive_config = adaptive_minimax_config(level, board_size, time_limit);
            MinimaxEngine engine(adaptive_config);
            auto result = engine.search(board, board.current_player());
            
            // Kiểm tra xem best_move có hợp lệ không
            if (!result.best_move.is_valid() || result.best_move.is_pass()) {
                // Nếu Minimax trả về pass hoặc move không hợp lệ, kiểm tra lại
                // Có thể game đã kết thúc hoặc không còn nước tốt
                if (board.is_game_over() || non_pass_moves == 0) {
                    return Move::Pass(board.current_player());
                }
                // Nếu Minimax chọn pass nhưng vẫn còn nước, có thể pass là best move
                // Nhưng để an toàn, kiểm tra lại game state
                if (result.best_move.is_pass()) {
                    return result.best_move;  // Pass là best move
                }
            }
            
            // Đảm bảo move hợp lệ
            if (result.best_move.is_valid() && board.is_legal_move(result.best_move)) {
                return result.best_move;
            }
            
            // Fallback: nếu move không hợp lệ, chọn move đầu tiên hợp lệ (không phải pass)
            for (const auto &move : all_legal_moves) {
                if (!move.is_pass() && board.is_legal_move(move)) {
                    return move;
                }
            }
            
            // Cuối cùng, nếu không còn nước nào, pass
            return Move::Pass(board.current_player());
        }
        case Algorithm::MCTS: {
            MCTSEngine engine(base_config.mcts);
            auto result = engine.search(board, board.current_player());
            
            // Kiểm tra tương tự như Minimax
            if (!result.best_move.is_valid() || result.best_move.is_pass()) {
                if (board.is_game_over() || non_pass_moves == 0) {
                    return Move::Pass(board.current_player());
                }
                if (result.best_move.is_pass()) {
                    return result.best_move;
                }
            }
            
            if (result.best_move.is_valid() && board.is_legal_move(result.best_move)) {
                return result.best_move;
            }
            
            // Fallback
            for (const auto &move : all_legal_moves) {
                if (!move.is_pass() && board.is_legal_move(move)) {
                    return move;
                }
            }
            
            return Move::Pass(board.current_player());
        }
        default:
            throw std::runtime_error("Unsupported algorithm");
    }
}

std::optional<MinimaxEngine::SearchResult> AIPlayer::minimax_result(const Board &board, int level) const {
    const auto &config = get_level_config(level);
    if (config.algorithm != Algorithm::Minimax) {
        return std::nullopt;
    }
    MinimaxEngine engine(config.minimax);
    return engine.search(board, board.current_player());
}

std::optional<MCTSEngine::SearchResult> AIPlayer::mcts_result(const Board &board, int level) const {
    const auto &config = get_level_config(level);
    if (config.algorithm != Algorithm::MCTS) {
        return std::nullopt;
    }
    MCTSEngine engine(config.mcts);
    return engine.search(board, board.current_player());
}

void AIPlayer::set_level_config(int level, LevelConfig config) {
    level_configs_[level] = std::move(config);
}

const AIPlayer::LevelConfig &AIPlayer::get_level_config(int level) const {
    auto it = level_configs_.find(level);
    if (it == level_configs_.end()) {
        throw std::out_of_range("AI level configuration not found");
    }
    return it->second;
}

