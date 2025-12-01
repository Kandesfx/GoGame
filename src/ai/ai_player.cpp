#include "ai_player.h"

#include <stdexcept>
#include <random>
#include <vector>
#include <algorithm>

namespace {

MinimaxEngine::Config default_minimax_config(int depth) {
    MinimaxEngine::Config config{};
    config.max_depth = depth;
    // Bật tính năng bổ trợ cho depth >= 2 (thay vì >= 3)
    config.use_alpha_beta = depth >= 2;
    config.use_move_ordering = depth >= 2;
    config.use_transposition = depth >= 3;  // Transposition chỉ khi depth >= 3
    config.time_limit_seconds = 0.0;
    config.board_size = 9;  // Sẽ được điều chỉnh theo board size thực tế
    return config;
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
    // Level 1 (Dễ): Minimax depth 1 (rất yếu) + random + mistake rate
    LevelConfig level1{
        Algorithm::Minimax,
        default_minimax_config(1),  // Depth 1 cho bàn cờ 9x9
        {}
    };

    // Level 2 (Trung Bình): Minimax depth 2 (cho bàn cờ 9x9)
    LevelConfig level2{
        Algorithm::Minimax,
        default_minimax_config(2),  // Depth 2 cho bàn cờ 9x9
        {}
    };

    // Level 3 (Khó): Minimax với depth cao và đầy đủ tính năng bổ trợ
    LevelConfig level3{
        Algorithm::Minimax,
        default_minimax_config(4),  // Depth 4 cho bàn cờ 9x9, tự động điều chỉnh theo board size
        {}
    };

    // Level 4 (Siêu Khó): Minimax với depth rất cao và đầy đủ tính năng bổ trợ
    LevelConfig level4{
        Algorithm::Minimax,
        default_minimax_config(5),  // Depth 5 cho bàn cờ 9x9, tự động điều chỉnh theo board size
        {}
    };

    level_configs_.emplace(1, level1);
    level_configs_.emplace(2, level2);
    level_configs_.emplace(3, level3);
    level_configs_.emplace(4, level4);
}

Move AIPlayer::select_move(const Board &board, int level) const {
    const auto &config = get_level_config(level);
    
    // Level 1: Thêm randomness và mistake rate để AI yếu hơn
    if (level == 1) {
        static std::random_device rd;
        static std::mt19937 gen(rd());
        std::uniform_real_distribution<> dis(0.0, 1.0);
        
        // 30% chance chọn move ngẫu nhiên thay vì best move
        if (dis(gen) < 0.30) {
            // Lấy tất cả legal moves (đã bao gồm pass move)
            std::vector<Move> legal_moves = board.get_legal_moves(board.current_player());
            
            if (!legal_moves.empty()) {
                std::uniform_int_distribution<> move_dis(0, legal_moves.size() - 1);
                return legal_moves[move_dis(gen)];
            }
        }
        
        // 20% chance chọn move không tối ưu (random từ tất cả legal moves)
        if (dis(gen) < 0.20) {
            std::vector<Move> legal_moves = board.get_legal_moves(board.current_player());
            
            if (legal_moves.size() > 1) {
                // Chọn random từ tất cả legal moves (không phải best move)
                std::uniform_int_distribution<> move_dis(0, legal_moves.size() - 1);
                return legal_moves[move_dis(gen)];
            }
        }
    }
    
    // Level 3-4: Minimax với depth tự động điều chỉnh theo board size
    if (level >= 3 && config.algorithm == Algorithm::Minimax) {
        const int board_size = board.size();
        MinimaxEngine::Config minimax_config = config.minimax;
        minimax_config.board_size = board_size;
        
        // Điều chỉnh depth theo board size để đảm bảo không quá chậm
        int target_depth = minimax_config.max_depth;
        
        if (board_size > 9) {
            // Giảm depth cho bàn cờ lớn hơn
            if (board_size <= 13) {
                // 13x13: giảm depth 1
                target_depth = std::max(2, target_depth - 1);
            } else {
                // 19x19: giảm depth 2
                target_depth = std::max(2, target_depth - 2);
            }
        }
        
        minimax_config.max_depth = target_depth;
        
        // Bật tất cả tính năng bổ trợ để Minimax mạnh nhất có thể
        minimax_config.use_alpha_beta = target_depth >= 2;  // Alpha-Beta pruning
        minimax_config.use_move_ordering = target_depth >= 2;  // Move ordering
        minimax_config.use_transposition = target_depth >= 3;  // Transposition table (chỉ khi depth >= 3)
        minimax_config.time_limit_seconds = 0.0;  // Không giới hạn thời gian
        
        MinimaxEngine engine(minimax_config);
        auto result = engine.search(board, board.current_player());
        return result.best_move;
    }
    
    switch (config.algorithm) {
        case Algorithm::Minimax: {
            MinimaxEngine engine(config.minimax);
            auto result = engine.search(board, board.current_player());
            return result.best_move;
        }
        case Algorithm::MCTS: {
            MCTSEngine engine(config.mcts);
            auto result = engine.search(board, board.current_player());
            return result.best_move;
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

