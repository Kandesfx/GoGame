#include "mcts_engine.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <random>

using gogame::Board;
using gogame::Color;
using gogame::Move;
using gogame::Stone;
using gogame::opposite_color;

namespace {

// Heuristic move evaluation for rollout
int evaluate_move_priority(const Board &board, const Move &move, Color to_move) {
    if (move.is_pass()) {
        return 0;  // Lowest priority
    }
    
    // Test move on temporary board
    Board temp = board;
    Board::UndoInfo undo{};
    
    // Create a test move with correct color
    Move test_move(move.x(), move.y(), to_move);
    
    // Check if move is legal first
    if (!temp.is_legal_move(test_move)) {
        return 0;
    }
    
    try {
        undo = temp.make_move(test_move);
    } catch (...) {
        return 0;
    }
    
    int priority = 1;  // Base priority
    
    // Check if move captures stones (high priority)
    if (!undo.captured.empty()) {
        priority += 1000;  // Capture is very important
        priority += static_cast<int>(undo.captured.size()) * 100;  // More captures = better
    }
    
    // Check if move puts opponent in atari (medium-high priority)
    const int x = move.x();
    const int y = move.y();
    const int size = board.size();
    const Color opponent = opposite_color(to_move);
    
    // Check neighbors for opponent groups with 1 liberty
    const int dx[] = {-1, 1, 0, 0};
    const int dy[] = {0, 0, -1, 1};
    
    for (int i = 0; i < 4; ++i) {
        const int nx = x + dx[i];
        const int ny = y + dy[i];
        
        if (nx >= 0 && nx < size && ny >= 0 && ny < size) {
            const Stone neighbor = temp.at(nx, ny);
            if (neighbor == Board::stone_from_color(opponent)) {
                // Check if this group has only 1 liberty now
                const auto group = temp.group_at({nx, ny});
                if (group.liberties.size() == 1) {
                    priority += 500;  // Atari is important
                }
            }
        }
    }
    
    // Prefer moves near existing stones (connection/extension)
    int nearby_friends = 0;
    for (int i = 0; i < 4; ++i) {
        const int nx = x + dx[i];
        const int ny = y + dy[i];
        
        if (nx >= 0 && nx < size && ny >= 0 && ny < size) {
            const Stone neighbor = board.at(nx, ny);  // Use original board
            if (neighbor == Board::stone_from_color(to_move)) {
                nearby_friends++;
            }
        }
    }
    priority += nearby_friends * 50;  // Connection bonus
    
    // Position evaluation: Prefer center, avoid corners
    const int center = size / 2;
    const int dist_from_center = std::abs(x - center) + std::abs(y - center);
    
    // Center bonus (stronger)
    if (dist_from_center < size / 3) {
        priority += 50;  // Center bonus (tăng từ 20)
    } else if (dist_from_center < size / 2) {
        priority += 20;  // Near center
    }
    
    // Corner penalty (tránh góc)
    const bool is_corner = (x == 0 || x == size - 1) && (y == 0 || y == size - 1);
    const bool is_edge = (x == 0 || x == size - 1 || y == 0 || y == size - 1);
    
    if (is_corner) {
        priority -= 100;  // Penalty cho góc (trừ điểm)
    } else if (is_edge && dist_from_center > size * 2 / 3) {
        priority -= 30;  // Penalty cho edge xa center
    }
    
    // Star points bonus (opening)
    if (size == 9) {
        const std::vector<std::pair<int, int>> star_points = {{2, 2}, {6, 2}, {2, 6}, {6, 6}, {4, 4}};
        for (const auto &sp : star_points) {
            if (x == sp.first && y == sp.second) {
                priority += 40;  // Star point bonus
                break;
            }
        }
    } else if (size == 19) {
        const std::vector<std::pair<int, int>> star_points = {
            {3, 3}, {3, 9}, {3, 15},
            {9, 3}, {9, 9}, {9, 15},
            {15, 3}, {15, 9}, {15, 15}
        };
        for (const auto &sp : star_points) {
            if (x == sp.first && y == sp.second) {
                priority += 40;  // Star point bonus
                break;
            }
        }
    }
    
    return priority;
}

// Heuristic rollout with move prioritization (optimized)
double heuristic_rollout(Board &board, Color to_move) {
    std::mt19937 rng{std::random_device{}()};
    int consecutive_passes = 0;
    int max_moves = board.size() * board.size() * 2;  // Safety limit
    int move_count = 0;

    while (!board.is_game_over() && consecutive_passes < 2 && move_count < max_moves) {
        std::vector<Move> moves = board.get_legal_moves(to_move);
        if (moves.empty()) {
            break;
        }

        Move selected_move;
        
        // Fast heuristic: chỉ evaluate nếu có nhiều moves (>10)
        // Với ít moves, dùng simple heuristics không cần test board
        if (moves.size() > 10) {
            // Quick evaluation: chỉ evaluate top 20 moves để tiết kiệm thời gian
            std::vector<std::pair<Move, int>> move_priorities;
            move_priorities.reserve(std::min(20, static_cast<int>(moves.size())));
            
            // Quick pass: tìm captures và atari trước (không cần test board)
            for (const auto &move : moves) {
                if (move.is_pass()) {
                    continue;
                }
                
                // Quick check: neighbors có opponent stones không?
                const int x = move.x();
                const int y = move.y();
                const int size = board.size();
                const Color opponent = opposite_color(to_move);
                int quick_priority = 1;
                
                const int dx[] = {-1, 1, 0, 0};
                const int dy[] = {0, 0, -1, 1};
                
                // Check for nearby opponent stones (potential capture/atari)
                for (int i = 0; i < 4; ++i) {
                    const int nx = x + dx[i];
                    const int ny = y + dy[i];
                    
                    if (nx >= 0 && nx < size && ny >= 0 && ny < size) {
                        const Stone neighbor = board.at(nx, ny);
                        if (neighbor == Board::stone_from_color(opponent)) {
                            quick_priority += 100;  // Potential capture/atari
                        } else if (neighbor == Board::stone_from_color(to_move)) {
                            quick_priority += 20;  // Connection
                        }
                    }
                }
                
                // Position evaluation: avoid corners, prefer center
                const bool is_corner = (x == 0 || x == size - 1) && (y == 0 || y == size - 1);
                const bool is_edge = (x == 0 || x == size - 1 || y == 0 || y == size - 1);
                const int center = size / 2;
                const int dist_from_center = std::abs(x - center) + std::abs(y - center);
                
                if (is_corner) {
                    quick_priority -= 50;  // Corner penalty (tránh góc)
                } else if (is_edge && dist_from_center > size * 2 / 3) {
                    quick_priority -= 20;  // Edge penalty (xa center)
                } else if (dist_from_center < size / 3) {
                    quick_priority += 30;  // Center bonus
                }
                
                move_priorities.emplace_back(move, quick_priority);
            }
            
            // Sort và chọn từ top
            if (!move_priorities.empty()) {
                std::sort(move_priorities.begin(), move_priorities.end(),
                          [](const auto &a, const auto &b) {
                              return a.second > b.second;
                          });
                
                const size_t top_n = std::max(static_cast<size_t>(3),
                                              move_priorities.size() / 3);
                std::uniform_int_distribution<size_t> dist(0, top_n - 1);
                selected_move = move_priorities[dist(rng)].first;
            } else {
                // Fallback: random
                std::uniform_int_distribution<size_t> dist(0, moves.size() - 1);
                selected_move = moves[dist(rng)];
            }
        } else {
            // Ít moves: dùng simple heuristics (nhanh hơn)
            // Ưu tiên moves gần quân ta
            std::vector<std::pair<Move, int>> quick_priorities;
            for (const auto &move : moves) {
                if (move.is_pass()) {
                    quick_priorities.emplace_back(move, 0);
                    continue;
                }
                
                int priority = 1;
                const int x = move.x();
                const int y = move.y();
                const int size = board.size();
                const int dx[] = {-1, 1, 0, 0};
                const int dy[] = {0, 0, -1, 1};
                
                for (int i = 0; i < 4; ++i) {
                    const int nx = x + dx[i];
                    const int ny = y + dy[i];
                    
                    if (nx >= 0 && nx < size && ny >= 0 && ny < size) {
                        const Stone neighbor = board.at(nx, ny);
                        if (neighbor == Board::stone_from_color(to_move)) {
                            priority += 50;
                        } else if (neighbor == Board::stone_from_color(opposite_color(to_move))) {
                            priority += 100;  // Potential capture
                        }
                    }
                }
                
                quick_priorities.emplace_back(move, priority);
            }
            
            std::sort(quick_priorities.begin(), quick_priorities.end(),
                      [](const auto &a, const auto &b) {
                          return a.second > b.second;
                      });
            
            const size_t top_n = std::min(static_cast<size_t>(5), quick_priorities.size());
            std::uniform_int_distribution<size_t> dist(0, top_n - 1);
            selected_move = quick_priorities[dist(rng)].first;
        }

        Board::UndoInfo ignore = board.make_move(selected_move);
        (void)ignore;

        if (selected_move.is_pass()) {
            consecutive_passes += 1;
        } else {
            consecutive_passes = 0;
        }

        to_move = opposite_color(to_move);
        move_count++;
    }

    // Improved scoring: prisoners + territory + influence + group safety
    const int black_prisoners = board.get_prisoners(Color::Black);
    const int white_prisoners = board.get_prisoners(Color::White);
    
    // Territory estimate: count empty points near our stones (improved)
    int black_territory = 0;
    int white_territory = 0;
    int black_influence = 0;
    int white_influence = 0;
    const int size = board.size();
    
    const int dx[] = {-1, 1, 0, 0, -1, -1, 1, 1};
    const int dy[] = {0, 0, -1, 1, -1, 1, -1, 1};
    
    for (int y = 0; y < size; ++y) {
        for (int x = 0; x < size; ++x) {
            if (board.at(x, y) != Stone::Empty) {
                continue;
            }
            
            // Count nearby stones (4 directions for territory)
            int black_nearby = 0;
            int white_nearby = 0;
            int black_influence_count = 0;
            int white_influence_count = 0;
            
            // Territory: immediate neighbors
            for (int i = 0; i < 4; ++i) {
                const int nx = x + dx[i];
                const int ny = y + dy[i];
                
                if (nx >= 0 && nx < size && ny >= 0 && ny < size) {
                    const Stone neighbor = board.at(nx, ny);
                    if (neighbor == Stone::Black) {
                        black_nearby++;
                    } else if (neighbor == Stone::White) {
                        white_nearby++;
                    }
                }
            }
            
            // Influence: all 8 directions (including diagonals)
            for (int i = 0; i < 8; ++i) {
                const int nx = x + dx[i];
                const int ny = y + dy[i];
                
                if (nx >= 0 && nx < size && ny >= 0 && ny < size) {
                    const Stone neighbor = board.at(nx, ny);
                    if (neighbor == Stone::Black) {
                        black_influence_count++;
                    } else if (neighbor == Stone::White) {
                        white_influence_count++;
                    }
                }
            }
            
            // Territory scoring
            if (black_nearby > white_nearby) {
                black_territory++;
            } else if (white_nearby > black_nearby) {
                white_territory++;
            }
            
            // Influence scoring (weighted)
            if (black_influence_count > white_influence_count) {
                black_influence += (black_influence_count - white_influence_count);
            } else if (white_influence_count > black_influence_count) {
                white_influence += (white_influence_count - black_influence_count);
            }
        }
    }
    
    // Group safety evaluation
    int black_safe_groups = 0;
    int white_safe_groups = 0;
    
    const auto black_groups = board.groups(Color::Black);
    const auto white_groups = board.groups(Color::White);
    
    for (const auto &group : black_groups) {
        // Group is safe if it has many liberties
        if (group.liberties.size() >= 3) {
            black_safe_groups += group.liberties.size();
        }
    }
    
    for (const auto &group : white_groups) {
        if (group.liberties.size() >= 3) {
            white_safe_groups += group.liberties.size();
        }
    }
    
    // Combined score: prisoners + territory + influence + safety
    const int black_score = black_prisoners * 2 +  // Prisoners worth more
                           black_territory +
                           black_influence / 2 +    // Influence bonus
                           black_safe_groups / 2;   // Safety bonus
    
    const int white_score = white_prisoners * 2 +
                           white_territory +
                           white_influence / 2 +
                           white_safe_groups / 2;
    
    const double score_diff = static_cast<double>(black_score - white_score);
    
    // Normalize to [0, 1] range
    const double max_possible_diff = size * size * 2;  // Adjusted for new scoring
    const double normalized = (score_diff / max_possible_diff + 1.0) / 2.0;
    
    return std::max(0.0, std::min(1.0, normalized));
}

double default_rollout(Board &board, Color to_move) {
    std::mt19937 rng{std::random_device{}()};
    int consecutive_passes = 0;

    while (!board.is_game_over() && consecutive_passes < 2) {
        std::vector<Move> moves = board.get_legal_moves(to_move);
        if (moves.empty()) {
            break;
        }

        std::uniform_int_distribution<std::size_t> dist(0, moves.size() - 1);
        const Move move = moves[dist(rng)];

        Board::UndoInfo ignore = board.make_move(move);
        (void)ignore;

        if (move.is_pass()) {
            consecutive_passes += 1;
        } else {
            consecutive_passes = 0;
        }

        to_move = gogame::opposite_color(to_move);
    }

    // Simple scoring: compare prisoners and territory heuristic.
    // For now use prisoners difference.
    const int my_prisoners = board.get_prisoners(Color::Black);
    const int opponent_prisoners = board.get_prisoners(Color::White);
    const double score = static_cast<double>(my_prisoners - opponent_prisoners);
    return score > 0 ? 1.0 : (score < 0 ? 0.0 : 0.5);
}

} // namespace

MCTSEngine::MCTSEngine(const Config &config)
    : config_(config) {
    if (config_.num_threads < 1) {
        config_.num_threads = 1;
    }
}

MCTSEngine::SearchResult MCTSEngine::search(const Board &board, Color to_move) {
    Board root_board = board;
    root_ = std::make_unique<MCTSNode>(root_board, to_move, nullptr);
    root_->set_untried_moves(board.get_legal_moves(to_move));
    root_->mark_terminal(board.is_game_over());
    total_playouts_ = 0;

    auto start_time = std::chrono::steady_clock::now();

    if (config_.parallel && config_.num_threads > 1) {
        parallel_search(board, to_move);
    } else {
        const int max_playouts = std::max(1, config_.num_playouts);
        for (int i = 0; i < max_playouts; ++i) {
            Board simulation_board = board;
            MCTSNode *selected = selection(root_.get(), simulation_board);
            if (!selected->is_terminal() && selected->has_untried_moves()) {
                selected = expansion(selected, simulation_board);
            }
            Color rollout_player = selected->player_to_move();
            double result = simulation(simulation_board, rollout_player);
            backpropagation(selected, result);
            total_playouts_++;

            if (config_.time_limit_seconds > 0.0) {
                auto now = std::chrono::steady_clock::now();
                double elapsed = std::chrono::duration<double>(now - start_time).count();
                if (elapsed >= config_.time_limit_seconds) {
                    break;
                }
            }
        }
    }

    auto end_time = std::chrono::steady_clock::now();
    double elapsed_seconds = std::chrono::duration<double>(end_time - start_time).count();

    SearchResult result;
    result.top_moves = get_move_statistics();
    if (!root_->children().empty()) {
        const auto *best_child = best_child_robust(root_.get());
        if (best_child) {
            result.best_move = best_child->move;
            result.win_rate = best_child->node->win_rate();
        }
        result.total_visits = root_->visits();
    }
    result.search_time = elapsed_seconds;
    return result;
}

std::vector<MCTSEngine::MoveStats> MCTSEngine::get_move_statistics() const {
    std::vector<MoveStats> stats;
    if (!root_) {
        return stats;
    }
    stats.reserve(root_->children().size());
    for (const auto &child : root_->children()) {
        stats.push_back(make_move_stats(child));
    }
    std::sort(stats.begin(),
              stats.end(),
              [](const MoveStats &a, const MoveStats &b) {
                  return a.visits > b.visits;
              });
    return stats;
}

MCTSNode *MCTSEngine::selection(MCTSNode *node, Board &board) {
    while (!node->children().empty() && node->is_fully_expanded()) {
        auto *child = best_child_ucb(node);
        if (!child) {
            break;
        }
        Board::UndoInfo ignore = board.make_move(child->move);
        (void)ignore;
        node = child->node.get();
    }
    return node;
}

MCTSNode *MCTSEngine::expansion(MCTSNode *node, Board &board) {
    if (!node->has_untried_moves()) {
        return node;
    }

    // If using heuristics, prioritize moves
    Move move;
    if (config_.use_heuristics && node->untried_moves().size() > 1) {
        // Evaluate and sort untried moves by priority
        std::vector<std::pair<Move, int>> move_priorities;
        const auto &untried = node->untried_moves();
        
        for (const auto &m : untried) {
            int priority = evaluate_move_priority(board, m, node->player_to_move());
            move_priorities.emplace_back(m, priority);
        }
        
        // Sort by priority (descending)
        std::sort(move_priorities.begin(), move_priorities.end(),
                  [](const auto &a, const auto &b) {
                      return a.second > b.second;
                  });
        
        // Select from top 50% with some randomness for exploration
        const size_t top_n = std::max(static_cast<size_t>(1),
                                      move_priorities.size() / 2);
        std::mt19937 rng{std::random_device{}()};
        std::uniform_int_distribution<size_t> dist(0, top_n - 1);
        move = move_priorities[dist(rng)].first;
        
        // Remove from untried moves
        node->remove_untried_move(move);
    } else {
        // Random selection (original behavior)
        move = node->pop_untried_move();
    }

    Board::UndoInfo ignore = board.make_move(move);
    (void)ignore;

    auto child = std::make_unique<MCTSNode>(board,
                                            gogame::opposite_color(move.color()),
                                            node);
    child->set_untried_moves(board.get_legal_moves(child->player_to_move()));
    child->mark_terminal(board.is_game_over());
    MCTSNode *child_ptr = child.get();
    node->add_child(std::move(child), move);
    return child_ptr;
}

double MCTSEngine::simulation(Board &board, Color to_move) {
    if (config_.use_heuristics) {
        return heuristic_rollout(board, to_move);
    }
    return default_rollout(board, to_move);
}

void MCTSEngine::backpropagation(MCTSNode *node, double result) {
    MCTSNode *current = node;
    Color player = node->player_to_move();
    while (current != nullptr) {
        current->increment_visits();
        // Assume result is win rate for Black.
        double value = (player == Color::Black) ? result : (1.0 - result);
        current->add_win(value);

        player = gogame::opposite_color(player);
        current = current->parent();
    }
}

double MCTSEngine::calculate_ucb(const MCTSNode *node, int parent_visits) const {
    if (node->visits() == 0) {
        return std::numeric_limits<double>::infinity();
    }
    if (parent_visits <= 0) {
        return std::numeric_limits<double>::infinity();
    }

    const double exploitation = node->wins() / static_cast<double>(node->visits());
    const double exploration = config_.ucb_constant *
        std::sqrt(std::log(static_cast<double>(parent_visits)) / node->visits());
    return exploitation + exploration;
}

MCTSNode::Child *MCTSEngine::best_child_ucb(MCTSNode *node) const {
    MCTSNode::Child *best = nullptr;
    double best_value = -std::numeric_limits<double>::infinity();

    for (auto &child_pair : node->children()) {
        MCTSNode *child = child_pair.node.get();
        const double ucb = calculate_ucb(child, node->visits());
        if (ucb > best_value) {
            best_value = ucb;
            best = &child_pair;
        }
    }
    return best;
}

const MCTSNode::Child *MCTSEngine::best_child_robust(const MCTSNode *node) const {
    if (!node) {
        return nullptr;
    }
    const MCTSNode::Child *best = nullptr;
    int best_visits = -1;
    for (const auto &child_pair : node->children()) {
        const MCTSNode *child = child_pair.node.get();
        if (child->visits() > best_visits) {
            best_visits = child->visits();
            best = &child_pair;
        }
    }
    return best;
}

void MCTSEngine::parallel_search(const Board &board, Color to_move) {
    (void)to_move;
    // Placeholder: implement threads with virtual loss in future iterations.
    const int iterations = std::max(1, config_.num_playouts);
    for (int i = 0; i < iterations; ++i) {
        Board simulation_board = board;
        MCTSNode *selected = selection(root_.get(), simulation_board);
        if (!selected->is_terminal() && selected->has_untried_moves()) {
            selected = expansion(selected, simulation_board);
        }
        Color rollout_player = selected->player_to_move();
        double result = simulation(simulation_board, rollout_player);
        backpropagation(selected, result);
        total_playouts_++;
    }
}

MCTSEngine::MoveStats MCTSEngine::make_move_stats(const MCTSNode::Child &child) const {
    MoveStats stats;
    if (!child.node) {
        return stats;
    }
    stats.move = child.move;
    stats.visits = child.node->visits();
    stats.win_rate = child.node->win_rate();
    stats.ucb_value = calculate_ucb(child.node.get(), std::max(1, root_->visits()));
    return stats;
}

