#include "evaluator.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <numeric>
#include <queue>
#include <set>

namespace {

constexpr std::array<gogame::Point, 4> kDirections{
    gogame::Point{1, 0},
    gogame::Point{-1, 0},
    gogame::Point{0, 1},
    gogame::Point{0, -1},
};

inline int to_index(int x, int y, int size) {
    return y * size + x;
}

inline bool in_bounds(int x, int y, int size) {
    return x >= 0 && y >= 0 && x < size && y < size;
}

} // namespace

Evaluator::Evaluator(int board_size, const Weights &weights)
    : board_size_(board_size), weights_(weights) {}

float Evaluator::evaluate(const Board &board, Color player) const {
    float score = 0.0f;
    
    // Điều chỉnh trọng số dựa trên giai đoạn game (opening vs midgame/endgame)
    int move_count = board.get_move_count();
    int board_size = board.size();
    int opening_threshold = (board_size == 9) ? 10 : (board_size == 13) ? 14 : 18;  // chỉ vài nước đầu dùng khai cuộc
    bool is_opening = move_count < opening_threshold;
    
    // GIẢM: Trong opening, giảm rất mạnh trọng số territory; midgame giữ ở mức vừa
    float territory_weight = is_opening ? weights_.territory * 0.25f : weights_.territory * 0.5f;
    score += territory_weight * evaluate_territory(board, player);
    
    score += weights_.prisoners * evaluate_prisoners(board, player);
    score += weights_.group_strength * evaluate_group_strength(board, player);
    score += weights_.influence * evaluate_influence(board, player);
    score += weights_.patterns * evaluate_patterns(board, player);
    
    // TĂNG: Trọng số cao hơn cho threats và opponent_response (ưu tiên tấn công)
    // Trong opening, tấn công và phản ứng quan trọng hơn chiếm đất
    float threat_weight = is_opening ? 14.0f : 10.0f;        // ưu tiên phát hiện/khai thác điểm yếu
    float response_weight = is_opening ? 18.0f : 12.0f;      // ưu tiên ngăn chặn đối thủ chiếm lãnh thổ
    score += threat_weight * evaluate_threats(board, player);
    score += 5.0f * evaluate_territory_expansion(board, player);
    score += response_weight * evaluate_opponent_response(board, player);

    return score;
}

float Evaluator::evaluate_territory(const Board &board, Color player) const {
    const int size = board.size();
    std::vector<bool> visited(static_cast<std::size_t>(size * size), false);
    float territory_score = 0.0f;
    float secure_territory = 0.0f;  // Lãnh thổ chắc chắn
    float potential_territory = 0.0f;  // Lãnh thổ tiềm năng

    for (int y = 0; y < size; ++y) {
        for (int x = 0; x < size; ++x) {
            const int index = to_index(x, y, size);
            if (visited[index]) {
                continue;
            }
            if (board.at(x, y) != Stone::Empty) {
                continue;
            }

            std::queue<Point> frontier;
            frontier.push(Point{x, y});
            visited[index] = true;

            int region_size = 0;
            bool touches_black = false;
            bool touches_white = false;
            int secure_points = 0;  // Số điểm chắc chắn trong vùng

            while (!frontier.empty()) {
                const Point current = frontier.front();
                frontier.pop();
                region_size += 1;
                
                // Kiểm tra xem điểm này có phải là lãnh thổ chắc chắn không
                if (touches_black && !touches_white) {
                    if (is_secure_territory(board, current.x, current.y, Color::Black)) {
                        secure_points++;
                    }
                } else if (touches_white && !touches_black) {
                    if (is_secure_territory(board, current.x, current.y, Color::White)) {
                        secure_points++;
                    }
                }

                for (const auto &dir : kDirections) {
                    const int nx = current.x + dir.x;
                    const int ny = current.y + dir.y;
                    if (!in_bounds(nx, ny, size)) {
                        continue;
                    }

                    const Stone stone = board.at(nx, ny);
                    const int neighbor_index = to_index(nx, ny, size);

                    if (stone == Stone::Empty) {
                        if (!visited[neighbor_index]) {
                            visited[neighbor_index] = true;
                            frontier.push(Point{nx, ny});
                        }
                    } else if (stone == Stone::Black) {
                        touches_black = true;
                    } else if (stone == Stone::White) {
                        touches_white = true;
                    }
                }
            }

            if (touches_black && touches_white) {
                continue; // Neutral territory.
            }

            if (touches_black || touches_white) {
                const Color owner = touches_black ? Color::Black : Color::White;
                const float contribution = static_cast<float>(region_size);
                
                // Lãnh thổ chắc chắn có giá trị cao hơn
                if (secure_points >= region_size * 0.7f) {
                    // 70% trở lên là chắc chắn -> lãnh thổ chắc chắn
                    secure_territory += (owner == player) ? contribution * 1.5f : -contribution * 1.5f;
                } else {
                    // Lãnh thổ tiềm năng
                    potential_territory += (owner == player) ? contribution : -contribution;
                }
                
                territory_score += (owner == player) ? contribution : -contribution;
            }
        }
    }

    // Kết hợp lãnh thổ chắc chắn và tiềm năng (ưu tiên lãnh thổ chắc chắn)
    return secure_territory * 1.5f + potential_territory * 0.8f + territory_score * 0.2f;
}

float Evaluator::evaluate_prisoners(const Board &board, Color player) const {
    const Color opponent = opposite_color(player);
    const int my_prisoners = board.get_prisoners(player);
    const int opp_prisoners = board.get_prisoners(opponent);
    return static_cast<float>(my_prisoners - opp_prisoners);
}

float Evaluator::evaluate_group_strength(const Board &board, Color player) const {
    const Color opponent = opposite_color(player);
    const auto my_groups = board.groups(player);
    const auto opp_groups = board.groups(opponent);

    auto group_value = [this, &board, player](const GroupSnapshot &group) -> float {
        const int liberties = static_cast<int>(group.liberties.size());
        const int stones = static_cast<int>(group.stones.size());

        float value = 0.0f;
        
        // Đánh giá dựa trên số liberties
        if (liberties <= 1) {
            value -= 40.0f;  // Nguy hiểm - có thể bị bắt
        } else if (liberties == 2) {
            value -= 15.0f;  // Atari - cần chú ý
        } else if (liberties == 3) {
            value += 5.0f;   // An toàn tương đối
        } else { // 4 or more
            value += 15.0f;  // Rất an toàn
        }

        // Bonus cho số quân (nhóm lớn hơn = mạnh hơn)
        value += std::sqrt(static_cast<float>(stones)) * 2.0f;
        
        // Bonus cho eyes (mắt) - nhóm có mắt rất mạnh
        int eyes = count_eyes(group, board);
        value += static_cast<float>(eyes) * 30.0f;  // Mỗi mắt = +30 điểm
        
        // Đánh giá độ an toàn của nhóm
        value += evaluate_group_safety(group, board, player) * 10.0f;

        return value;
    };

    float my_value = 0.0f;
    for (const auto &group : my_groups) {
        my_value += group_value(group);
    }

    float opponent_value = 0.0f;
    for (const auto &group : opp_groups) {
        opponent_value += group_value(group);
    }

    return my_value - opponent_value;
}

float Evaluator::evaluate_influence(const Board &board, Color player) const {
    const int size = board.size();
    std::vector<float> influence(static_cast<std::size_t>(size * size), 0.0f);
    const int radius = std::min(4, size);

    for (int y = 0; y < size; ++y) {
        for (int x = 0; x < size; ++x) {
            const Stone stone = board.at(x, y);
            if (stone == Stone::Empty) {
                continue;
            }

            const Color stone_color = Board::color_from_stone(stone);
            const float sign = (stone_color == player) ? 1.0f : -1.0f;

            for (int dx = -radius; dx <= radius; ++dx) {
                for (int dy = -radius; dy <= radius; ++dy) {
                    const int dist = std::abs(dx) + std::abs(dy);
                    if (dist == 0 || dist > radius) {
                        continue;
                    }

                    const int nx = x + dx;
                    const int ny = y + dy;
                    if (!in_bounds(nx, ny, size)) {
                        continue;
                    }

                    const float amount = 1.0f / static_cast<float>(dist + 1);
                    influence[to_index(nx, ny, size)] += sign * amount;
                }
            }
        }
    }

    float influence_score = std::accumulate(influence.begin(), influence.end(), 0.0f);
    return influence_score / static_cast<float>(size * size);
}

float Evaluator::evaluate_patterns(const Board &board, Color player) const {
    const int size = board.size();
    std::vector<Point> star_points;

    if (size == 9) {
        star_points = {
            {2, 2}, {6, 2}, {2, 6}, {6, 6}, {4, 4}
        };
    } else if (size == 19) {
        star_points = {
            {3, 3}, {3, 9}, {3, 15},
            {9, 3}, {9, 9}, {9, 15},
            {15, 3}, {15, 9}, {15, 15}
        };
    }

    int player_hits = 0;
    int opponent_hits = 0;
    const Color opponent = opposite_color(player);

    for (const auto &point : star_points) {
        if (!in_bounds(point.x, point.y, size)) {
            continue;
        }

        const Stone stone = board.at(point.x, point.y);
        if (stone == Stone::Empty) {
            continue;
        }

        const Color stone_color = Board::color_from_stone(stone);
        if (stone_color == player) {
            player_hits += 1;
        } else if (stone_color == opponent) {
            opponent_hits += 1;
        }
    }

    float pattern_score = static_cast<float>(player_hits - opponent_hits);
    
    // Thêm đánh giá cho các pattern phổ biến (eye patterns, shape patterns)
    // Đếm số mắt (eyes) của mỗi nhóm
    const auto my_groups = board.groups(player);
    const auto opp_groups = board.groups(opponent);
    
    int my_eyes = 0;
    for (const auto &group : my_groups) {
        my_eyes += count_eyes(group, board);
    }
    
    int opp_eyes = 0;
    for (const auto &group : opp_groups) {
        opp_eyes += count_eyes(group, board);
    }
    
    // Bonus cho số mắt (eyes rất quan trọng trong Go)
    pattern_score += static_cast<float>(my_eyes - opp_eyes) * 5.0f;

    return pattern_score;
}

int Evaluator::count_eyes(const GroupSnapshot &group, const Board &board) const {
    // Đếm số mắt (eyes) của một nhóm
    // Mắt là vùng trống được bao quanh bởi quân cùng màu
    int eyes = 0;
    const int size = board.size();
    std::set<int> checked_liberties;  // Để tránh đếm trùng
    
    for (const auto &liberty : group.liberties) {
        const int index = to_index(liberty.x, liberty.y, size);
        if (checked_liberties.find(index) != checked_liberties.end()) {
            continue;
        }
        
        // Kiểm tra xem liberty này có phải là mắt không
        // Mắt: tất cả neighbors đều là quân cùng màu
        bool is_eye = true;
        int same_color_neighbors = 0;
        
        for (const auto &dir : kDirections) {
            const int nx = liberty.x + dir.x;
            const int ny = liberty.y + dir.y;
            if (!in_bounds(nx, ny, size)) {
                is_eye = false;  // Ở biên không phải mắt thật
                break;
            }
            
            const Stone neighbor = board.at(nx, ny);
            if (neighbor == Stone::Empty) {
                is_eye = false;  // Có neighbor trống -> không phải mắt
                break;
            }
            
            // Kiểm tra xem neighbor có phải là quân trong nhóm không
            bool is_group_stone = false;
            for (const auto &stone : group.stones) {
                if (stone.x == nx && stone.y == ny) {
                    is_group_stone = true;
                    same_color_neighbors++;
                    break;
                }
            }
            
            if (!is_group_stone) {
                is_eye = false;  // Có neighbor không phải quân nhóm -> không phải mắt
                break;
            }
        }
        
        if (is_eye && same_color_neighbors >= 3) {
            eyes++;
            // Đánh dấu tất cả liberties trong vùng này
            std::queue<Point> region;
            region.push(liberty);
            checked_liberties.insert(index);
            
            while (!region.empty()) {
                const Point current = region.front();
                region.pop();
                
                for (const auto &dir : kDirections) {
                    const int nx = current.x + dir.x;
                    const int ny = current.y + dir.y;
                    if (!in_bounds(nx, ny, size)) {
                        continue;
                    }
                    
                    const int neighbor_index = to_index(nx, ny, size);
                    if (checked_liberties.find(neighbor_index) != checked_liberties.end()) {
                        continue;
                    }
                    
                    // Kiểm tra xem có phải là liberty của nhóm không
                    for (const auto &lib : group.liberties) {
                        if (lib.x == nx && lib.y == ny) {
                            region.push({nx, ny});
                            checked_liberties.insert(neighbor_index);
                            break;
                        }
                    }
                }
            }
        }
    }
    
    return eyes;
}

bool Evaluator::is_secure_territory(const Board &board, int x, int y, Color player) const {
    // Kiểm tra xem một điểm có phải là lãnh thổ chắc chắn không
    // Lãnh thổ chắc chắn: được bao quanh bởi quân cùng màu, không có quân đối thủ gần
    const int size = board.size();
    const Stone player_stone = Board::stone_from_color(player);
    const Stone opponent_stone = Board::stone_from_color(opposite_color(player));
    
    int player_neighbors = 0;
    int opponent_neighbors = 0;
    int empty_neighbors = 0;
    
    // Kiểm tra 4 hướng chính
    for (const auto &dir : kDirections) {
        const int nx = x + dir.x;
        const int ny = y + dir.y;
        if (!in_bounds(nx, ny, size)) {
            continue;
        }
        
        const Stone stone = board.at(nx, ny);
        if (stone == player_stone) {
            player_neighbors++;
        } else if (stone == opponent_stone) {
            opponent_neighbors++;
        } else {
            empty_neighbors++;
        }
    }
    
    // Cải thiện: Trong opening, territory có thể được nhận diện sớm hơn
    // Nếu có ít nhất 2 quân mình và không có quân đối thủ, có thể là territory
    // Đặc biệt quan trọng cho 19x19
    if (size >= 19) {
        // Trên bàn lớn, territory có thể được nhận diện với ít quân hơn
        if (player_neighbors >= 2 && opponent_neighbors == 0) {
            return true;
        }
    }
    
    // Lãnh thổ chắc chắn: có ít nhất 3 neighbors là quân mình, không có quân đối thủ
    return player_neighbors >= 3 && opponent_neighbors == 0;
}

float Evaluator::evaluate_group_safety(const GroupSnapshot &group, const Board &board, Color player) const {
    // Đánh giá độ an toàn của nhóm (0.0 = nguy hiểm, 1.0 = rất an toàn)
    (void)player;  // Suppress unused parameter warning (có thể dùng trong tương lai)
    const int liberties = static_cast<int>(group.liberties.size());
    const int stones = static_cast<int>(group.stones.size());
    
    float safety = 0.0f;
    
    // Dựa trên số liberties
    if (liberties >= 4) {
        safety = 1.0f;  // Rất an toàn
    } else if (liberties == 3) {
        safety = 0.7f;  // An toàn tương đối
    } else if (liberties == 2) {
        safety = 0.3f;  // Nguy hiểm
    } else {
        safety = 0.0f;  // Rất nguy hiểm
    }
    
    // Bonus cho số mắt
    int eyes = count_eyes(group, board);
    safety += static_cast<float>(eyes) * 0.2f;  // Mỗi mắt tăng 20% safety
    safety = std::min(safety, 1.0f);  // Giới hạn tối đa 1.0
    
    // Bonus cho nhóm lớn
    if (stones >= 5) {
        safety += 0.1f;  // Nhóm lớn an toàn hơn
        safety = std::min(safety, 1.0f);
    }
    
    return safety;
}

float Evaluator::evaluate_threats(const Board &board, Color player) const {
    // Đánh giá mối đe dọa từ đối thủ và cơ hội tấn công
    const Color opponent = opposite_color(player);
    const int size = board.size();
    float threat_score = 0.0f;
    
    // 1. Đánh giá các nhóm đối thủ yếu (có thể tấn công)
    const auto opp_groups = board.groups(opponent);
    for (const auto &group : opp_groups) {
        const int liberties = static_cast<int>(group.liberties.size());
        if (liberties == 1) {
            // Atari - có thể bắt quân đối thủ
            threat_score += 50.0f;  // Cơ hội tấn công lớn
        } else if (liberties == 2) {
            // Có thể tạo atari
            threat_score += 20.0f;  // Cơ hội tấn công
        }
    }
    
    // 2. Đánh giá các nhóm của mình bị đe dọa
    const auto my_groups = board.groups(player);
    for (const auto &group : my_groups) {
        const int liberties = static_cast<int>(group.liberties.size());
        if (liberties == 1) {
            // Nhóm của mình bị atari - cần bảo vệ
            threat_score -= 60.0f;  // Penalty lớn cho mối đe dọa
        } else if (liberties == 2) {
            // Nhóm của mình có thể bị atari
            threat_score -= 25.0f;  // Penalty cho mối đe dọa tiềm năng
        }
    }
    
    // 3. Đánh giá vùng đối thủ đang chiếm (cần phản ứng)
    // Tìm các vùng trống gần quân đối thủ nhưng xa quân mình
    int opponent_territory_threat = 0;
    int my_territory_control = 0;
    
    for (int y = 0; y < size; ++y) {
        for (int x = 0; x < size; ++x) {
            if (board.at(x, y) != Stone::Empty) {
                continue;
            }
            
            // Đếm quân đối thủ và quân mình trong bán kính 3
            int opp_nearby = 0;
            int my_nearby = 0;
            
            for (int dx = -3; dx <= 3; ++dx) {
                for (int dy = -3; dy <= 3; ++dy) {
                    const int dist = std::abs(dx) + std::abs(dy);
                    if (dist == 0 || dist > 3) {
                        continue;
                    }
                    
                    const int nx = x + dx;
                    const int ny = y + dy;
                    if (!in_bounds(nx, ny, size)) {
                        continue;
                    }
                    
                    const Stone stone = board.at(nx, ny);
                    if (stone == Board::stone_from_color(opponent)) {
                        opp_nearby++;
                    } else if (stone == Board::stone_from_color(player)) {
                        my_nearby++;
                    }
                }
            }
            
            // Nếu đối thủ đang chiếm vùng này (nhiều quân đối thủ, ít quân mình)
            if (opp_nearby >= 2 && my_nearby == 0) {
                opponent_territory_threat++;
            } else if (my_nearby >= 2 && opp_nearby == 0) {
                my_territory_control++;
            }
        }
    }
    
    // Penalty cho việc đối thủ đang chiếm nhiều vùng
    threat_score -= static_cast<float>(opponent_territory_threat) * 15.0f;
    // Bonus cho việc mình đang kiểm soát vùng
    threat_score += static_cast<float>(my_territory_control) * 10.0f;
    
    return threat_score;
}

float Evaluator::evaluate_territory_expansion(const Board &board, Color player) const {
    // Đánh giá khả năng mở rộng lãnh thổ từ các vị trí đã chiếm
    const int size = board.size();
    float expansion_score = 0.0f;
    
    const auto my_groups = board.groups(player);
    
    for (const auto &group : my_groups) {
        // Với mỗi nhóm, đánh giá khả năng mở rộng
        int expansion_potential = 0;
        
        // Kiểm tra các liberties của nhóm
        for (const auto &liberty : group.liberties) {
            // Đếm số ô trống xung quanh liberty (có thể mở rộng)
            int empty_around = 0;
            for (const auto &dir : kDirections) {
                const int nx = liberty.x + dir.x;
                const int ny = liberty.y + dir.y;
                if (in_bounds(nx, ny, size) && board.at(nx, ny) == Stone::Empty) {
                    empty_around++;
                }
            }
            
            // Càng nhiều ô trống xung quanh, càng dễ mở rộng
            expansion_potential += empty_around;
        }
        
        // Bonus cho nhóm có khả năng mở rộng tốt
        expansion_score += static_cast<float>(expansion_potential) * 2.0f;
        
        // Bonus cho nhóm lớn (dễ mở rộng hơn)
        expansion_score += std::sqrt(static_cast<float>(group.stones.size())) * 3.0f;
    }
    
    return expansion_score;
}

float Evaluator::evaluate_opponent_response(const Board &board, Color player) const {
    // Đánh giá phản ứng với nước đi của đối thủ
    // Tìm các vùng đối thủ vừa chiếm và cần phản ứng
    const Color opponent = opposite_color(player);
    const int size = board.size();
    float response_score = 0.0f;
    
    // Tìm các quân đối thủ gần đây (giả định là các quân ở biên của nhóm lớn)
    const auto opp_groups = board.groups(opponent);
    
    for (const auto &group : opp_groups) {
        // Kiểm tra xem nhóm này có đang mở rộng không (nhiều liberties)
        const int liberties = static_cast<int>(group.liberties.size());
        
        if (liberties >= 4) {
            // Nhóm đối thủ đang mở rộng - cần phản ứng
            // Tìm các vị trí gần nhóm này để chặn hoặc tấn công
            
            for (const auto &liberty : group.liberties) {
                // Đánh giá xem có nên đánh gần liberty này không
                int my_stones_nearby = 0;
                int empty_around = 0;
                
                for (const auto &dir : kDirections) {
                    const int nx = liberty.x + dir.x;
                    const int ny = liberty.y + dir.y;
                    if (!in_bounds(nx, ny, size)) {
                        continue;
                    }
                    
                    const Stone stone = board.at(nx, ny);
                    if (stone == Board::stone_from_color(player)) {
                        my_stones_nearby++;
                    } else if (stone == Stone::Empty) {
                        empty_around++;
                    }
                }
                
                // Nếu có quân mình gần, có thể tấn công hoặc chặn
                if (my_stones_nearby > 0 && empty_around >= 2) {
                    response_score += 30.0f;  // Cơ hội phản ứng tốt
                } else if (empty_around >= 3) {
                    // Vùng trống lớn - đối thủ có thể mở rộng
                    response_score -= 20.0f;  // Penalty - cần phản ứng
                }
            }
        }
    }
    
    // Đánh giá các vùng trống lớn mà đối thủ có thể chiếm
    // Tìm các vùng trống xa quân mình nhưng gần quân đối thủ
    int unguarded_territory = 0;
    
    for (int y = 0; y < size; ++y) {
        for (int x = 0; x < size; ++x) {
            if (board.at(x, y) != Stone::Empty) {
                continue;
            }
            
            // Đếm khoảng cách đến quân gần nhất của mỗi bên
            int min_dist_to_me = size * 2;
            int min_dist_to_opp = size * 2;
            
            for (int dy = 0; dy < size; ++dy) {
                for (int dx = 0; dx < size; ++dx) {
                    const Stone stone = board.at(dx, dy);
                    if (stone == Stone::Empty) {
                        continue;
                    }
                    
                    const int dist = std::abs(x - dx) + std::abs(y - dy);
                    if (stone == Board::stone_from_color(player)) {
                        min_dist_to_me = std::min(min_dist_to_me, dist);
                    } else if (stone == Board::stone_from_color(opponent)) {
                        min_dist_to_opp = std::min(min_dist_to_opp, dist);
                    }
                }
            }
            
            // Nếu vùng này gần đối thủ hơn mình, cần phản ứng
            if (min_dist_to_opp < min_dist_to_me && min_dist_to_opp <= 3) {
                unguarded_territory++;
            }
        }
    }
    
    // Penalty cho việc có nhiều vùng không được bảo vệ
    response_score -= static_cast<float>(unguarded_territory) * 12.0f;
    
    return response_score;
}

