#include "move_ordering.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <vector>
#include <queue>

using gogame::Point;
using gogame::Color;
using gogame::Stone;
using gogame::Board;
using gogame::Move;
using gogame::opposite_color;

namespace {

constexpr std::array<Point, 4> kDirections{
    Point{1, 0},
    Point{-1, 0},
    Point{0, 1},
    Point{0, -1},
};

inline bool in_bounds(int x, int y, int size) {
    return x >= 0 && y >= 0 && x < size && y < size;
}

inline bool is_point_in_list(const std::vector<Point> &list, const Point &point) {
    return std::find(list.begin(), list.end(), point) != list.end();
}

inline int to_index(int x, int y, int size) {
    return y * size + x;
}

} // namespace

void MoveOrdering::order_moves(std::vector<Move> &moves, const Board &board, Color player) {
    std::vector<std::pair<float, Move>> scored_moves;
    scored_moves.reserve(moves.size());

    for (const auto &move : moves) {
        float score = score_move(move, board, player);
        scored_moves.emplace_back(score, move);
    }

    std::sort(scored_moves.begin(),
              scored_moves.end(),
              [](const auto &lhs, const auto &rhs) {
                  return lhs.first > rhs.first;
              });

    moves.clear();
    moves.reserve(scored_moves.size());
    for (const auto &[score, move] : scored_moves) {
        (void)score;
        moves.push_back(move);
    }
}

float MoveOrdering::score_move(const Move &move, const Board &board, Color player) {
    float score = 0.0f;
    
    // Pass move: đánh giá dựa trên game state
    if (move.is_pass()) {
        int move_count = board.get_move_count();
        int board_size = board.size();
        
        // Trong endgame (sau 80% số nước có thể), pass có thể là tốt
        int max_moves = board_size * board_size * 2;  // Ước tính số nước tối đa
        int endgame_threshold = static_cast<int>(max_moves * 0.8);
        
        if (move_count >= endgame_threshold) {
            // Trong endgame, pass có thể là nước tốt nếu đã có nhiều nước
            // Nhưng vẫn ưu tiên các nước khác trước
            score = -100.0f;  // Pass có điểm thấp hơn các nước thường, nhưng không quá thấp
        } else {
            // Trong khai cuộc và trung cuộc, pass thường không tốt
            score = -1000.0f;  // Pass rất thấp trong khai cuộc/trung cuộc
        }
        
        return score;
    }

    if (is_capturing_move(move, board)) {
        score += 1000.0f;
    }

    if (saves_atari(move, board, player)) {
        score += 500.0f;
    }

    score += position_value(move, board.size());
    
    // Thêm heuristics cho khai cuộc và endgame
    int move_count = board.get_move_count();
    int board_size = board.size();
    int max_moves = board_size * board_size * 2;  // Ước tính số nước tối đa
    
    // Trong khai cuộc: chỉ vài nước đầu dùng heuristics mở cờ
    int opening_threshold = (board_size == 9) ? 6 : (board_size == 13) ? 7 : 8;
    if (move_count < opening_threshold) {
        // QUAN TRỌNG: Ưu tiên tấn công/phòng thủ, giảm ham đất
        score += opponent_response_bonus(move, board, player) * 1.7f;  // tăng trọng số phản ứng
        
        // Bonus vừa phải cho star points (giữ tự nhiên)
        score += star_point_bonus(move, board_size) * 0.3f;
        
        // Penalty cho việc đánh quá gần nhau trong opening
        score -= clustering_penalty(move, board, player);
        
        // Bonus cho việc mở rộng ra các vùng khác (tránh tập trung một góc)
        score += territory_diversification_bonus(move, board, player) * 0.8f;
        
        // Bonus cho việc tạo connection tốt (gần quân mình) - quan trọng cho tấn công
        score += connection_bonus(move, board, player);
        
        // Bonus cho việc mở rộng influence
        score += influence_bonus(move, board, player) * 0.8f;
        
        // GIẢM: Chiếm đất sớm chỉ điểm rất thấp
        score += opening_territory_bonus(move, board, player) * 0.25f;
        
        // Mở rộng nhẹ nếu an toàn
        score += expansion_bonus(move, board, player) * 0.4f;
    } else {
        // Trung cuộc / tàn cuộc: ưu tiên công thủ và ngăn chặn chiếm đất
        score += opponent_response_bonus(move, board, player) * 1.3f;
        score += expansion_bonus(move, board, player) * 0.6f;  // giảm mở rộng quá nhiều đất
    }
    
    // Trong endgame (sau 70% số nước), ưu tiên các nước quan trọng
    int endgame_threshold = static_cast<int>(max_moves * 0.7);
    if (move_count >= endgame_threshold) {
        // Bonus cho các nước bảo vệ lãnh thổ
        score += endgame_territory_bonus(move, board, player);
        
        // Bonus cho các nước tạo mắt (eyes)
        score += endgame_eye_bonus(move, board, player);
    }

    return score;
}

bool MoveOrdering::is_capturing_move(const Move &move, const Board &board) {
    if (!move.is_valid() || move.is_pass()) {
        return false;
    }

    const int size = board.size();
    if (board.at(move.x(), move.y()) != Stone::Empty) {
        return false;
    }

    const Color opponent = opposite_color(move.color());
    for (const auto &dir : kDirections) {
        const int nx = move.x() + dir.x;
        const int ny = move.y() + dir.y;
        if (!in_bounds(nx, ny, size)) {
            continue;
        }

        if (board.at(nx, ny) == Board::stone_from_color(opponent)) {
            const GroupSnapshot group = board.group_at(Point{nx, ny});
            if (group.liberties.size() == 1 && group.liberties.front() == move.point()) {
                return true;
            }
        }
    }

    return false;
}

bool MoveOrdering::saves_atari(const Move &move, const Board &board, Color player) {
    if (!move.is_valid() || move.is_pass()) {
        return false;
    }

    const int size = board.size();
    if (board.at(move.x(), move.y()) != Stone::Empty) {
        return false;
    }

    for (const auto &dir : kDirections) {
        const int nx = move.x() + dir.x;
        const int ny = move.y() + dir.y;
        if (!in_bounds(nx, ny, size)) {
            continue;
        }

        if (board.at(nx, ny) == Board::stone_from_color(player)) {
            const GroupSnapshot group = board.group_at(Point{nx, ny});
            if (group.liberties.size() == 1 && group.liberties.front() == move.point()) {
                return true;
            }
        }
    }

    return false;
}

float MoveOrdering::position_value(const Move &move, int board_size) {
    if (!move.is_valid() || move.is_pass()) {
        return 0.0f;
    }

    const int x = move.x();
    const int y = move.y();
    
    // QUAN TRỌNG: Penalty lớn cho corner moves (góc thực sự)
    // Corner moves: (0,0), (0, size-1), (size-1, 0), (size-1, size-1)
    const bool is_corner = (x == 0 || x == board_size - 1) && (y == 0 || y == board_size - 1);
    if (is_corner) {
        // Corner moves rất kém trong khai cuộc - penalty lớn
        return -200.0f;
    }
    
    // Penalty nhẹ cho edge moves (cạnh nhưng không phải góc)
    const bool is_edge = (x == 0 || x == board_size - 1 || y == 0 || y == board_size - 1);
    if (is_edge) {
        // Edge moves không tốt trong opening, nhưng có thể chấp nhận trong midgame
        return -50.0f;
    }

    const float center = static_cast<float>(board_size - 1) / 2.0f;
    const float dx = static_cast<float>(x) - center;
    const float dy = static_cast<float>(y) - center;

    float score = 50.0f / (1.0f + std::sqrt(dx * dx + dy * dy));

    // Bonus for being on star points / influence points.
    std::vector<Point> star_points;
    if (board_size == 9) {
        star_points = {{2, 2}, {6, 2}, {2, 6}, {6, 6}, {4, 4}};
    } else if (board_size == 13) {
        star_points = {{3, 3}, {3, 9}, {9, 3}, {9, 9}, {6, 6}};
    } else if (board_size == 19) {
        star_points = {
            {3, 3}, {3, 9}, {3, 15},
            {9, 3}, {9, 9}, {9, 15},
            {15, 3}, {15, 9}, {15, 15}
        };
    }

    if (is_point_in_list(star_points, move.point())) {
        score += 50.0f;  // Tăng bonus cho star points
    }
    
    // Bonus cho các vị trí gần star points (tự nhiên hơn)
    for (const auto& star : star_points) {
        const float dist = std::sqrt(
            static_cast<float>((x - star.x) * (x - star.x) + (y - star.y) * (y - star.y))
        );
        if (dist <= 2.0f && dist > 0.0f) {
            score += 20.0f / dist;  // Bonus giảm dần theo khoảng cách
        }
    }

    return score;
}

float MoveOrdering::connection_bonus(const Move &move, const Board &board, Color player) {
    if (!move.is_valid() || move.is_pass()) {
        return 0.0f;
    }
    
    float bonus = 0.0f;
    const int size = board.size();
    const Stone player_stone = Board::stone_from_color(player);
    
    // Kiểm tra xem move có gần quân mình không (tạo connection tốt)
    for (const auto &dir : kDirections) {
        const int nx = move.x() + dir.x;
        const int ny = move.y() + dir.y;
        if (!in_bounds(nx, ny, size)) {
            continue;
        }
        
        if (board.at(nx, ny) == player_stone) {
            // Gần quân mình: bonus cho connection
            bonus += 20.0f;
        }
    }
    
    // Bonus cho việc tạo shape tốt (knight's move, diagonal connection)
    const std::array<Point, 8> extended_directions{
        Point{1, 0}, Point{-1, 0}, Point{0, 1}, Point{0, -1},
        Point{1, 1}, Point{-1, -1}, Point{1, -1}, Point{-1, 1}
    };
    
    for (const auto &dir : extended_directions) {
        const int nx = move.x() + dir.x;
        const int ny = move.y() + dir.y;
        if (!in_bounds(nx, ny, size)) {
            continue;
        }
        
        if (board.at(nx, ny) == player_stone) {
            // Knight's move hoặc diagonal: tạo shape tốt hơn
            bonus += 10.0f;
        }
    }
    
    return bonus;
}

float MoveOrdering::influence_bonus(const Move &move, const Board &board, Color player) {
    if (!move.is_valid() || move.is_pass()) {
        return 0.0f;
    }
    
    float bonus = 0.0f;
    const int size = board.size();
    const Stone player_stone = Board::stone_from_color(player);
    const Stone opponent_stone = Board::stone_from_color(opposite_color(player));
    
    // Kiểm tra influence: move có tạo áp lực lên vùng trống không
    // Đếm số ô trống xung quanh (liberties)
    int empty_neighbors = 0;
    int friendly_neighbors = 0;
    int opponent_neighbors = 0;
    
    for (const auto &dir : kDirections) {
        const int nx = move.x() + dir.x;
        const int ny = move.y() + dir.y;
        if (!in_bounds(nx, ny, size)) {
            continue;
        }
        
        Stone stone = board.at(nx, ny);
        if (stone == Stone::Empty) {
            empty_neighbors++;
        } else if (stone == player_stone) {
            friendly_neighbors++;
        } else if (stone == opponent_stone) {
            opponent_neighbors++;
        }
    }
    
    // Bonus cho việc có nhiều liberties (tự do hơn)
    bonus += empty_neighbors * 5.0f;
    
    // Bonus cho việc tạo áp lực lên đối thủ (gần quân đối thủ nhưng an toàn)
    if (opponent_neighbors > 0 && empty_neighbors >= 2) {
        bonus += 15.0f;  // Tạo áp lực nhưng vẫn an toàn
    }
    
    return bonus;
}

float MoveOrdering::opening_territory_bonus(const Move &move, const Board &board, Color player) {
    if (!move.is_valid() || move.is_pass()) {
        return 0.0f;
    }
    
    float bonus = 0.0f;
    const int size = board.size();
    const Stone player_stone = Board::stone_from_color(player);
    const Stone opponent_stone = Board::stone_from_color(opposite_color(player));
    
    // Đếm neighbors
    int friendly_neighbors = 0;
    int empty_neighbors = 0;
    int opponent_neighbors = 0;
    
    // Kiểm tra 4 hướng chính
    for (const auto &dir : kDirections) {
        const int nx = move.x() + dir.x;
        const int ny = move.y() + dir.y;
        if (!in_bounds(nx, ny, size)) {
            continue;
        }
        
        Stone stone = board.at(nx, ny);
        if (stone == player_stone) {
            friendly_neighbors++;
        } else if (stone == Stone::Empty) {
            empty_neighbors++;
        } else if (stone == opponent_stone) {
            opponent_neighbors++;
        }
    }
    
    // GIẢM: Bonus cho moves chiếm territory trong opening (không quá cao)
    // Ưu tiên tấn công/phòng thủ hơn là chiếm đất
    // 1. Moves gần quân mình tạo vùng territory (nhưng không quá quan trọng)
    if (friendly_neighbors >= 1) {
        // Giảm trọng số: càng nhiều quân mình xung quanh, càng tạo territory tốt
        bonus += friendly_neighbors * 6.0f;  // Giảm từ 15 xuống 6
        
        // Nếu có nhiều ô trống xung quanh, territory lớn hơn
        if (empty_neighbors >= 2) {
            bonus += empty_neighbors * 4.0f;  // Giảm từ 10 xuống 4
        }
    }
    
    // 2. Moves mở rộng từ vị trí đã đánh (extension) - giảm trọng số
    // Kiểm tra các vị trí xa hơn (knight's move, 2-space jump)
    const std::array<Point, 8> extended_directions{
        Point{2, 0}, Point{-2, 0}, Point{0, 2}, Point{0, -2},
        Point{2, 1}, Point{-2, 1}, Point{1, 2}, Point{1, -2}
    };
    
    int friendly_extended = 0;
    for (const auto &dir : extended_directions) {
        const int nx = move.x() + dir.x;
        const int ny = move.y() + dir.y;
        if (!in_bounds(nx, ny, size)) {
            continue;
        }
        
        if (board.at(nx, ny) == player_stone) {
            friendly_extended++;
        }
    }
    
    // Extension moves tạo territory lớn hơn (giảm trọng số)
    if (friendly_extended > 0 && empty_neighbors >= 3) {
        bonus += friendly_extended * 8.0f;  // Giảm từ 20 xuống 8
    }
    
    // 3. Moves tạo influence lớn (gần center, nhiều liberties) - giảm trọng số
    const float center = static_cast<float>(size - 1) / 2.0f;
    const float dx = static_cast<float>(move.x()) - center;
    const float dy = static_cast<float>(move.y()) - center;
    const float dist_from_center = std::sqrt(dx * dx + dy * dy);
    
    // Moves gần center hơn tạo influence tốt hơn (nhưng không quá quan trọng)
    if (dist_from_center < static_cast<float>(size) * 0.3f) {
        bonus += 10.0f;  // Giảm từ 25 xuống 10
    } else if (dist_from_center < static_cast<float>(size) * 0.5f) {
        bonus += 6.0f;  // Giảm từ 15 xuống 6
    }
    
    // 4. Moves tạo shape tốt cho territory (có nhiều liberties) - giảm trọng số
    if (empty_neighbors >= 3) {
        bonus += 8.0f;  // Giảm từ 20 xuống 8
    }
    
    // 5. Penalty cho moves quá gần đối thủ (không tạo territory tốt)
    if (opponent_neighbors >= 2 && friendly_neighbors == 0) {
        bonus -= 30.0f;  // Quá gần đối thủ, không tạo territory
    }
    
    // 6. GIẢM: Bonus đặc biệt cho 19x19 (không quá cao)
    if (size == 19) {
        // Trên bàn lớn, territory vẫn quan trọng nhưng không quá ưu tiên
        if (friendly_neighbors >= 1 && empty_neighbors >= 3) {
            bonus += 12.0f;  // Giảm từ 30 xuống 12
        }
        
        // Moves ở vùng giữa các star points tạo territory tốt
        if (dist_from_center < static_cast<float>(size) * 0.4f && empty_neighbors >= 2) {
            bonus += 10.0f;  // Giảm từ 25 xuống 10
        }
    }
    
    return bonus;
}

float MoveOrdering::endgame_territory_bonus(const Move &move, const Board &board, Color player) {
    if (!move.is_valid() || move.is_pass()) {
        return 0.0f;
    }
    
    float bonus = 0.0f;
    const int size = board.size();
    const Stone player_stone = Board::stone_from_color(player);
    
    // Kiểm tra xem move có bảo vệ lãnh thổ không (gần quân mình)
    int friendly_neighbors = 0;
    int empty_neighbors = 0;
    
    for (const auto &dir : kDirections) {
        const int nx = move.x() + dir.x;
        const int ny = move.y() + dir.y;
        if (!in_bounds(nx, ny, size)) {
            continue;
        }
        
        Stone stone = board.at(nx, ny);
        if (stone == player_stone) {
            friendly_neighbors++;
        } else if (stone == Stone::Empty) {
            empty_neighbors++;
        }
    }
    
    // Bonus cho việc bảo vệ lãnh thổ (gần quân mình và có nhiều ô trống xung quanh)
    if (friendly_neighbors >= 2 && empty_neighbors >= 2) {
        bonus += 25.0f;  // Bảo vệ lãnh thổ tốt
    }
    
    // Bonus cho việc mở rộng lãnh thổ (gần quân mình nhưng không quá gần)
    if (friendly_neighbors == 1 && empty_neighbors >= 3) {
        bonus += 15.0f;  // Mở rộng lãnh thổ
    }
    
    return bonus;
}

float MoveOrdering::endgame_eye_bonus(const Move &move, const Board &board, Color player) {
    if (!move.is_valid() || move.is_pass()) {
        return 0.0f;
    }
    
    float bonus = 0.0f;
    const int size = board.size();
    const Stone player_stone = Board::stone_from_color(player);
    
    // Kiểm tra xem move có tạo mắt (eye) không
    // Mắt: vùng trống được bao quanh bởi quân cùng màu
    int same_color_neighbors = 0;
    int empty_neighbors = 0;
    
    for (const auto &dir : kDirections) {
        const int nx = move.x() + dir.x;
        const int ny = move.y() + dir.y;
        if (!in_bounds(nx, ny, size)) {
            continue;
        }
        
        Stone stone = board.at(nx, ny);
        if (stone == player_stone) {
            same_color_neighbors++;
        } else if (stone == Stone::Empty) {
            empty_neighbors++;
        }
    }
    
    // Nếu move được bao quanh bởi 3-4 quân cùng màu, có thể tạo mắt
    if (same_color_neighbors >= 3) {
        bonus += 30.0f;  // Tạo mắt - rất quan trọng trong endgame
    } else if (same_color_neighbors == 2 && empty_neighbors <= 1) {
        bonus += 15.0f;  // Có thể tạo mắt
    }
    
    return bonus;
}

float MoveOrdering::expansion_bonus(const Move &move, const Board &board, Color player) {
    if (!move.is_valid() || move.is_pass()) {
        return 0.0f;
    }
    
    float bonus = 0.0f;
    const int size = board.size();
    const Stone player_stone = Board::stone_from_color(player);
    
    // 1. Mở rộng từ các nhóm hiện có (extension)
    int friendly_nearby = 0;
    int empty_around_move = 0;
    
    // Kiểm tra trong bán kính 3
    for (int dx = -3; dx <= 3; ++dx) {
        for (int dy = -3; dy <= 3; ++dy) {
            const int dist = std::abs(dx) + std::abs(dy);
            if (dist == 0 || dist > 3) {
                continue;
            }
            
            const int nx = move.x() + dx;
            const int ny = move.y() + dy;
            if (!in_bounds(nx, ny, size)) {
                continue;
            }
            
            const Stone stone = board.at(nx, ny);
            if (stone == player_stone) {
                friendly_nearby++;
            } else if (stone == Stone::Empty && dist <= 2) {
                empty_around_move++;
            }
        }
    }
    
    // Bonus cho extension từ nhóm hiện có
    if (friendly_nearby >= 1) {
        if (friendly_nearby == 1 || friendly_nearby == 2) {
            bonus += 40.0f;  // Extension tốt
        }
        if (empty_around_move >= 3) {
            bonus += empty_around_move * 8.0f;  // Mở rộng lãnh thổ
        }
    }
    
    // 2. Mở rộng vào vùng trống lớn
    int empty_neighbors = 0;
    for (const auto &dir : kDirections) {
        const int nx = move.x() + dir.x;
        const int ny = move.y() + dir.y;
        if (in_bounds(nx, ny, size) && board.at(nx, ny) == Stone::Empty) {
            empty_neighbors++;
        }
    }
    
    if (empty_neighbors >= 3) {
        bonus += 25.0f;  // Shape tốt cho mở rộng
    }
    
    // 3. Bonus cho moves ở vùng trống lớn
    int large_empty_region = 0;
    std::vector<bool> checked(size * size, false);
    std::queue<Point> region_queue;
    region_queue.push({move.x(), move.y()});
    checked[to_index(move.x(), move.y(), size)] = true;
    
    while (!region_queue.empty() && large_empty_region < 20) {
        const Point current = region_queue.front();
        region_queue.pop();
        large_empty_region++;
        
        for (const auto &dir : kDirections) {
            const int nx = current.x + dir.x;
            const int ny = current.y + dir.y;
            if (!in_bounds(nx, ny, size)) {
                continue;
            }
            
            const int index = to_index(nx, ny, size);
            if (checked[index]) {
                continue;
            }
            
            if (board.at(nx, ny) == Stone::Empty) {
                checked[index] = true;
                region_queue.push({nx, ny});
            }
        }
    }
    
    if (large_empty_region >= 10) {
        bonus += 30.0f;  // Vùng trống lớn - cơ hội chiếm đất
    }
    
    return bonus;
}

float MoveOrdering::opponent_response_bonus(const Move &move, const Board &board, Color player) {
    if (!move.is_valid() || move.is_pass()) {
        return 0.0f;
    }
    
    float bonus = 0.0f;
    const int size = board.size();
    const Color opponent = opposite_color(player);
    const Stone opponent_stone = Board::stone_from_color(opponent);
    const Stone player_stone = Board::stone_from_color(player);
    
    // TĂNG: Ưu tiên tấn công và phản ứng hơn
    // 1. Phản ứng với các nhóm đối thủ đang mở rộng
    const auto opp_groups = board.groups(opponent);
    
    for (const auto &group : opp_groups) {
        const int liberties = static_cast<int>(group.liberties.size());
        
        if (liberties >= 4) {
            for (const auto &liberty : group.liberties) {
                const int dist = std::abs(move.x() - liberty.x) + std::abs(move.y() - liberty.y);
                
                if (dist <= 2) {
                    bonus += 80.0f;  // Tăng từ 50 lên 80 - phản ứng với đối thủ đang mở rộng
                    
                    int my_stones_nearby = 0;
                    for (const auto &dir : kDirections) {
                        const int nx = liberty.x + dir.x;
                        const int ny = liberty.y + dir.y;
                        if (in_bounds(nx, ny, size) && board.at(nx, ny) == player_stone) {
                            my_stones_nearby++;
                        }
                    }
                    
                    if (my_stones_nearby > 0) {
                        bonus += 50.0f;  // Tăng từ 30 lên 50 - có thể tấn công nhóm đối thủ
                    }
                }
            }
        }
    }
    
    // 2. Phản ứng với các vùng đối thủ đang chiếm
    int opp_nearby = 0;
    int my_nearby = 0;
    int empty_around = 0;
    
    for (int dx = -3; dx <= 3; ++dx) {
        for (int dy = -3; dy <= 3; ++dy) {
            const int dist = std::abs(dx) + std::abs(dy);
            if (dist == 0 || dist > 3) {
                continue;
            }
            
            const int nx = move.x() + dx;
            const int ny = move.y() + dy;
            if (!in_bounds(nx, ny, size)) {
                continue;
            }
            
            const Stone stone = board.at(nx, ny);
            if (stone == opponent_stone) {
                opp_nearby++;
            } else if (stone == player_stone) {
                my_nearby++;
            } else if (stone == Stone::Empty && dist <= 2) {
                empty_around++;
            }
        }
    }
    
    if (opp_nearby >= 2 && my_nearby == 0) {
        bonus += 90.0f;  // Tăng từ 60 lên 90 - phản ứng với đối thủ chiếm đất
        if (empty_around >= 3) {
            bonus += 40.0f;  // Tăng từ 25 lên 40 - có thể chặn đối thủ
        }
    } else if (opp_nearby >= 1 && my_nearby >= 1) {
        bonus += 60.0f;  // Tăng từ 35 lên 60 - cơ hội tấn công
    }
    
    // 3. TĂNG: Phản ứng với các nhóm đối thủ yếu (ưu tiên tấn công)
    for (const auto &group : opp_groups) {
        const int liberties = static_cast<int>(group.liberties.size());
        
        if (liberties <= 2) {
            for (const auto &liberty : group.liberties) {
                const int dist = std::abs(move.x() - liberty.x) + std::abs(move.y() - liberty.y);
                
                if (dist == 1) {
                    if (liberties == 1) {
                        bonus += 150.0f;  // Tăng từ 100 lên 150 - có thể bắt quân
                    } else if (liberties == 2) {
                        bonus += 70.0f;  // Tăng từ 40 lên 70 - có thể tạo atari
                    }
                }
            }
        }
    }
    
    // 4. Phản ứng với các vùng trống lớn mà đối thủ có thể chiếm
    int unguarded_territory_nearby = 0;
    
    for (int dx = -4; dx <= 4; ++dx) {
        for (int dy = -4; dy <= 4; ++dy) {
            const int dist = std::abs(dx) + std::abs(dy);
            if (dist == 0 || dist > 4) {
                continue;
            }
            
            const int nx = move.x() + dx;
            const int ny = move.y() + dy;
            if (!in_bounds(nx, ny, size)) {
                continue;
            }
            
            if (board.at(nx, ny) != Stone::Empty) {
                continue;
            }
            
            int min_dist_to_me = size * 2;
            int min_dist_to_opp = size * 2;
            
            for (int dy2 = 0; dy2 < size; ++dy2) {
                for (int dx2 = 0; dx2 < size; ++dx2) {
                    const Stone stone = board.at(dx2, dy2);
                    if (stone == Stone::Empty) {
                        continue;
                    }
                    
                    const int dist2 = std::abs(nx - dx2) + std::abs(ny - dy2);
                    if (stone == player_stone) {
                        min_dist_to_me = std::min(min_dist_to_me, dist2);
                    } else if (stone == opponent_stone) {
                        min_dist_to_opp = std::min(min_dist_to_opp, dist2);
                    }
                }
            }
            
            if (min_dist_to_opp < min_dist_to_me && min_dist_to_opp <= 3) {
                unguarded_territory_nearby++;
            }
        }
    }
    
    if (unguarded_territory_nearby >= 3) {
        bonus += 40.0f;  // Phản ứng với mối đe dọa
    }
    
    return bonus;
}

float MoveOrdering::star_point_bonus(const Move &move, int board_size) {
    if (!move.is_valid() || move.is_pass()) {
        return 0.0f;
    }
    
    // GIỮ: Bonus vừa phải cho star points (tự nhiên, không ép buộc)
    std::vector<Point> star_points;
    if (board_size == 9) {
        star_points = {{2, 2}, {6, 2}, {2, 6}, {6, 6}, {4, 4}};
    } else if (board_size == 13) {
        star_points = {{3, 3}, {9, 3}, {3, 9}, {9, 9}, {6, 6}};
    } else if (board_size == 19) {
        star_points = {
            {3, 3}, {3, 9}, {3, 15},
            {9, 3}, {9, 9}, {9, 15},
            {15, 3}, {15, 9}, {15, 15}
        };
    }
    
    for (const auto &star : star_points) {
        if (move.x() == star.x && move.y() == star.y) {
            // Star point - bonus vừa phải (giảm từ 200 xuống 60)
            return 60.0f;  // Bonus vừa phải, không quá ưu tiên
        }
    }
    
    // Bonus nhỏ hơn cho các vị trí gần star points (trong bán kính 1)
    for (const auto &star : star_points) {
        const int dist = std::abs(move.x() - star.x) + std::abs(move.y() - star.y);
        if (dist == 1) {
            return 20.0f;  // Gần star point cũng tốt nhưng không quá cao
        }
    }
    
    return 0.0f;
}

float MoveOrdering::clustering_penalty(const Move &move, const Board &board, Color player) {
    if (!move.is_valid() || move.is_pass()) {
        return 0.0f;
    }
    
    // QUAN TRỌNG: Penalty cho việc đánh quá gần nhau trong opening
    // Điều này giúp AI tránh tập trung vào một góc
    const int size = board.size();
    const Stone player_stone = Board::stone_from_color(player);
    int nearby_stones = 0;
    
    // Đếm số quân mình trong bán kính 2
    for (int dx = -2; dx <= 2; ++dx) {
        for (int dy = -2; dy <= 2; ++dy) {
            const int dist = std::abs(dx) + std::abs(dy);
            if (dist == 0 || dist > 2) {
                continue;
            }
            
            const int nx = move.x() + dx;
            const int ny = move.y() + dy;
            if (!in_bounds(nx, ny, size)) {
                continue;
            }
            
            if (board.at(nx, ny) == player_stone) {
                nearby_stones++;
            }
        }
    }
    
    // Penalty nếu có quá nhiều quân gần nhau (clustering)
    // Trong opening, nên mở rộng ra các vùng khác
    if (nearby_stones >= 3) {
        return 150.0f;  // Penalty lớn cho clustering
    } else if (nearby_stones == 2) {
        return 50.0f;  // Penalty nhỏ hơn
    }
    
    return 0.0f;
}

float MoveOrdering::territory_diversification_bonus(const Move &move, const Board &board, Color player) {
    if (!move.is_valid() || move.is_pass()) {
        return 0.0f;
    }
    
    // QUAN TRỌNG: Bonus cho việc mở rộng ra các vùng khác của bàn cờ
    // Giúp AI tránh tập trung vào một góc
    const int size = board.size();
    const Stone player_stone = Board::stone_from_color(player);
    
    // Tìm vùng có nhiều quân mình nhất (có thể là góc)
    int total_stones = 0;
    
    // Chia bàn cờ thành 4 góc và center
    const int center_x = size / 2;
    const int center_y = size / 2;
    
    // Đếm quân mình trong mỗi vùng
    int top_left = 0, top_right = 0, bottom_left = 0, bottom_right = 0;
    
    for (int y = 0; y < size; ++y) {
        for (int x = 0; x < size; ++x) {
            if (board.at(x, y) == player_stone) {
                total_stones++;
                
                if (x < center_x && y < center_y) {
                    top_left++;
                } else if (x >= center_x && y < center_y) {
                    top_right++;
                } else if (x < center_x && y >= center_y) {
                    bottom_left++;
                } else if (x >= center_x && y >= center_y) {
                    bottom_right++;
                }
            }
        }
    }
    
    if (total_stones == 0) {
        return 0.0f;
    }
    
    int max_stones_in_region = std::max({top_left, top_right, bottom_left, bottom_right});
    
    // Nếu có quá nhiều quân tập trung ở một vùng (ví dụ: góc), bonus cho việc mở rộng ra vùng khác
    if (max_stones_in_region >= total_stones * 0.6f) {
        // Xác định vùng của move hiện tại
        bool move_in_crowded_region = false;
        
        if (move.x() < center_x && move.y() < center_y && top_left >= total_stones * 0.6f) {
            move_in_crowded_region = true;
        } else if (move.x() >= center_x && move.y() < center_y && top_right >= total_stones * 0.6f) {
            move_in_crowded_region = true;
        } else if (move.x() < center_x && move.y() >= center_y && bottom_left >= total_stones * 0.6f) {
            move_in_crowded_region = true;
        } else if (move.x() >= center_x && move.y() >= center_y && bottom_right >= total_stones * 0.6f) {
            move_in_crowded_region = true;
        }
        
        // Nếu move không ở vùng đông đúc, bonus cao
        if (!move_in_crowded_region) {
            return 120.0f;  // Bonus cao cho việc mở rộng ra vùng khác
        } else {
            // Nếu move ở vùng đông đúc, penalty
            return -80.0f;  // Penalty cho việc tiếp tục tập trung
        }
    }
    
    // Bonus cho việc đánh ở vùng chưa có quân mình
    int stones_in_move_region = 0;
    if (move.x() < center_x && move.y() < center_y) {
        stones_in_move_region = top_left;
    } else if (move.x() >= center_x && move.y() < center_y) {
        stones_in_move_region = top_right;
    } else if (move.x() < center_x && move.y() >= center_y) {
        stones_in_move_region = bottom_left;
    } else if (move.x() >= center_x && move.y() >= center_y) {
        stones_in_move_region = bottom_right;
    }
    
    // Nếu vùng này chưa có quân mình, bonus
    if (stones_in_move_region == 0) {
        return 100.0f;  // Bonus cho việc mở rộng ra vùng mới
    }
    
    return 0.0f;
}
