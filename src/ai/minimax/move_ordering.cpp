#include "move_ordering.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <vector>

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

    if (is_capturing_move(move, board)) {
        score += 1000.0f;
    }

    if (saves_atari(move, board, player)) {
        score += 500.0f;
    }

    score += position_value(move, board.size());

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

    const float center = static_cast<float>(board_size - 1) / 2.0f;
    const float dx = static_cast<float>(move.x()) - center;
    const float dy = static_cast<float>(move.y()) - center;

    float score = 50.0f / (1.0f + std::sqrt(dx * dx + dy * dy));

    // Bonus for being on star points / influence points.
    std::vector<Point> star_points;
    if (board_size == 9) {
        star_points = {{2, 2}, {6, 2}, {2, 6}, {6, 6}, {4, 4}};
    } else if (board_size == 19) {
        star_points = {
            {3, 3}, {3, 9}, {3, 15},
            {9, 3}, {9, 9}, {9, 15},
            {15, 3}, {15, 9}, {15, 15}
        };
    }

    if (is_point_in_list(star_points, move.point())) {
        score += 30.0f;
    }

    return score;
}

