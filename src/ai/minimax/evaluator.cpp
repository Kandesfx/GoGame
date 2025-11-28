#include "evaluator.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <numeric>
#include <queue>

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

    score += weights_.territory * evaluate_territory(board, player);
    score += weights_.prisoners * evaluate_prisoners(board, player);
    score += weights_.group_strength * evaluate_group_strength(board, player);
    score += weights_.influence * evaluate_influence(board, player);
    score += weights_.patterns * evaluate_patterns(board, player);

    return score;
}

float Evaluator::evaluate_territory(const Board &board, Color player) const {
    const int size = board.size();
    std::vector<bool> visited(static_cast<std::size_t>(size * size), false);
    float territory_score = 0.0f;

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

            while (!frontier.empty()) {
                const Point current = frontier.front();
                frontier.pop();
                region_size += 1;

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
                territory_score += (owner == player) ? contribution : -contribution;
            }
        }
    }

    return territory_score;
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

    auto group_value = [](const GroupSnapshot &group) -> float {
        const int liberties = static_cast<int>(group.liberties.size());
        const int stones = static_cast<int>(group.stones.size());

        float value = 0.0f;
        if (liberties <= 1) {
            value -= 40.0f;
        } else if (liberties == 2) {
            value -= 15.0f;
        } else if (liberties == 3) {
            value += 5.0f;
        } else { // 4 or more
            value += 15.0f;
        }

        value += std::sqrt(static_cast<float>(stones));
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

    return static_cast<float>(player_hits - opponent_hits);
}

