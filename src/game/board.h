#ifndef GOGAME_BOARD_H
#define GOGAME_BOARD_H

#include <array>
#include <cstdint>
#include <vector>

namespace gogame {

constexpr int kMaxBoardSize = 19;

enum class Color : std::uint8_t {
    Black = 0,
    White = 1,
};

inline Color opposite_color(Color color) {
    return color == Color::Black ? Color::White : Color::Black;
}

enum class Stone : std::uint8_t {
    Empty = 0,
    Black = 1,
    White = 2,
};

struct Point {
    int x{0};
    int y{0};

    bool operator==(const Point &other) const {
        return x == other.x && y == other.y;
    }
};

struct GroupSnapshot {
    std::vector<Point> stones;
    std::vector<Point> liberties;
};

class Move {
public:
    Move() = default;
    Move(int x, int y, Color color);

    static Move Pass(Color color);

    [[nodiscard]] bool is_valid() const { return valid_; }
    [[nodiscard]] bool is_pass() const { return is_pass_; }
    [[nodiscard]] int x() const { return point_.x; }
    [[nodiscard]] int y() const { return point_.y; }
    [[nodiscard]] Point point() const { return point_; }
    [[nodiscard]] Color color() const { return color_; }

private:
    Point point_{};
    Color color_{Color::Black};
    bool is_pass_{false};
    bool valid_{false};
};

class Board {
public:
    struct CapturedStone {
        int index{0};
        Stone stone{Stone::Empty};
    };

    struct UndoInfo {
        Move move{};
        std::vector<CapturedStone> captured{};
        int previous_consecutive_passes{0};
        Color previous_player{Color::Black};
        std::array<int, 2> previous_prisoners{0, 0};
        std::uint64_t previous_hash{0};
        int previous_ko_index{-1};
    };

    explicit Board(int size = 9);

    [[nodiscard]] int size() const { return size_; }
    [[nodiscard]] Color current_player() const { return to_move_; }
    [[nodiscard]] bool is_game_over() const { return consecutive_passes_ >= 2; }
    [[nodiscard]] int get_prisoners(Color color) const { return prisoners_[color_index(color)]; }
    [[nodiscard]] Stone at(int x, int y) const { return grid_[to_index(x, y)]; }
    [[nodiscard]] int get_move_count() const { return move_count_; }

    [[nodiscard]] std::vector<Move> get_legal_moves(Color player) const;
    [[nodiscard]] bool is_legal_move(const Move &move) const;

    UndoInfo make_move(const Move &move);
    void undo_move(const UndoInfo &undo);

    [[nodiscard]] std::uint64_t zobrist_hash() const { return hash_; }

    static Stone stone_from_color(Color color) { return color == Color::Black ? Stone::Black : Stone::White; }
    static Color color_from_stone(Stone stone) { return stone == Stone::Black ? Color::Black : Color::White; }

    [[nodiscard]] GroupSnapshot group_at(Point point) const;
    [[nodiscard]] std::vector<GroupSnapshot> groups(Color color) const;

private:
    struct GroupInfo {
        std::vector<int> stones;
        std::vector<int> liberties;
    };

    static int color_index(Color color) { return color == Color::Black ? 0 : 1; }

    [[nodiscard]] bool in_bounds(int x, int y) const;
    [[nodiscard]] int to_index(int x, int y) const;
    [[nodiscard]] Point from_index(int index) const;

    [[nodiscard]] std::vector<int> neighbors(int index) const;
    [[nodiscard]] GroupInfo collect_group(int index) const;

    void apply_move(const Move &move, UndoInfo &undo);
    void remove_stone(int index, UndoInfo &undo);

    void init_zobrist();
    void recompute_hash();

    int size_{9};
    std::vector<Stone> grid_{};
    Color to_move_{Color::Black};
    int consecutive_passes_{0};
    int move_count_{0};
    int ko_index_{-1};
    std::array<int, 2> prisoners_{0, 0};
    std::uint64_t hash_{0};

    std::vector<std::array<std::uint64_t, 3>> zobrist_table_;
    std::uint64_t zobrist_player_{0};
};

} // namespace gogame

using gogame::Board;
using gogame::Color;
using gogame::Move;
using gogame::Point;
using gogame::Stone;
using gogame::opposite_color;
using gogame::GroupSnapshot;

#endif // GOGAME_BOARD_H

