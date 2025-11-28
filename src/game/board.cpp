#include "board.h"

#include <algorithm>
#include <queue>
#include <random>
#include <set>
#include <stdexcept>
#include <utility>

namespace gogame {
// NOTE: In Go, stones are placed on INTERSECTIONS (giao điểm) of grid lines,
// NOT in squares. The board is represented as a grid where each (x, y) coordinate
// represents an intersection point.

namespace {
constexpr int kMaxStones = kMaxBoardSize * kMaxBoardSize;
inline int stone_index(Stone stone) {
    return static_cast<int>(stone);
}
} // namespace

Move::Move(int x, int y, Color color) {
    point_ = {x, y};
    color_ = color;
    is_pass_ = false;
    valid_ = true;
}

Move Move::Pass(Color color) {
    Move move;
    move.color_ = color;
    move.is_pass_ = true;
    move.valid_ = true;
    return move;
}

Board::Board(int size)
    : size_(size),
      grid_(size * size, Stone::Empty) {
    if (size <= 0 || size > kMaxBoardSize) {
        throw std::invalid_argument("Board size out of supported range");
    }
    init_zobrist();
    recompute_hash();
}

std::vector<Move> Board::get_legal_moves(Color player) const {
    std::vector<Move> moves;
    moves.reserve(size_ * size_ + 1);

    for (int y = 0; y < size_; ++y) {
        for (int x = 0; x < size_; ++x) {
            Move move{x, y, player};
            if (is_legal_move(move)) {
                moves.push_back(move);
            }
        }
    }

    moves.push_back(Move::Pass(player));
    return moves;
}

bool Board::is_legal_move(const Move &move) const {
    if (!move.is_valid()) {
        return false;
    }

    if (move.is_pass()) {
        return true;
    }

    if (!in_bounds(move.x(), move.y())) {
        return false;
    }

    const int index = to_index(move.x(), move.y());

    if (grid_[index] != Stone::Empty) {
        return false;
    }

    if (ko_index_ == index) {
        return false;
    }

    Board temp(*this);
    temp.to_move_ = move.color();
    UndoInfo undo{};
    try {
        temp.apply_move(move, undo);
    } catch (const std::runtime_error &) {
        return false;
    }

    const Stone placed_stone = temp.grid_[index];
    if (placed_stone == Stone::Empty) {
        return false;
    }

    const GroupInfo group = temp.collect_group(index);
    return !group.liberties.empty();
}

Board::UndoInfo Board::make_move(const Move &move) {
    if (!is_legal_move(move)) {
        throw std::invalid_argument("Illegal move attempted");
    }

    UndoInfo undo{};
    apply_move(move, undo);
    return undo;
}

void Board::undo_move(const UndoInfo &undo) {
    move_count_--;
    to_move_ = undo.previous_player;
    consecutive_passes_ = undo.previous_consecutive_passes;
    prisoners_ = undo.previous_prisoners;
    hash_ = undo.previous_hash;
    ko_index_ = undo.previous_ko_index;

    if (undo.move.is_pass()) {
        return;
    }

    const Point point = undo.move.point();
    const int index = to_index(point.x, point.y);
    grid_[index] = Stone::Empty;

    for (const auto &captured : undo.captured) {
        grid_[captured.index] = captured.stone;
    }
}

bool Board::in_bounds(int x, int y) const {
    return x >= 0 && x < size_ && y >= 0 && y < size_;
}

int Board::to_index(int x, int y) const {
    return y * size_ + x;
}

Point Board::from_index(int index) const {
    const int y = index / size_;
    const int x = index % size_;
    return {x, y};
}

std::vector<int> Board::neighbors(int index) const {
    const Point p = from_index(index);
    std::vector<int> result;
    result.reserve(4);

    const std::array<Point, 4> directions{{{1, 0}, {-1, 0}, {0, 1}, {0, -1}}};
    for (const auto &dir : directions) {
        const int nx = p.x + dir.x;
        const int ny = p.y + dir.y;
        if (in_bounds(nx, ny)) {
            result.push_back(to_index(nx, ny));
        }
    }
    return result;
}

Board::GroupInfo Board::collect_group(int index) const {
    GroupInfo info{};
    const Stone color = grid_[index];
    if (color == Stone::Empty) {
        return info;
    }

    std::vector<bool> visited(grid_.size(), false);
    std::vector<bool> liberty_seen(grid_.size(), false);
    std::queue<int> frontier;

    frontier.push(index);
    visited[index] = true;

    while (!frontier.empty()) {
        const int current = frontier.front();
        frontier.pop();
        info.stones.push_back(current);

        for (const int neighbor : neighbors(current)) {
            const Stone neighbor_stone = grid_[neighbor];
            if (neighbor_stone == color && !visited[neighbor]) {
                visited[neighbor] = true;
                frontier.push(neighbor);
            } else if (neighbor_stone == Stone::Empty && !liberty_seen[neighbor]) {
                liberty_seen[neighbor] = true;
                info.liberties.push_back(neighbor);
            }
        }
    }

    return info;
}

void Board::apply_move(const Move &move, UndoInfo &undo) {
    undo.move = move;
    undo.previous_player = to_move_;
    undo.previous_consecutive_passes = consecutive_passes_;
    undo.previous_prisoners = prisoners_;
    undo.previous_hash = hash_;
    undo.previous_ko_index = ko_index_;

    to_move_ = opposite_color(move.color());
    move_count_++;
    hash_ ^= zobrist_player_;

    if (move.is_pass()) {
        consecutive_passes_ += 1;
        ko_index_ = -1;
        return;
    }

    consecutive_passes_ = 0;

    const int index = to_index(move.x(), move.y());
    grid_[index] = stone_from_color(move.color());
    hash_ ^= zobrist_table_[index][stone_index(grid_[index])];
    ko_index_ = -1;

    // Capture rule: Check opponent groups around the placed stone
    // A group is captured when it has NO liberties (không còn khí)
    // IMPORTANT: After placing the stone, it occupies one liberty of adjacent opponent groups
    const Stone opponent_stone = stone_from_color(opposite_color(move.color()));
    std::set<int> captured_indices_set; // Use set to avoid duplicates
    std::vector<bool> processed_group(grid_.size(), false); // Track processed groups

    for (const int neighbor : neighbors(index)) {
        if (grid_[neighbor] == opponent_stone && !processed_group[neighbor]) {
            // Collect the opponent group starting from this neighbor
            const GroupInfo opponent_group = collect_group(neighbor);
            if (opponent_group.stones.empty()) {
                continue;
            }
            
            // Mark all stones in this group as processed
            for (const int stone_index_value : opponent_group.stones) {
                processed_group[stone_index_value] = true;
            }
            
            // If opponent group has no liberties (after our stone was placed),
            // capture all stones in the group
            // NOTE: After placing our stone at index, it occupies one liberty of adjacent groups
            // So if a group now has 0 liberties, it should be captured
            if (opponent_group.liberties.empty()) {
                for (const int stone_index_value : opponent_group.stones) {
                    captured_indices_set.insert(stone_index_value);
                }
            }
        }
    }

    // Remove captured stones
    for (const int captured_index : captured_indices_set) {
        remove_stone(captured_index, undo);
    }

    const GroupInfo own_group = collect_group(index);
    if (own_group.liberties.empty()) {
        // Suicide should not happen due to legality check.
        // Revert and throw to catch logic bugs.
        undo_move(undo);
        throw std::runtime_error("Suicide move applied unexpectedly");
    }

    if (captured_indices_set.size() == 1 && own_group.stones.size() == 1) {
        ko_index_ = *captured_indices_set.begin();
    }
}

void Board::remove_stone(int index, UndoInfo &undo) {
    const Stone stone = grid_[index];
    if (stone == Stone::Empty) {
        return;
    }

    undo.captured.push_back({index, stone});

    const Color color = color_from_stone(stone);
    prisoners_[color_index(opposite_color(color))] += 1;

    hash_ ^= zobrist_table_[index][stone_index(stone)];
    grid_[index] = Stone::Empty;
}

void Board::init_zobrist() {
    if (!zobrist_table_.empty()) {
        return;
    }

    std::mt19937_64 rng{0xA15A4C93u};
    std::uniform_int_distribution<std::uint64_t> dist;

    zobrist_table_.resize(size_ * size_);
    for (auto &entry : zobrist_table_) {
        entry[stone_index(Stone::Empty)] = 0;
        entry[stone_index(Stone::Black)] = dist(rng);
        entry[stone_index(Stone::White)] = dist(rng);
    }

    zobrist_player_ = dist(rng);
}

void Board::recompute_hash() {
    hash_ = 0;
    for (std::size_t index = 0; index < grid_.size(); ++index) {
        const Stone stone = grid_[index];
        if (stone != Stone::Empty) {
            hash_ ^= zobrist_table_[index][stone_index(stone)];
        }
    }
    if (to_move_ == Color::White) {
        hash_ ^= zobrist_player_;
    }
}

GroupSnapshot Board::group_at(Point point) const {
    GroupSnapshot snapshot;
    if (!in_bounds(point.x, point.y)) {
        return snapshot;
    }

    const int index = to_index(point.x, point.y);
    const Stone stone = grid_[index];
    if (stone == Stone::Empty) {
        return snapshot;
    }

    const GroupInfo group_info = collect_group(index);
    snapshot.stones.reserve(group_info.stones.size());
    snapshot.liberties.reserve(group_info.liberties.size());

    for (const int stone_index_value : group_info.stones) {
        snapshot.stones.push_back(from_index(stone_index_value));
    }

    for (const int liberty_index : group_info.liberties) {
        snapshot.liberties.push_back(from_index(liberty_index));
    }

    return snapshot;
}

std::vector<GroupSnapshot> Board::groups(Color color) const {
    std::vector<GroupSnapshot> result;
    if (grid_.empty()) {
        return result;
    }

    const Stone target = stone_from_color(color);
    std::vector<bool> visited(grid_.size(), false);

    for (std::size_t index = 0; index < grid_.size(); ++index) {
        if (grid_[index] != target || visited[index]) {
            continue;
        }

        const GroupInfo info = collect_group(static_cast<int>(index));

        GroupSnapshot snapshot;
        snapshot.stones.reserve(info.stones.size());
        snapshot.liberties.reserve(info.liberties.size());

        for (const int stone_index_value : info.stones) {
            visited[stone_index_value] = true;
            snapshot.stones.push_back(from_index(stone_index_value));
        }

        for (const int liberty_index : info.liberties) {
            snapshot.liberties.push_back(from_index(liberty_index));
        }

        result.push_back(std::move(snapshot));
    }

    return result;
}

} // namespace gogame

