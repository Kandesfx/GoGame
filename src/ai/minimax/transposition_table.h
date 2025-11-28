#ifndef MINIMAX_TRANSPOSITION_TABLE_H
#define MINIMAX_TRANSPOSITION_TABLE_H

#include <cstdint>
#include <unordered_map>

#include "../../game/board.h"

class TranspositionTable {
public:
    struct Entry {
        bool valid{false};
        int depth{0};
        float evaluation{0.0f};
        Move best_move{};
    };

    explicit TranspositionTable(std::size_t max_entries = 1'000'000);

    [[nodiscard]] Entry lookup(std::uint64_t hash) const;
    void store(std::uint64_t hash, int depth, float evaluation, const Move &best_move);
    void clear();

private:
    using Table = std::unordered_map<std::uint64_t, Entry>;

    std::size_t max_entries_;
    Table table_;
};

#endif // MINIMAX_TRANSPOSITION_TABLE_H

