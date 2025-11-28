#include "transposition_table.h"

#include <algorithm>

TranspositionTable::TranspositionTable(std::size_t max_entries)
    : max_entries_(max_entries) {}

TranspositionTable::Entry TranspositionTable::lookup(std::uint64_t hash) const {
    auto it = table_.find(hash);
    if (it != table_.end()) {
        return it->second;
    }
    return Entry{};
}

void TranspositionTable::store(std::uint64_t hash, int depth, float evaluation, const Move &best_move) {
    if (max_entries_ == 0) {
        return;
    }

    if (table_.size() >= max_entries_) {
        table_.erase(table_.begin());
    }

    Entry entry{};
    entry.valid = true;
    entry.depth = depth;
    entry.evaluation = evaluation;
    entry.best_move = best_move;
    table_[hash] = entry;
}

void TranspositionTable::clear() {
    table_.clear();
}

