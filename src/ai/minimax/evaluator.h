#ifndef MINIMAX_EVALUATOR_H
#define MINIMAX_EVALUATOR_H

#include "../../game/board.h"

class Evaluator {
public:
    struct Weights {
        float territory;
        float prisoners;
        float group_strength;
        float influence;
        float patterns;

        constexpr Weights(float territory_val = 10.0f,
                          float prisoners_val = 5.0f,
                          float group_strength_val = 3.0f,
                          float influence_val = 2.0f,
                          float patterns_val = 1.0f)
            : territory(territory_val),
              prisoners(prisoners_val),
              group_strength(group_strength_val),
              influence(influence_val),
              patterns(patterns_val) {}
    };

    Evaluator() = default;
    explicit Evaluator(int board_size, const Weights &weights = Weights{});

    [[nodiscard]] float evaluate(const Board &board, Color player) const;

private:
    int board_size_{9};
    Weights weights_{};

    float evaluate_territory(const Board &board, Color player) const;
    float evaluate_prisoners(const Board &board, Color player) const;
    float evaluate_group_strength(const Board &board, Color player) const;
    float evaluate_influence(const Board &board, Color player) const;
    float evaluate_patterns(const Board &board, Color player) const;
    
    // Các hàm helper mới
    int count_eyes(const GroupSnapshot &group, const Board &board) const;
    bool is_secure_territory(const Board &board, int x, int y, Color player) const;
    float evaluate_group_safety(const GroupSnapshot &group, const Board &board, Color player) const;
    
    // Các hàm đánh giá mới để cải thiện AI
    float evaluate_threats(const Board &board, Color player) const;
    float evaluate_territory_expansion(const Board &board, Color player) const;
    float evaluate_opponent_response(const Board &board, Color player) const;
};

#endif // MINIMAX_EVALUATOR_H

