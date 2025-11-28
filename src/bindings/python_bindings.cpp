#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>
#include <pybind11/numpy.h>

#include "../game/board.h"
#include "../ai/ai_player.h"

namespace py = pybind11;

PYBIND11_MODULE(gogame_py, m) {
    py::enum_<Color>(m, "Color")
        .value("Black", Color::Black)
        .value("White", Color::White);

    py::enum_<Stone>(m, "Stone")
        .value("Empty", Stone::Empty)
        .value("Black", Stone::Black)
        .value("White", Stone::White);

    py::class_<Point>(m, "Point")
        .def(py::init<int, int>())
        .def_readwrite("x", &Point::x)
        .def_readwrite("y", &Point::y);

    py::class_<Move>(m, "Move")
        .def(py::init<>())
        .def(py::init<int, int, Color>())
        .def_static("pass_move", &Move::Pass)
        .def_property_readonly("is_valid", &Move::is_valid)
        .def_property_readonly("is_pass", &Move::is_pass)
        .def_property_readonly("x", &Move::x)
        .def_property_readonly("y", &Move::y)
        .def_property_readonly("color", &Move::color);

    py::class_<Board::UndoInfo>(m, "UndoInfo");

    py::class_<Board>(m, "Board")
        .def(py::init<int>(), py::arg("size") = 9)
        .def("size", &Board::size)
        .def("current_player", &Board::current_player)
        .def("is_game_over", &Board::is_game_over)
        .def("get_prisoners", &Board::get_prisoners)
        .def("at", &Board::at)
        .def("get_legal_moves", &Board::get_legal_moves)
        .def("is_legal_move", &Board::is_legal_move)
        .def("make_move", &Board::make_move)
        .def("undo_move", &Board::undo_move)
        .def("zobrist_hash", &Board::zobrist_hash);

    py::class_<MinimaxEngine::Config>(m, "MinimaxConfig")
        .def(py::init<>())
        .def_readwrite("max_depth", &MinimaxEngine::Config::max_depth)
        .def_readwrite("use_alpha_beta", &MinimaxEngine::Config::use_alpha_beta)
        .def_readwrite("use_move_ordering", &MinimaxEngine::Config::use_move_ordering)
        .def_readwrite("use_transposition", &MinimaxEngine::Config::use_transposition)
        .def_readwrite("time_limit_seconds", &MinimaxEngine::Config::time_limit_seconds)
        .def_readwrite("board_size", &MinimaxEngine::Config::board_size);

    py::class_<MinimaxEngine::SearchResult>(m, "MinimaxSearchResult")
        .def_readonly("best_move", &MinimaxEngine::SearchResult::best_move)
        .def_readonly("evaluation", &MinimaxEngine::SearchResult::evaluation)
        .def_readonly("nodes_searched", &MinimaxEngine::SearchResult::nodes_searched)
        .def_readonly("nodes_pruned", &MinimaxEngine::SearchResult::nodes_pruned)
        .def_readonly("search_time", &MinimaxEngine::SearchResult::search_time)
        .def_readonly("principal_variation", &MinimaxEngine::SearchResult::principal_variation);

    py::class_<MinimaxEngine>(m, "MinimaxEngine")
        .def(py::init<const MinimaxEngine::Config &>())
        .def("search", &MinimaxEngine::search);

    py::class_<MCTSEngine::Config>(m, "MCTSConfig")
        .def(py::init<int, double, double, bool, bool, int>(),
             py::arg("num_playouts") = 3000,
             py::arg("time_limit_seconds") = 0.0,
             py::arg("ucb_constant") = 1.41421356237,
             py::arg("use_heuristics") = false,
             py::arg("parallel") = false,
             py::arg("num_threads") = 1)
        .def_readwrite("num_playouts", &MCTSEngine::Config::num_playouts)
        .def_readwrite("time_limit_seconds", &MCTSEngine::Config::time_limit_seconds)
        .def_readwrite("ucb_constant", &MCTSEngine::Config::ucb_constant)
        .def_readwrite("use_heuristics", &MCTSEngine::Config::use_heuristics)
        .def_readwrite("parallel", &MCTSEngine::Config::parallel)
        .def_readwrite("num_threads", &MCTSEngine::Config::num_threads);

    py::class_<MCTSEngine::MoveStats>(m, "MCTSMoveStats")
        .def_readonly("move", &MCTSEngine::MoveStats::move)
        .def_readonly("visits", &MCTSEngine::MoveStats::visits)
        .def_readonly("win_rate", &MCTSEngine::MoveStats::win_rate)
        .def_readonly("ucb_value", &MCTSEngine::MoveStats::ucb_value);

    py::class_<MCTSEngine::SearchResult>(m, "MCTSSearchResult")
        .def_readonly("best_move", &MCTSEngine::SearchResult::best_move)
        .def_readonly("win_rate", &MCTSEngine::SearchResult::win_rate)
        .def_readonly("total_visits", &MCTSEngine::SearchResult::total_visits)
        .def_readonly("search_time", &MCTSEngine::SearchResult::search_time)
        .def_readonly("top_moves", &MCTSEngine::SearchResult::top_moves);

    py::class_<MCTSEngine>(m, "MCTSEngine")
        .def(py::init<const MCTSEngine::Config &>())
        .def("search", &MCTSEngine::search)
        .def("get_move_statistics", &MCTSEngine::get_move_statistics);

    py::enum_<AIPlayer::Algorithm>(m, "AIAlgorithm")
        .value("Minimax", AIPlayer::Algorithm::Minimax)
        .value("MCTS", AIPlayer::Algorithm::MCTS);

    py::class_<AIPlayer::LevelConfig>(m, "LevelConfig")
        .def(py::init<>())
        .def_readwrite("algorithm", &AIPlayer::LevelConfig::algorithm)
        .def_readwrite("minimax", &AIPlayer::LevelConfig::minimax)
        .def_readwrite("mcts", &AIPlayer::LevelConfig::mcts);

    py::class_<AIPlayer>(m, "AIPlayer")
        .def(py::init<>())
        .def("select_move", &AIPlayer::select_move)
        .def("minimax_result", &AIPlayer::minimax_result)
        .def("mcts_result", &AIPlayer::mcts_result)
        .def("set_level_config", &AIPlayer::set_level_config)
        .def("get_level_config", &AIPlayer::get_level_config, py::return_value_policy::reference);
}

