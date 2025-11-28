cmake_minimum_required(VERSION 3.20)

project(GoGame LANGUAGES CXX)

enable_testing()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(SRC_ROOT "${PROJECT_SOURCE_DIR}/src")

set(CORE_SOURCES
    "${SRC_ROOT}/game/board.cpp"
)

add_library(gogame_core STATIC ${CORE_SOURCES})

target_include_directories(gogame_core
    PUBLIC
        "${SRC_ROOT}"
        "${SRC_ROOT}/game"
)

target_compile_features(gogame_core PUBLIC cxx_std_20)

if (MSVC)
    target_compile_options(gogame_core PRIVATE /W4 /permissive-)
else()
    target_compile_options(gogame_core PRIVATE -Wall -Wextra -Wpedantic)
endif()

set(MINIMAX_SOURCES
    "${SRC_ROOT}/ai/minimax/minimax_engine.cpp"
    "${SRC_ROOT}/ai/minimax/move_ordering.cpp"
    "${SRC_ROOT}/ai/minimax/transposition_table.cpp"
    "${SRC_ROOT}/ai/minimax/game_tree.cpp"
    "${SRC_ROOT}/ai/minimax/evaluator.cpp"
)

add_library(gogame_minimax STATIC ${MINIMAX_SOURCES})

target_include_directories(gogame_minimax
    PUBLIC
        "${SRC_ROOT}"
        "${SRC_ROOT}/ai"
)

target_link_libraries(gogame_minimax PUBLIC gogame_core)

target_compile_features(gogame_minimax PUBLIC cxx_std_20)

if (MSVC)
    target_compile_options(gogame_minimax PRIVATE /W4 /permissive-)
else()
    target_compile_options(gogame_minimax PRIVATE -Wall -Wextra -Wpedantic)
endif()

set(MCTS_SOURCES
    "${SRC_ROOT}/ai/mcts/mcts_engine.cpp"
    "${SRC_ROOT}/ai/mcts/mcts_node.cpp"
)

add_library(gogame_mcts STATIC ${MCTS_SOURCES})

target_include_directories(gogame_mcts
    PUBLIC
        "${SRC_ROOT}"
        "${SRC_ROOT}/ai"
)

target_link_libraries(gogame_mcts PUBLIC gogame_core)

target_compile_features(gogame_mcts PUBLIC cxx_std_20)

if (MSVC)
    target_compile_options(gogame_mcts PRIVATE /W4 /permissive-)
else()
    target_compile_options(gogame_mcts PRIVATE -Wall -Wextra -Wpedantic)
endif()

set(AI_PLAYER_SOURCES
    "${SRC_ROOT}/ai/ai_player.cpp"
)

add_library(gogame_ai STATIC ${AI_PLAYER_SOURCES})

target_include_directories(gogame_ai
    PUBLIC
        "${SRC_ROOT}"
        "${SRC_ROOT}/ai"
)

target_link_libraries(gogame_ai
    PUBLIC
        gogame_core
        gogame_minimax
        gogame_mcts
)

target_compile_features(gogame_ai PUBLIC cxx_std_20)

if (MSVC)
    target_compile_options(gogame_ai PRIVATE /W4 /permissive-)
else()
    target_compile_options(gogame_ai PRIVATE -Wall -Wextra -Wpedantic)
endif()

find_package(pybind11 CONFIG QUIET)

if (pybind11_FOUND)
    message(STATUS "pybind11 found. Building Python bindings.")
    pybind11_add_module(gogame_py src/bindings/python_bindings.cpp)
    target_include_directories(gogame_py
        PRIVATE
            "${SRC_ROOT}"
            "${SRC_ROOT}/ai"
            "${SRC_ROOT}/game"
    )
    target_link_libraries(gogame_py
        PRIVATE
            gogame_core
            gogame_ai
    )
else()
    message(STATUS "pybind11 not found. Skipping Python bindings.")
endif()

add_executable(test_board tests/test_board.cpp)
target_include_directories(test_board PRIVATE "${SRC_ROOT}")
target_link_libraries(test_board PRIVATE gogame_core)
add_test(NAME test_board COMMAND test_board)

add_executable(test_minimax tests/test_minimax.cpp)
target_include_directories(test_minimax PRIVATE "${SRC_ROOT}")
target_link_libraries(test_minimax PRIVATE gogame_core gogame_minimax)
add_test(NAME test_minimax COMMAND test_minimax)

add_executable(test_mcts tests/test_mcts.cpp)
target_include_directories(test_mcts PRIVATE "${SRC_ROOT}")
target_link_libraries(test_mcts PRIVATE gogame_core gogame_mcts)
add_test(NAME test_mcts COMMAND test_mcts)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

