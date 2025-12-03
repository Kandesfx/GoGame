import { useState, useEffect } from "react";
import { FaCircle, FaTimes, FaCopy, FaCheck } from "react-icons/fa";
import { useAuth } from "../contexts/AuthContext";
import Board from "./Board";
import GameControls from "./GameControls";
import MoveHistory from "./MoveHistory";
import StatisticsPanel from "./StatisticsPanel";
import MatchDialog from "./MatchDialog";
import SettingsDialog from "./SettingsDialog";
import KoDialog from "./KoDialog";
import api from "../services/api";
import { playStoneSound, resetStoneSoundCounter } from "../utils/sound";
import "./MainWindow.css";
import PropTypes from "prop-types";

// Force reload v2
console.log("🔄 MainWindow.jsx loaded - version 2");

const MainWindow = ({ onLogout, onBackToHome, initialMatch = null }) => {
  const { user } = useAuth();
  const [currentMatch, setCurrentMatch] = useState(initialMatch);
  // Lưu lịch sử trận đấu nếu cần dùng cho tương lai (hiện tại chưa dùng trong UI)
  // eslint-disable-next-line no-unused-vars
  const [matchHistory, setMatchHistory] = useState([]);
  const [statistics, setStatistics] = useState(null);
  const [moveHistory, setMoveHistory] = useState([]);
  const [showMatchDialog, setShowMatchDialog] = useState(false);
  const [boardState, setBoardState] = useState({
    stones: {},
    boardSize: 9,
    lastMove: null,
    prisonersBlack: 0,
    prisonersWhite: 0,
    currentPlayer: "B",
    blackTimeRemaining: null, // Thời gian còn lại của Black (giây)
    whiteTimeRemaining: null, // Thời gian còn lại của White (giây)
    koPosition: null, // Vị trí KO (nếu có)
  });
  const [isProcessing, setIsProcessing] = useState(false); // Prevent duplicate moves
  const [isDataLoaded, setIsDataLoaded] = useState(false); // Track if data has been loaded
  const [gameOver, setGameOver] = useState(false); // Track game over state
  const [gameResult, setGameResult] = useState(null); // Game result: "B+X", "W+X", "DRAW", "B+R", "W+R"
  const [showGameOverModal, setShowGameOverModal] = useState(false); // Control game over modal
  const [gameOverMessage, setGameOverMessage] = useState(null); // Game over message to display
  const [finalElo, setFinalElo] = useState(null); // ELO cuối trận đấu
  const [eloChange, setEloChange] = useState(null); // ELO change từ trận đấu
  const [gameScoreDetails, setGameScoreDetails] = useState(null); // Chi tiết điểm số: {stonesBlack, stonesWhite, territoryBlack, territoryWhite, komi}
  const [showSettingsDialog, setShowSettingsDialog] = useState(false);
  const [settings, setSettings] = useState(() => {
    const saved = localStorage.getItem("goGameSettings");
    return saved
      ? JSON.parse(saved)
      : {
          soundEnabled: true,
          showCoordinates: true,
          showLastMove: true,
          boardTheme: "classic",
          animationSpeed: "normal",
        };
  });
  const [roomCodeCopied, setRoomCodeCopied] = useState(false);
  const [showPlayerColorModal, setShowPlayerColorModal] = useState(false); // Modal thông báo màu quân cờ
  const [playerColor, setPlayerColor] = useState(null); // 'B' hoặc 'W'
  const [showKoDialog, setShowKoDialog] = useState(false); // Dialog thông báo tình trạng cướp cờ KO
  const [koPosition, setKoPosition] = useState(null); // Vị trí KO hiện tại
  const [previousKoPosition, setPreviousKoPosition] = useState(null); // Vị trí KO trước đó để detect thay đổi
  const [showOpponentPassDialog, setShowOpponentPassDialog] = useState(false); // Dialog thông báo đối phương bỏ lượt
  const [opponentPassMessage, setOpponentPassMessage] = useState(""); // Nội dung thông báo bỏ lượt

  // Debug: Log dialog state changes
  useEffect(() => {
    console.log("🔍 MatchDialog state:", showMatchDialog);
    if (showMatchDialog) {
      console.warn(
        "⚠️ MatchDialog is OPEN - if stuck, press Esc or click outside"
      );
    }
  }, [showMatchDialog]);

  // Hiển thị dialog khi đối phương (AI hoặc người chơi khác) bỏ lượt
  useEffect(() => {
    if (!currentMatch || !playerColor || moveHistory.length === 0) return;

    const lastMove = moveHistory[moveHistory.length - 1];
    if (!lastMove) return;

    // Chỉ quan tâm đến pass (position null) và không phải nước đi của mình
    if (
      lastMove.position === null &&
      lastMove.color &&
      lastMove.color !== playerColor
    ) {
      const isAiOpponent = !!currentMatch.ai_level;
      const opponentName = isAiOpponent ? "AI" : "Đối thủ";
      setOpponentPassMessage(`${opponentName} đã bỏ lượt`);
      setShowOpponentPassDialog(true);
    }
  }, [moveHistory, currentMatch, playerColor]);

  // Load initial data only once on mount
  // eslint-disable-next-line react-hooks/exhaustive-deps
  useEffect(() => {
    if (isDataLoaded) return; // Prevent duplicate calls

    let isMounted = true;

    const loadData = async () => {
      if (!isMounted) return;
      await loadInitialData();
      setIsDataLoaded(true);

      if (initialMatch && isMounted) {
        setCurrentMatch(initialMatch);
        // Load match state if needed
        await loadMatchState(initialMatch.id).catch((err) => {
          console.error("Failed to load initial match state:", err);
        });
      }
    };

    loadData();

    return () => {
      isMounted = false;
    };
  }, []); // Only run once on mount

  // Handle initialMatch changes separately
  useEffect(() => {
    if (initialMatch && initialMatch.id !== currentMatch?.id) {
      setCurrentMatch(initialMatch);
      // Update boardSize immediately from initialMatch
      if (initialMatch.board_size) {
        setBoardState((prev) => ({
          ...prev,
          boardSize: initialMatch.board_size,
        }));
      }
      loadMatchState(initialMatch.id).catch((err) => {
        console.error("Failed to load initial match state:", err);
      });
    }
  }, [initialMatch]);

  // Xác định màu quân cờ của người chơi và hiển thị thông báo cho PvP matches
  useEffect(() => {
    if (!currentMatch || !user) return;

    // Chỉ hiển thị thông báo cho PvP matches (không có ai_level)
    // Xác định màu quân cờ của người chơi (cho cả AI và PvP matches)
    // user.id có thể là UUID object hoặc string
    const userId = user.id || user.user_id;
    if (!userId) return;

    // Convert cả hai về string để so sánh (xử lý cả UUID object và string)
    const userIdStr = String(userId);
    let color = null;

    console.log("🎨 Determining player color:", {
      userId: userIdStr,
      black_player_id: currentMatch.black_player_id,
      white_player_id: currentMatch.white_player_id,
      ai_level: currentMatch.ai_level,
    });

    if (currentMatch.black_player_id) {
      const blackPlayerIdStr = String(currentMatch.black_player_id);
      if (blackPlayerIdStr === userIdStr) {
        color = "B"; // Người chơi là Black
      }
    }

    if (!color && currentMatch.white_player_id) {
      const whitePlayerIdStr = String(currentMatch.white_player_id);
      if (whitePlayerIdStr === userIdStr) {
        color = "W"; // Người chơi là White
      }
    }

    // Đối với AI match: nếu không tìm thấy user trong black/white player, xác định dựa trên player_id nào có giá trị
    if (!color && currentMatch.ai_level) {
      if (currentMatch.black_player_id) {
        color = "B"; // User là black player
      } else if (currentMatch.white_player_id) {
        color = "W"; // User là white player
      }
    }

    console.log("🎨 Determined player color:", color, {
      userId: userIdStr,
      blackPlayerId: currentMatch.black_player_id
        ? String(currentMatch.black_player_id)
        : null,
      whitePlayerId: currentMatch.white_player_id
        ? String(currentMatch.white_player_id)
        : null,
      matchId: currentMatch.id,
    });

    if (color) {
      setPlayerColor(color);
      console.log("✅ Set playerColor state to:", color);
      // Hiển thị modal thông báo khi vào game lần đầu
      // Kiểm tra xem đã hiển thị cho match này chưa
      const shownKey = `playerColorShown_${currentMatch.id}`;
      if (!localStorage.getItem(shownKey)) {
        setShowPlayerColorModal(true);
        localStorage.setItem(shownKey, "true");
      }
    } else {
      console.warn(
        "⚠️ Could not determine player color for match:",
        currentMatch.id
      );
    }
  }, [currentMatch, user]);

  // Real-time polling for PvP matches (không phải AI match)
  // eslint-disable-next-line react-hooks/exhaustive-deps
  useEffect(() => {
    // Chỉ poll cho PvP matches (không có ai_level)
    if (
      !currentMatch ||
      (currentMatch.ai_level !== null && currentMatch.ai_level !== undefined)
    ) {
      return; // Không poll cho AI matches
    }

    if (gameOver) {
      return; // Không poll nếu game đã kết thúc
    }

    const pollInterval = setInterval(async () => {
      try {
        // Chỉ poll khi không đang xử lý move
        if (isProcessing) {
          return;
        }

        const response = await api.get(`/matches/${currentMatch.id}`);
        const matchData = response.data;

        // QUAN TRỌNG: Kiểm tra match đã kết thúc chưa (từ result hoặc finished_at)
        if (matchData && (matchData.result || matchData.finished_at)) {
          console.log("🏁 Match finished detected in polling:", {
            result: matchData.result,
            finished_at: matchData.finished_at,
          });
          await setGameOverState(matchData);
          // Dừng polling khi game over
          clearInterval(pollInterval);
          return;
        }

        // Kiểm tra nếu white_player_id đã thay đổi (người chơi thứ 2 đã join)
        if (
          matchData &&
          matchData.white_player_id &&
          !currentMatch.white_player_id
        ) {
          console.log("🔄 Player 2 joined, updating match info...");
          await loadMatchState(currentMatch.id);
          return; // Đã cập nhật, không cần kiểm tra state nữa
        }

        // Kiểm tra nếu có state mới
        if (matchData.state) {
          const newState = matchData.state;
          const newStones = newState.board_position || {};
          const newMoveCount = Object.keys(newStones).length;
          const currentMoveCount = Object.keys(boardState.stones).length;

          // Nếu có thay đổi (đối thủ đã đánh)
          if (
            newMoveCount !== currentMoveCount ||
            newState.current_player !== boardState.currentPlayer ||
            newState.prisoners_black !== boardState.prisonersBlack ||
            newState.prisoners_white !== boardState.prisonersWhite
          ) {
            console.log("🔄 Opponent made a move, updating board state...");
            await loadMatchState(currentMatch.id);
          }
        }
      } catch (error) {
        console.error("Error polling match state:", error);
        // Không làm gì nếu lỗi, sẽ thử lại lần sau
      }
    }, 2000); // Poll mỗi 2 giây

    return () => {
      clearInterval(pollInterval);
    };
  }, [
    currentMatch,
    gameOver,
    isProcessing,
    boardState.stones,
    boardState.currentPlayer,
  ]);

  // Đếm ngược thời gian real-time cho PvP matches với time control
  // eslint-disable-next-line react-hooks/exhaustive-deps
  useEffect(() => {
    // Chỉ đếm ngược cho PvP matches (không có ai_level) và có time control
    if (
      !currentMatch ||
      (currentMatch.ai_level !== null && currentMatch.ai_level !== undefined)
    ) {
      return; // Không đếm ngược cho AI matches
    }

    if (gameOver) {
      return; // Không đếm ngược nếu game đã kết thúc
    }

    // Chỉ đếm ngược nếu có thời gian còn lại
    if (
      boardState.blackTimeRemaining === null &&
      boardState.whiteTimeRemaining === null
    ) {
      return;
    }

    const timerInterval = setInterval(() => {
      setBoardState((prev) => {
        let newBlackTime = prev.blackTimeRemaining;
        let newWhiteTime = prev.whiteTimeRemaining;

        // Chỉ đếm ngược cho người chơi hiện tại
        if (
          prev.currentPlayer === "B" &&
          newBlackTime !== null &&
          newBlackTime > 0
        ) {
          newBlackTime = Math.max(0, newBlackTime - 1);
        } else if (
          prev.currentPlayer === "W" &&
          newWhiteTime !== null &&
          newWhiteTime > 0
        ) {
          newWhiteTime = Math.max(0, newWhiteTime - 1);
        }

        // Nếu hết thời gian, không cần làm gì (backend sẽ xử lý)
        return {
          ...prev,
          blackTimeRemaining: newBlackTime,
          whiteTimeRemaining: newWhiteTime,
        };
      });
    }, 1000); // Cập nhật mỗi giây

    return () => {
      clearInterval(timerInterval);
    };
  }, [
    currentMatch,
    gameOver,
    boardState.currentPlayer,
    boardState.blackTimeRemaining,
    boardState.whiteTimeRemaining,
  ]);

  // Hiển thị dialog khi có tình trạng cướp cờ KO
  useEffect(() => {
    const currentKoPosition = boardState.koPosition;

    // Nếu có ko_position mới (khác với previous) và không phải null → hiển thị dialog
    if (
      currentKoPosition &&
      JSON.stringify(currentKoPosition) !== JSON.stringify(previousKoPosition)
    ) {
      console.log("🔔 KO position detected:", currentKoPosition);
      setKoPosition(currentKoPosition);
      setShowKoDialog(true);
      setPreviousKoPosition(currentKoPosition);
    } else if (!currentKoPosition && previousKoPosition) {
      // Nếu ko_position bị clear (từ có về không có) → reset previous
      setPreviousKoPosition(null);
    }
  }, [boardState.koPosition, previousKoPosition]);

  // Helper function để format thời gian (MM:SS)
  const formatTime = (seconds) => {
    if (seconds === null || seconds === undefined) return "--:--";
    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return `${mins.toString().padStart(2, "0")}:${secs
      .toString()
      .padStart(2, "0")}`;
  };

  // Helper function để set game over state (tránh duplicate code)
  const setGameOverState = async (matchData) => {
    if (!matchData) return;

    const result = matchData.result;
    const finishedAt = matchData.finished_at;

    // Chỉ set nếu match thực sự đã kết thúc
    if (!result && !finishedAt) {
      return;
    }

    console.log("🏁 Setting game over state:", { result, finishedAt });
    setGameOver(true);

    if (result) {
      setGameResult(result);
      const resultMsg = formatGameResult(result);
      setGameOverMessage(resultMsg);

      // Tính toán chi tiết điểm số (chỉ khi không phải resign)
      if (!result.endsWith("+R")) {
        const scoreDetails = calculateScoreDetails(
          boardState.stones,
          boardState.boardSize
        );
        setGameScoreDetails(scoreDetails);
      } else {
        setGameScoreDetails(null);
      }

      // Load ELO cuối trận đấu (chỉ cho PvP matches)
      if (!currentMatch?.ai_level) {
        try {
          await loadFinalElo(matchData);
        } catch (err) {
          console.error("Failed to load final ELO:", err);
        }
      }

      // Hiển thị modal game over
      setTimeout(() => {
        setShowGameOverModal(true);
      }, 500);
    }
  };

  // Verify authentication on mount
  useEffect(() => {
    const token = localStorage.getItem("access_token");
    if (!token) {
      console.warn("No token found - user should be redirected to login");
    }
  }, []);

  const loadMatchState = async (matchId) => {
    try {
      // Use /matches/{match_id} endpoint which includes state
      const response = await api.get(`/matches/${matchId}`);
      // Kiểm tra xem match có phải AI match không
      const isAiMatch =
        response.data &&
        response.data.ai_level !== null &&
        response.data.ai_level !== undefined;

      if (response.data && response.data.state) {
        const state = response.data.state;

        // QUAN TRỌNG: Dùng board_position từ backend (backend đã đảm bảo màu đúng)
        // Backend là source of truth - không rebuild ở frontend
        let stones = {};
        if (
          state.board_position &&
          typeof state.board_position === "object" &&
          !Array.isArray(state.board_position)
        ) {
          // Backend đã đảm bảo màu đúng trong board_position
          stones = { ...state.board_position }; // Tạo copy để tránh mutation
          console.log("✅ Using board_position from backend:", stones);
        }
        console.log("🎮 Backend state:", {
          current_player: state.current_player,
          moves_count: state.moves?.length,
          board_position: stones,
        });
        // Nếu không có board_position, stones sẽ là {} (empty board)

        // Get last move
        let lastMove = null;
        if (state.moves && state.moves.length > 0) {
          const lastMoveData = state.moves[state.moves.length - 1];
          if (
            lastMoveData &&
            Array.isArray(lastMoveData) &&
            lastMoveData.length === 2
          ) {
            lastMove = { x: lastMoveData[0], y: lastMoveData[1] };
          }
        }

        // Update move history - convert moves từ backend sang format đúng
        // CHỈ cập nhật nếu số lượng moves từ backend lớn hơn số lượng moves hiện tại
        // Để tránh ghi đè moves đã được thêm local (như AI pass hoặc user moves với màu đúng)
        if (state.moves && Array.isArray(state.moves)) {
          setMoveHistory((prev) => {
            // Nếu backend có nhiều moves hơn, chỉ thêm moves mới, không ghi đè toàn bộ
            if (state.moves.length > prev.length) {
              // Chỉ convert và thêm các moves mới (từ index prev.length trở đi)
              const newMoves = state.moves
                .slice(prev.length)
                .map((move, relativeIndex) => {
                  const index = prev.length + relativeIndex;
                  // Trong AI match: moves chẵn (0, 2, 4...) là Black (user), moves lẻ (1, 3, 5...) là White (AI)
                  // Force màu dựa trên index để tránh màu sai từ backend
                  const correctColor = index % 2 === 0 ? "B" : "W";

                  // Move có thể là array [x, y] hoặc object với position
                  if (Array.isArray(move) && move.length === 2) {
                    // Legacy format: [x, y]
                    return {
                      number: index + 1,
                      color: isAiMatch
                        ? correctColor
                        : index % 2 === 0
                        ? "B"
                        : "W",
                      position: move,
                      captured: [],
                    };
                  } else if (move && typeof move === "object") {
                    // Object format - có thể có position, x/y, hoặc null (pass)
                    let position = null;
                    if (move.position !== null && move.position !== undefined) {
                      if (
                        Array.isArray(move.position) &&
                        move.position.length === 2
                      ) {
                        position = move.position;
                      } else if (
                        move.position.x !== undefined &&
                        move.position.y !== undefined
                      ) {
                        position = [move.position.x, move.position.y];
                      }
                    } else if (move.x !== undefined && move.y !== undefined) {
                      position = [move.x, move.y];
                    }

                    // Trong AI match, force màu dựa trên index thay vì dùng move.color từ backend
                    // Vì backend có thể đã lưu sai màu
                    return {
                      number: move.number || index + 1,
                      color: isAiMatch
                        ? correctColor
                        : move.color || (index % 2 === 0 ? "B" : "W"),
                      position: position,
                      captured: move.captured || [],
                    };
                  }
                  return null;
                })
                .filter((move) => move !== null);

              console.log(
                "📝 Adding new moves from backend:",
                newMoves.length,
                "moves (prev:",
                prev.length,
                ", backend total:",
                state.moves.length,
                ")"
              );
              return [...prev, ...newMoves];
            } else {
              // Giữ nguyên move history hiện tại nếu backend không có thêm moves
              // Điều này bảo vệ moves đã được thêm local với màu đúng
              console.log(
                "📝 Keeping current move history (backend has",
                state.moves.length,
                "moves, current has",
                prev.length,
                ")"
              );
              return prev;
            }
          });
        }

        setBoardState((prev) => {
          // Log để debug nếu có sự khác biệt
          const prevStonesKeys = Object.keys(prev.stones).sort();
          const newStonesKeys = Object.keys(stones).sort();
          if (
            prevStonesKeys.length !== newStonesKeys.length ||
            JSON.stringify(prevStonesKeys) !== JSON.stringify(newStonesKeys)
          ) {
            console.log("🔄 Board state changed in loadMatchState:", {
              prevCount: prevStonesKeys.length,
              newCount: newStonesKeys.length,
              prevKeys: prevStonesKeys.slice(0, 10),
              newKeys: newStonesKeys.slice(0, 10),
              removed: prevStonesKeys.filter((k) => !newStonesKeys.includes(k)),
              added: newStonesKeys.filter((k) => !prevStonesKeys.includes(k)),
            });
          }

          // QUAN TRỌNG: Luôn dùng board state từ backend (board_position) để đảm bảo đồng bộ hoàn toàn
          // Backend là source of truth cho board state, đặc biệt sau khi có captured stones
          // QUAN TRỌNG: Trong AI match, user luôn là Black, AI luôn là White
          // Đảm bảo currentPlayer đúng sau reload
          let correctCurrentPlayer =
            state.current_player || state.to_move || prev.currentPlayer || "B";
          if (isAiMatch) {
            // Trong AI match, nếu currentPlayer không phải 'B' hoặc 'W', có thể bị sai
            // Nhưng vẫn dùng từ backend vì backend có thể đúng
            // Chỉ log để debug
            if (correctCurrentPlayer !== "B" && correctCurrentPlayer !== "W") {
              console.warn(
                `⚠️ Invalid currentPlayer from backend: ${correctCurrentPlayer}, using previous: ${prev.currentPlayer}`
              );
              correctCurrentPlayer = prev.currentPlayer || "B";
            }
          }

          // Lấy ko_position từ state
          let koPositionValue = null;
          if (
            state.ko_position &&
            Array.isArray(state.ko_position) &&
            state.ko_position.length === 2
          ) {
            koPositionValue = state.ko_position;
          }

          return {
            ...prev,
            stones, // Dùng stones từ backend (đã được sửa màu nếu là AI match)
            lastMove,
            boardSize: state.size || prev.boardSize || 9, // Update boardSize from state
            prisonersBlack:
              state.prisoners_black !== undefined
                ? state.prisoners_black
                : prev.prisonersBlack,
            prisonersWhite:
              state.prisoners_white !== undefined
                ? state.prisoners_white
                : prev.prisonersWhite,
            currentPlayer: correctCurrentPlayer,
            blackTimeRemaining:
              state.black_time_remaining_seconds !== undefined
                ? state.black_time_remaining_seconds
                : prev.blackTimeRemaining,
            whiteTimeRemaining:
              state.white_time_remaining_seconds !== undefined
                ? state.white_time_remaining_seconds
                : prev.whiteTimeRemaining,
            koPosition: koPositionValue,
          };
        });

        // Cập nhật currentMatch với thông tin mới nhất từ backend (đặc biệt là white_player_id)
        if (response.data) {
          setCurrentMatch((prev) => {
            // Chỉ cập nhật nếu có thay đổi quan trọng (như white_player_id)
            if (
              prev &&
              (prev.white_player_id !== response.data.white_player_id ||
                prev.black_player_id !== response.data.black_player_id ||
                prev.black_player_username !==
                  response.data.black_player_username ||
                prev.white_player_username !==
                  response.data.white_player_username)
            ) {
              console.log("🔄 Updating currentMatch with new player info:", {
                white_player_id: response.data.white_player_id,
                black_player_id: response.data.black_player_id,
              });
              return {
                ...prev,
                white_player_id: response.data.white_player_id,
                black_player_id: response.data.black_player_id,
                black_player_username: response.data.black_player_username,
                white_player_username: response.data.white_player_username,
              };
            }
            return prev;
          });
        }

        // QUAN TRỌNG: Kiểm tra game over từ match data (result hoặc finished_at)
        if (
          response.data &&
          (response.data.result || response.data.finished_at)
        ) {
          console.log("🏁 Game over detected in loadMatchState:", {
            result: response.data.result,
            finished_at: response.data.finished_at,
          });
          await setGameOverState(response.data);
        } else {
          // Chỉ reset gameOver nếu match thực sự chưa kết thúc
          // (tránh reset khi đang trong quá trình kết thúc)
          if (
            !response.data ||
            (!response.data.result && !response.data.finished_at)
          ) {
            setGameOver(false);
            setGameResult(null);
          }
        }

        console.log("✅ Loaded match state:", {
          stones,
          lastMove,
          boardSize: state.size,
          prisoners: {
            black: state.prisoners_black,
            white: state.prisoners_white,
          },
          gameOver: !!response.data?.result,
        });
      } else {
        // No state available - match might be new or empty
        // But we still need to get boardSize from match
        console.log("⚠️ No match state available - match might be new");
        if (response.data && response.data.board_size) {
          setBoardState((prev) => ({
            ...prev,
            stones: {},
            lastMove: null,
            boardSize: response.data.board_size, // Use board_size from match
            prisonersBlack: 0,
            prisonersWhite: 0,
            currentPlayer: "B",
            koPosition: null,
          }));
        } else {
          // Fallback: keep current boardSize
          setBoardState((prev) => ({
            ...prev,
            stones: {},
            lastMove: null,
            prisonersBlack: 0,
            prisonersWhite: 0,
            currentPlayer: "B",
            koPosition: null,
          }));
        }
      }
    } catch (error) {
      console.error("Failed to load match state:", error);
      // Don't throw - just log, allow user to continue
    }
  };

  const loadInitialData = async () => {
    try {
      // Load match history and statistics in parallel
      const [matchesRes, statsRes] = await Promise.all([
        api.get("/matches/history"),
        api.get("/statistics/me"),
      ]);
      setMatchHistory(matchesRes.data || []);
      setStatistics(statsRes.data);
    } catch (error) {
      console.error("Failed to load initial data:", error);
      // Set empty arrays to prevent UI errors
      setMatchHistory([]);
    }
  };

  // Expose refresh function cho tương lai (hiện chưa dùng trực tiếp trong UI)
  // eslint-disable-next-line no-unused-vars
  const refreshData = async () => {
    setIsDataLoaded(false);
    await loadInitialData();
    setIsDataLoaded(true);
  };

  const handleCreateMatch = async (
    matchType,
    level,
    boardSize,
    playerColor = "black"
  ) => {
    console.log("🎮 handleCreateMatch called with:", {
      matchType,
      level,
      boardSize,
      playerColor,
    });
    try {
      // Reset tất cả state liên quan đến game over trước khi tạo trận mới
      setGameOver(false);
      setGameResult(null);
      setShowGameOverModal(false);
      setGameOverMessage(null);
      setGameScoreDetails(null);
      setIsProcessing(false);
      setMoveHistory([]);

      // Reset counter âm thanh khi bắt đầu trận mới
      resetStoneSoundCounter();

      let response;
      if (matchType === "ai") {
        // Gửi player_color để backend biết người chơi muốn cầm quân gì
        const requestBody = {
          level,
          board_size: boardSize,
          player_color: playerColor,
        };
        console.log(
          "🎨 Creating AI match with request body:",
          JSON.stringify(requestBody)
        );
        response = await api.post("/matches/ai", requestBody);
        console.log("🎨 Match created:", response.data);
      } else {
        response = await api.post("/matches/pvp", { board_size: boardSize });
      }

      const match = response.data.match || response.data;
      setCurrentMatch(match);
      setBoardState({
        stones: {},
        boardSize: match.board_size || boardSize,
        lastMove: null,
        prisonersBlack: 0,
        prisonersWhite: 0,
        currentPlayer: "B",
      });
      setShowMatchDialog(false);
      await loadMatchState(match.id);
      await loadInitialData();
    } catch (error) {
      alert(
        "Không thể tạo trận đấu: " +
          (error.response?.data?.detail || error.message)
      );
    }
  };

  const handleBoardClick = async (x, y) => {
    if (!currentMatch || isProcessing || gameOver) {
      console.log(
        "⚠️ Ignoring click - no match, already processing, or game over"
      );
      return;
    }

    // QUAN TRỌNG: Double-check game over từ backend trước khi process
    try {
      const matchCheckResponse = await api.get(`/matches/${currentMatch.id}`);
      if (
        matchCheckResponse.data?.result ||
        matchCheckResponse.data?.finished_at
      ) {
        console.log("🏁 Match already finished, setting game over state");
        await setGameOverState(matchCheckResponse.data);
        return;
      }
    } catch (error) {
      console.error("Error checking match status:", error);
      // Continue với move nếu check fail (có thể là network issue)
    }

    // Check đúng lượt cho cả AI và PvP matches
    // Sử dụng playerColor state (đã được set trong useEffect) thay vì tính lại
    // Nếu playerColor chưa được set, tính lại từ currentMatch
    let userColor = playerColor;

    console.log("🎯 handleBoardClick - Turn check:", {
      playerColorState: playerColor,
      currentPlayer: boardState.currentPlayer,
      matchId: currentMatch.id,
      blackPlayerId: currentMatch.black_player_id,
      whitePlayerId: currentMatch.white_player_id,
      userId: user?.id,
    });

    if (!userColor) {
      // Fallback: tính lại nếu playerColor chưa được set
      console.log(
        "⚠️ playerColor state not set, calculating from currentMatch..."
      );
      const userIdStr = String(user?.id || "");

      if (currentMatch.ai_level) {
        // AI match: xác định màu user dựa trên player_id
        if (currentMatch.black_player_id) {
          userColor = "B"; // User là black
        } else if (currentMatch.white_player_id) {
          userColor = "W"; // User là white
        }
      } else {
        // PvP match: kiểm tra cả black và white player
        const blackPlayerIdStr = String(currentMatch.black_player_id || "");
        const whitePlayerIdStr = String(currentMatch.white_player_id || "");

        if (blackPlayerIdStr === userIdStr) {
          userColor = "B";
        } else if (whitePlayerIdStr === userIdStr) {
          userColor = "W";
        }

        console.log("🔍 Calculated userColor from match:", {
          userColor,
          userIdStr,
          blackPlayerIdStr,
          whitePlayerIdStr,
          match:
            blackPlayerIdStr === userIdStr || whitePlayerIdStr === userIdStr,
        });

        // Check đủ người chơi
        if (!currentMatch.black_player_id || !currentMatch.white_player_id) {
          alert("Chưa đủ người chơi. Vui lòng đợi người chơi khác tham gia.");
          return;
        }
      }
    }

    // Check đúng lượt
    if (!userColor) {
      console.warn("⚠️ Cannot determine user color", {
        playerColor,
        currentMatch: {
          id: currentMatch.id,
          black_player_id: currentMatch.black_player_id,
          white_player_id: currentMatch.white_player_id,
          ai_level: currentMatch.ai_level,
        },
        userId: user?.id,
      });
      alert("Không thể xác định màu quân của bạn. Vui lòng thử lại.");
      return;
    }

    if (boardState.currentPlayer !== userColor) {
      console.log(
        `⚠️ Not your turn. Current: ${boardState.currentPlayer}, You: ${userColor}, playerColor state: ${playerColor}`,
        {
          matchId: currentMatch.id,
          boardStateCurrentPlayer: boardState.currentPlayer,
          userColor,
          playerColorState: playerColor,
        }
      );
      alert(
        `Không phải lượt của bạn. Hiện tại là lượt của ${
          boardState.currentPlayer === "B" ? "Đen" : "Trắng"
        }`
      );
      return;
    }

    console.log("✅ Turn check passed:", {
      currentPlayer: boardState.currentPlayer,
      userColor,
    });

    // Check if position already has a stone
    const key = `${x},${y}`;
    if (boardState.stones[key]) {
      console.log("⚠️ Ignoring click - position already occupied");
      return;
    }

    setIsProcessing(true);
    try {
      // QUAN TRỌNG: Sử dụng currentPlayer từ state thay vì tính từ số lượng stones
      // Vì số lượng stones có thể không phản ánh đúng số move (do captured stones)
      const color = boardState.currentPlayer || "B";
      // Sử dụng moveHistory.length để tính moveNumber chính xác (bao gồm cả pass moves)
      const moveNumber = moveHistory.length + 1;

      console.log("🎯 Making move:", {
        x,
        y,
        color,
        moveNumber,
        currentPlayer: boardState.currentPlayer,
        currentStonesCount: Object.keys(boardState.stones).length,
        moveHistoryLength: moveHistory.length,
      });

      // Phát âm thanh đánh cờ (nếu bật)
      if (settings.soundEnabled) {
        playStoneSound("/assets/zz-un-floor-goban-rich.v7.webm", true);
      }

      // Không cần optimistic update nữa vì sẽ dùng board_diff từ response
      // Optimistic update có thể conflict với captured stones

      // Dùng api instance chung (có interceptor refresh token)
      // Set timeout riêng cho request này (AI có thể mất nhiều thời gian)
      const response = await api.post(
        `/matches/${currentMatch.id}/move`,
        {
          x,
          y,
          move_number: moveNumber,
          color,
        },
        {
          timeout: 60000, // 60 seconds for AI moves
        }
      );

      console.log("✅ Move response:", response.data);
      console.log("📋 Response keys:", Object.keys(response.data));
      console.log("📋 Response details:", {
        has_captured: !!response.data.captured,
        captured_count: response.data.captured?.length || 0,
        has_board_diff: !!response.data.board_diff,
        has_prisoners:
          "prisoners_black" in response.data &&
          "prisoners_white" in response.data,
      });

      // Xử lý captured stones và board_diff từ response
      const moveData = response.data;
      const moveKey = `${x},${y}`;

      // ĐƠN GIẢN HÓA: Chỉ dùng board_diff để biết vị trí thêm/xóa, KHÔNG dùng màu từ backend
      // Luôn force màu đúng: user = currentPlayer (Black trong AI match)
      setBoardState((prev) => {
        try {
          const newStones = { ...prev.stones };

          // BƯỚC 1: Xóa quân bị bắt (từ board_diff.removed hoặc captured array)
          if (
            moveData.board_diff &&
            moveData.board_diff.removed &&
            Array.isArray(moveData.board_diff.removed)
          ) {
            moveData.board_diff.removed.forEach((key) => {
              if (newStones[key]) {
                delete newStones[key];
                console.log(`🗑️ Removed captured stone at ${key}`);
              }
            });
          } else if (moveData.captured && Array.isArray(moveData.captured)) {
            moveData.captured.forEach(([cx, cy]) => {
              const capturedKey = `${cx},${cy}`;
              if (newStones[capturedKey]) {
                delete newStones[capturedKey];
                console.log(`🗑️ Removed captured stone at ${capturedKey}`);
              }
            });
          }

          // BƯỚC 2: Thêm quân user mới - LUÔN force màu từ currentPlayer (KHÔNG dùng màu từ backend)
          // Trong AI match: user luôn là Black, nên force màu 'B'
          if (
            moveData.board_diff &&
            moveData.board_diff.added &&
            typeof moveData.board_diff.added === "object"
          ) {
            // Chỉ lấy vị trí từ added, không dùng màu
            Object.keys(moveData.board_diff.added).forEach((key) => {
              // Force màu từ currentPlayer (đã lấy ở đầu hàm)
              newStones[key] = color;
              console.log(
                `➕ Added user stone at ${key}: ${color} (forced, ignoring backend color)`
              );
            });
          } else {
            // Fallback: Thêm quân ở vị trí user đánh
            newStones[moveKey] = color;
            console.log(`➕ Added user stone at ${moveKey}: ${color} (forced)`);
          }

          const updatedState = {
            ...prev,
            stones: newStones,
            prisonersBlack:
              moveData.prisoners_black !== undefined
                ? moveData.prisoners_black
                : prev.prisonersBlack,
            prisonersWhite:
              moveData.prisoners_white !== undefined
                ? moveData.prisoners_white
                : prev.prisonersWhite,
            currentPlayer: moveData.current_player || prev.currentPlayer,
            lastMove: { x, y },
          };

          console.log("📊 Updated board state:", {
            stonesCount: Object.keys(updatedState.stones).length,
            prisonersBlack: updatedState.prisonersBlack,
            prisonersWhite: updatedState.prisonersWhite,
            captured: moveData.captured,
            board_diff: moveData.board_diff,
          });

          return updatedState;
        } catch (stateError) {
          console.error("❌ Error in setBoardState callback:", stateError);
          // Trả về state cũ với quân mới được thêm đơn giản
          return {
            ...prev,
            stones: { ...prev.stones, [`${x},${y}`]: color },
            lastMove: { x, y },
          };
        }
      });

      // Update move history (will be synced with loadMatchState later, but update now for responsiveness)
      // QUAN TRỌNG: Sử dụng màu từ currentPlayer trước khi đánh, không phải từ backend
      setMoveHistory((prev) => {
        const newMove = {
          number: prev.length + 1,
          color: color, // Sử dụng màu từ boardState.currentPlayer (đã lấy ở đầu hàm)
          position: [x, y],
          captured: moveData.captured || [],
        };
        console.log(
          "📝 Adding user move to history:",
          newMove,
          "from currentPlayer:",
          color
        );
        return [...prev, newMove];
      });

      // Xử lý game over từ response
      if (moveData.game_over) {
        // Reload match để lấy result đầy đủ
        try {
          const matchResponse = await api.get(`/matches/${currentMatch.id}`);
          if (matchResponse.data) {
            await setGameOverState(matchResponse.data);
          }
        } catch (err) {
          console.error("Failed to load game result:", err);
          // Fallback: set game over với data hiện có
          setGameOver(true);
        }
      }

      // Xử lý AI move nếu có (sau khi đã xử lý user move)
      if (moveData.ai_move) {
        const aiMove = moveData.ai_move;
        console.log("🤖 AI move received:", aiMove);

        // Đợi một chút để user thấy move của mình trước
        await new Promise((resolve) => setTimeout(resolve, 500));

        // Phát âm thanh đánh cờ cho AI move (nếu bật và không phải pass)
        if (settings.soundEnabled && !aiMove.is_pass) {
          playStoneSound("/assets/zz-un-floor-goban-rich.v7.webm", true);
        }

        // Cập nhật board với AI move
        // ĐƠN GIẢN HÓA: Chỉ dùng board_diff để biết vị trí, LUÔN force màu 'W' cho AI
        setBoardState((prev) => {
          const newStones = { ...prev.stones };

          // BƯỚC 1: Xóa quân bị AI bắt
          if (
            aiMove.board_diff &&
            aiMove.board_diff.removed &&
            Array.isArray(aiMove.board_diff.removed)
          ) {
            aiMove.board_diff.removed.forEach((key) => {
              if (newStones[key]) {
                delete newStones[key];
                console.log(`🗑️ AI removed captured stone at ${key}`);
              }
            });
          }

          // BƯỚC 2: Thêm quân AI - LUÔN force màu 'W' (KHÔNG dùng màu từ backend)
          // Xác định màu AI dựa trên player IDs - AI là bên không có player_id
          const aiColor = currentMatch.black_player_id ? "W" : "B";
          const userColor = currentMatch.black_player_id ? "B" : "W";

          if (
            aiMove.board_diff &&
            aiMove.board_diff.added &&
            typeof aiMove.board_diff.added === "object"
          ) {
            // Chỉ lấy vị trí từ added
            Object.keys(aiMove.board_diff.added).forEach((key) => {
              newStones[key] = aiColor;
              console.log(`🤖 AI added stone at ${key}: ${aiColor}`);
            });
          } else if (
            aiMove.x !== undefined &&
            aiMove.y !== undefined &&
            aiMove.x !== null &&
            aiMove.y !== null
          ) {
            // Fallback: Thêm quân AI trực tiếp
            const aiKey = `${aiMove.x},${aiMove.y}`;
            newStones[aiKey] = aiColor;
            console.log(`🤖 AI placed stone at ${aiKey}: ${aiColor}`);
          }

          return {
            ...prev,
            stones: newStones,
            prisonersBlack:
              aiMove.prisoners_black !== undefined
                ? aiMove.prisoners_black
                : prev.prisonersBlack,
            prisonersWhite:
              aiMove.prisoners_white !== undefined
                ? aiMove.prisoners_white
                : prev.prisonersWhite,
            currentPlayer: userColor, // Sau AI move, đến lượt user
            lastMove:
              aiMove.x !== undefined
                ? { x: aiMove.x, y: aiMove.y }
                : prev.lastMove,
          };
        });

        // Xác định màu AI cho move history
        const aiColorForHistory = currentMatch.black_player_id ? "W" : "B";

        // Update move history for AI move
        setMoveHistory((prev) => {
          const moveNumber = prev.length + 1;

          // Kiểm tra AI pass trước - ưu tiên is_pass flag
          if (
            aiMove.is_pass === true ||
            (aiMove.x === null && aiMove.y === null)
          ) {
            // AI passed
            const newMove = {
              number: moveNumber,
              color: aiColorForHistory,
              position: null,
              captured: [],
            };
            console.log("📝 Adding AI pass to history:", newMove);
            return [...prev, newMove];
          }

          // Kiểm tra nhiều format: position array, x/y riêng, hoặc position object
          let position = null;
          if (aiMove.position !== null && aiMove.position !== undefined) {
            // position có thể là array [x, y] hoặc object {x, y}
            if (
              Array.isArray(aiMove.position) &&
              aiMove.position.length === 2
            ) {
              position = aiMove.position;
            } else if (
              aiMove.position.x !== undefined &&
              aiMove.position.y !== undefined
            ) {
              position = [aiMove.position.x, aiMove.position.y];
            }
          } else if (
            aiMove.x !== null &&
            aiMove.x !== undefined &&
            aiMove.y !== null &&
            aiMove.y !== undefined
          ) {
            // Fallback: x và y riêng biệt (phải khác null và undefined)
            position = [aiMove.x, aiMove.y];
          }

          if (position !== null) {
            // AI đã đánh một nước
            const newMove = {
              number: moveNumber,
              color: aiColorForHistory,
              position: position,
              captured: aiMove.captured || [],
            };
            console.log("📝 Adding AI move to history:", newMove);
            return [...prev, newMove];
          } else {
            // Fallback: Nếu không có position và không phải pass rõ ràng, vẫn ghi là pass
            const newMove = {
              number: moveNumber,
              color: aiColorForHistory,
              position: null,
              captured: [],
            };
            console.log("📝 Adding AI pass to history (fallback):", newMove);
            return [...prev, newMove];
          }
        });

        // Kiểm tra game over sau AI move (có thể AI pass dẫn đến 2 lần pass liên tiếp)
        if (aiMove.game_over || moveData.game_over) {
          console.log("🏁 Game over detected after AI move:", {
            aiMoveGameOver: aiMove.game_over,
            moveDataGameOver: moveData.game_over,
          });
          // Reload match để lấy result đầy đủ
          try {
            const matchResponse = await api.get(`/matches/${currentMatch.id}`);
            if (matchResponse.data) {
              await setGameOverState(matchResponse.data);
            }
          } catch (err) {
            console.error("Failed to load game result:", err);
            // Fallback: set game over với data hiện có
            setGameOver(true);
          }
        }

        // Đợi một chút để đảm bảo MongoDB đã được cập nhật
        await new Promise((resolve) => setTimeout(resolve, 500));

        // Reload match state sau AI move để đảm bảo sync và kiểm tra game over
        console.log("🔄 Reloading match state after AI move...");
        await loadMatchState(currentMatch.id);

        // Kiểm tra lại game over sau khi reload (backend có thể đã cập nhật)
        const finalMatchResponse = await api.get(`/matches/${currentMatch.id}`);
        if (finalMatchResponse.data) {
          await setGameOverState(finalMatchResponse.data);
        }
      }

      // KHÔNG reload board state ngay sau user move vì:
      // 1. board_diff từ response đã đủ để cập nhật state đúng
      // 2. loadMatchState có thể overwrite màu đúng với màu sai từ board_position
      // 3. Tránh race condition giữa user move và AI move
      // Chỉ reload khi cần thiết (ví dụ: sau AI move hoặc khi user reload trang)
      console.log(
        "✅ User move completed, skipping reload to avoid color overwrite"
      );

      // Reload match list
      await loadInitialData();
    } catch (error) {
      console.error("❌ Move error:", error);
      console.error("Error details:", {
        status: error.response?.status,
        statusText: error.response?.statusText,
        data: error.response?.data,
        message: error.message,
        code: error.code,
      });

      // Revert optimistic update on error (nếu có)
      const key = `${x},${y}`;
      if (boardState.stones[key]) {
        const revertedStones = { ...boardState.stones };
        delete revertedStones[key];
        setBoardState((prev) => ({
          ...prev,
          stones: revertedStones,
          lastMove: prev.lastMove,
        }));
      }

      if (error.response?.status === 401) {
        console.error("🔓 401 Unauthorized - Token expired or invalid");
        alert("Session expired. Please login again.");
        if (onLogout) onLogout();
      } else if (error.code === "ECONNABORTED") {
        alert("Move timeout: AI is taking too long. Please wait or try again.");
      } else {
        const errorMessage =
          error.response?.data?.detail || error.message || "";
        // Kiểm tra nếu lỗi liên quan đến KO rule
        if (
          errorMessage.includes("Ko rule") ||
          errorMessage.includes("ko rule") ||
          errorMessage.includes("KO")
        ) {
          // Hiển thị dialog thay vì alert
          setKoPosition(boardState.koPosition || [x, y]);
          setShowKoDialog(true);
        } else {
          alert("Failed to submit move: " + errorMessage);
        }
      }
    } finally {
      setIsProcessing(false);
    }
  };

  const handlePass = async () => {
    if (!currentMatch || isProcessing || gameOver) return;

    // QUAN TRỌNG: Double-check game over từ backend trước khi process
    try {
      const matchCheckResponse = await api.get(`/matches/${currentMatch.id}`);
      if (
        matchCheckResponse.data?.result ||
        matchCheckResponse.data?.finished_at
      ) {
        console.log("🏁 Match already finished, setting game over state");
        await setGameOverState(matchCheckResponse.data);
        return;
      }
    } catch (error) {
      console.error("Error checking match status:", error);
      // Continue với pass nếu check fail
    }

    // Check đúng lượt cho PvP matches
    // Sử dụng playerColor state (đã được set trong useEffect) thay vì tính lại
    let userColor = playerColor;

    if (!userColor) {
      // Fallback: tính lại nếu playerColor chưa được set
      const userIdStr = String(user?.id || "");

      if (currentMatch.ai_level) {
        // AI match: xác định màu user dựa trên player_id
        if (currentMatch.black_player_id) {
          userColor = "B"; // User là black
        } else if (currentMatch.white_player_id) {
          userColor = "W"; // User là white
        }
      } else {
        // PvP match: kiểm tra cả black và white player
        const blackPlayerIdStr = String(currentMatch.black_player_id || "");
        const whitePlayerIdStr = String(currentMatch.white_player_id || "");

        if (blackPlayerIdStr === userIdStr) {
          userColor = "B";
        } else if (whitePlayerIdStr === userIdStr) {
          userColor = "W";
        }

        // Check đủ người chơi
        if (!currentMatch.black_player_id || !currentMatch.white_player_id) {
          alert("Chưa đủ người chơi. Vui lòng đợi người chơi khác tham gia.");
          return;
        }
      }
    }

    if (!userColor) {
      console.warn("⚠️ Cannot determine user color for pass");
      alert("Không thể xác định màu quân của bạn. Vui lòng thử lại.");
      return;
    }

    if (boardState.currentPlayer !== userColor) {
      alert(
        `Không phải lượt của bạn. Hiện tại là lượt của ${
          boardState.currentPlayer === "B" ? "Đen" : "Trắng"
        }`
      );
      return;
    }

    setIsProcessing(true);
    try {
      // Lấy số moves thực tế từ match state
      const matchResponse = await api.get(`/matches/${currentMatch.id}`);
      const currentMoves = matchResponse.data?.state?.moves || [];
      const moveNumber = currentMoves.length + 1;

      // Sử dụng currentPlayer từ boardState (đã được sync từ backend)
      const color = boardState.currentPlayer;

      console.log("⏭️ Passing:", {
        moveNumber,
        color,
        currentPlayer: boardState.currentPlayer,
      });

      // Dùng api instance chung (có interceptor refresh token)
      // Set timeout riêng cho request này (AI có thể mất nhiều thời gian)
      const passResponse = await api.post(
        `/matches/${currentMatch.id}/pass`,
        {
          move_number: moveNumber,
          color,
        },
        {
          timeout: 60000, // 60 seconds for AI response
        }
      );

      // Update move history for pass
      setMoveHistory((prev) => {
        const newMove = {
          number: prev.length + 1,
          color,
          position: null,
          captured: [],
        };
        return [...prev, newMove];
      });

      // Xử lý game over từ pass response
      if (passResponse.data && passResponse.data.game_over) {
        // Reload match để lấy result đầy đủ
        try {
          const matchResponse = await api.get(`/matches/${currentMatch.id}`);
          if (matchResponse.data) {
            await setGameOverState(matchResponse.data);
          }
        } catch (err) {
          console.error("Failed to load game result:", err);
          // Fallback: set game over với data hiện có
          setGameOver(true);
        }
      }

      // Xử lý AI move nếu có từ response
      if (passResponse.data && passResponse.data.ai_move) {
        const aiMove = passResponse.data.ai_move;
        console.log("🤖 AI move after pass:", aiMove);

        // Đợi một chút để user thấy pass được ghi nhận
        await new Promise((resolve) => setTimeout(resolve, 500));

        // Phát âm thanh đánh cờ cho AI move sau pass (nếu bật và không phải pass)
        if (settings.soundEnabled && !aiMove.is_pass) {
          playStoneSound("/assets/zz-un-floor-goban-rich.v7.webm", true);
        }

        // Cập nhật board với AI move
        setBoardState((prev) => {
          const newStones = { ...prev.stones };

          // ĐƠN GIẢN HÓA: Chỉ dùng board_diff để biết vị trí, LUÔN force màu 'W' cho AI
          // BƯỚC 1: Xóa quân bị AI bắt
          if (
            aiMove.board_diff &&
            aiMove.board_diff.removed &&
            Array.isArray(aiMove.board_diff.removed)
          ) {
            aiMove.board_diff.removed.forEach((key) => {
              if (newStones[key]) {
                delete newStones[key];
                console.log(`🗑️ AI removed captured stone at ${key}`);
              }
            });
          }

          // Xác định màu AI dựa trên player IDs
          const aiColorPass = currentMatch.black_player_id ? "W" : "B";
          const userColorPass = currentMatch.black_player_id ? "B" : "W";

          // BƯỚC 2: Thêm quân AI
          if (
            aiMove.board_diff &&
            aiMove.board_diff.added &&
            typeof aiMove.board_diff.added === "object"
          ) {
            Object.keys(aiMove.board_diff.added).forEach((key) => {
              newStones[key] = aiColorPass;
              console.log(`🤖 AI added stone at ${key}: ${aiColorPass}`);
            });
          } else if (
            aiMove.x !== undefined &&
            aiMove.y !== undefined &&
            aiMove.x !== null &&
            aiMove.y !== null
          ) {
            // Fallback: Thêm quân AI trực tiếp
            const aiKey = `${aiMove.x},${aiMove.y}`;
            newStones[aiKey] = aiColorPass;
            console.log(`🤖 AI placed stone at ${aiKey}: ${aiColorPass}`);
          }

          return {
            ...prev,
            stones: newStones,
            prisonersBlack:
              aiMove.prisoners_black !== undefined
                ? aiMove.prisoners_black
                : prev.prisonersBlack,
            prisonersWhite:
              aiMove.prisoners_white !== undefined
                ? aiMove.prisoners_white
                : prev.prisonersWhite,
            currentPlayer: userColorPass, // Sau AI move, đến lượt user
            lastMove:
              aiMove.x !== undefined
                ? { x: aiMove.x, y: aiMove.y }
                : prev.lastMove,
          };
        });

        // Xác định màu AI cho move history (after pass)
        const aiColorHistoryPass = currentMatch.black_player_id ? "W" : "B";

        // Update move history for AI move after pass
        setMoveHistory((prev) => {
          const moveNumber = prev.length + 1;

          // Kiểm tra AI pass trước - ưu tiên is_pass flag
          if (
            aiMove.is_pass === true ||
            (aiMove.x === null && aiMove.y === null)
          ) {
            // AI passed
            const newMove = {
              number: moveNumber,
              color: aiColorHistoryPass,
              position: null,
              captured: [],
            };
            console.log("📝 Adding AI pass to history (after pass):", newMove);
            return [...prev, newMove];
          }

          // Kiểm tra nhiều format: position array, x/y riêng, hoặc position object
          let position = null;
          if (aiMove.position !== null && aiMove.position !== undefined) {
            // position có thể là array [x, y] hoặc object {x, y}
            if (
              Array.isArray(aiMove.position) &&
              aiMove.position.length === 2
            ) {
              position = aiMove.position;
            } else if (
              aiMove.position.x !== undefined &&
              aiMove.position.y !== undefined
            ) {
              position = [aiMove.position.x, aiMove.position.y];
            }
          } else if (
            aiMove.x !== null &&
            aiMove.x !== undefined &&
            aiMove.y !== null &&
            aiMove.y !== undefined
          ) {
            // Fallback: x và y riêng biệt (phải khác null và undefined)
            position = [aiMove.x, aiMove.y];
          }

          if (position !== null) {
            // AI đã đánh một nước
            const newMove = {
              number: moveNumber,
              color: aiColorHistoryPass,
              position: position,
              captured: aiMove.captured || [],
            };
            console.log("📝 Adding AI move to history (after pass):", newMove);
            return [...prev, newMove];
          } else {
            // Fallback: Nếu không có position và không phải pass rõ ràng, vẫn ghi là pass
            const newMove = {
              number: moveNumber,
              color: aiColorHistoryPass,
              position: null,
              captured: [],
            };
            console.log(
              "📝 Adding AI pass to history (after pass, fallback):",
              newMove
            );
            return [...prev, newMove];
          }
        });

        // Kiểm tra game over sau AI move (có thể AI pass hoặc không còn đánh được)
        // QUAN TRỌNG: Kiểm tra game_over từ cả aiMove và passResponse
        const isGameOver = aiMove.game_over || passResponse.data?.game_over;
        if (isGameOver) {
          console.log("🏁 Game over detected after AI move:", {
            aiMoveGameOver: aiMove.game_over,
            passResponseGameOver: passResponse.data?.game_over,
          });
          // Reload match để lấy result đầy đủ
          try {
            const matchResponse = await api.get(`/matches/${currentMatch.id}`);
            if (matchResponse.data) {
              await setGameOverState(matchResponse.data);
            }
          } catch (err) {
            console.error("Failed to load game result:", err);
            // Fallback: set game over với data hiện có
            setGameOver(true);
          }
        }

        // Reload match state sau AI move để đảm bảo sync và kiểm tra game over
        console.log("🔄 Reloading match state after AI move (pass)...");
        await loadMatchState(currentMatch.id);

        // Kiểm tra lại game over sau khi reload (backend có thể đã cập nhật)
        const finalMatchResponse = await api.get(`/matches/${currentMatch.id}`);
        if (finalMatchResponse.data) {
          await setGameOverState(finalMatchResponse.data);
        }
      } else {
        // Không có AI move → có thể AI không còn đánh được hoặc game đã kết thúc
        // Kiểm tra lại game over từ response
        if (passResponse.data && passResponse.data.game_over) {
          // Reload match để lấy result đầy đủ
          try {
            const matchResponse = await api.get(`/matches/${currentMatch.id}`);
            if (matchResponse.data) {
              await setGameOverState(matchResponse.data);
            }
          } catch (err) {
            console.error("Failed to load game result:", err);
            // Fallback: set game over với data hiện có
            setGameOver(true);
          }
        }

        // Cập nhật currentPlayer từ response hoặc đảo ngược
        const newCurrentPlayer =
          passResponse.data?.current_player ||
          (boardState.currentPlayer === "B" ? "W" : "B");
        setBoardState((prev) => ({
          ...prev,
          currentPlayer: newCurrentPlayer,
        }));
        console.log("🔄 Updated currentPlayer after pass:", newCurrentPlayer);

        // Đợi một chút để đảm bảo state đã được cập nhật
        await new Promise((resolve) => setTimeout(resolve, 100));

        // Reload match state để kiểm tra game over và đồng bộ board state
        // QUAN TRỌNG: Sau khi reload, đảm bảo currentPlayer không bị ghi đè sai
        await loadMatchState(currentMatch.id);

        // Đảm bảo currentPlayer đúng sau reload (trong AI match, user luôn là Black)
        // Nếu backend trả về sai, force thành 'B' cho user
        if (currentMatch.ai_level) {
          setBoardState((prev) => {
            // Trong AI match, sau khi user pass, đến lượt AI (White)
            // Sau khi AI pass, đến lượt user (Black)
            // Nếu currentPlayer không phải 'B' sau khi user pass và AI pass, có thể bị sai
            const expectedPlayer =
              newCurrentPlayer === "W" ? "B" : newCurrentPlayer;
            if (prev.currentPlayer !== expectedPlayer) {
              console.log(
                `🔧 Fixing currentPlayer after reload: ${prev.currentPlayer} -> ${expectedPlayer}`
              );
              return { ...prev, currentPlayer: expectedPlayer };
            }
            return prev;
          });
        }
      }

      await loadInitialData();
    } catch (error) {
      console.error("❌ Pass error:", error);
      if (error.code === "ECONNABORTED") {
        alert("Pass timeout: AI is taking too long. Please wait or try again.");
      } else {
        alert(
          "Failed to pass: " + (error.response?.data?.detail || error.message)
        );
      }
    } finally {
      setIsProcessing(false);
    }
  };

  const formatGameResult = (result) => {
    if (!result) return "Game ended";

    // Format: "B+X", "W+X", "B+X(total)", "W+X(total)", "DRAW", "B+R", "W+R"
    if (result === "DRAW") {
      return "Kết quả: Hòa (Draw)";
    }

    if (result.endsWith("+R")) {
      const winner = result.startsWith("B") ? "Đen (Black)" : "Trắng (White)";
      return `Kết quả: ${winner} thắng do đối phương đầu hàng (Resign)`;
    }

    if (result.includes("+")) {
      const [winner, rest] = result.split("+");
      const winnerName = winner === "B" ? "Đen (Black)" : "Trắng (White)";

      // Kiểm tra format mới: "B+30.5(62)" hoặc format cũ: "B+30.5"
      const match = rest.match(/^([\d.]+)(?:\(([\d.]+)\))?$/);
      if (match) {
        const totalScore = match[2] || match[1]; // Nếu có total score trong ngoặc, dùng nó; nếu không, dùng difference (backward compatible)
        return `Kết quả: ${winnerName} thắng với ${totalScore} điểm`;
      }

      // Fallback cho format cũ
      return `Kết quả: ${winnerName} thắng với ${rest} điểm`;
    }

    return `Kết quả: ${result}`;
  };

  // Helper function để tính toán chi tiết điểm số
  const calculateScoreDetails = (stones, boardSize) => {
    // Đếm số quân trên bàn
    let stonesBlack = 0;
    let stonesWhite = 0;

    for (const key in stones) {
      if (stones[key] === "B") {
        stonesBlack++;
      } else if (stones[key] === "W") {
        stonesWhite++;
      }
    }

    // Tính territory bằng flood-fill
    const calculateTerritory = () => {
      let territoryBlack = 0;
      let territoryWhite = 0;
      const visited = new Set();

      const floodFillTerritory = (startX, startY) => {
        const region = [];
        const frontier = [[startX, startY]];
        const visitedRegion = new Set();

        // Bước 1: Flood-fill để thu thập tất cả các ô trống trong vùng
        while (frontier.length > 0) {
          const [x, y] = frontier.shift();
          const key = `${x},${y}`;

          if (visitedRegion.has(key)) continue;
          visitedRegion.add(key);
          region.push([x, y]);

          // Kiểm tra neighbors
          const neighbors = [
            [x + 1, y],
            [x - 1, y],
            [x, y + 1],
            [x, y - 1],
          ];

          for (const [nx, ny] of neighbors) {
            // Nếu ra ngoài bàn cờ, bỏ qua (không ảnh hưởng đến territory)
            if (nx < 0 || nx >= boardSize || ny < 0 || ny >= boardSize) {
              continue;
            }

            const neighborKey = `${nx},${ny}`;
            const neighborStone = stones[neighborKey];

            // Chỉ tiếp tục flood-fill nếu là ô trống
            if (!neighborStone && !visitedRegion.has(neighborKey)) {
              frontier.push([nx, ny]);
            }
          }
        }

        // Bước 2: Kiểm tra tất cả neighbors của toàn bộ vùng để xác định owner
        let owner = null;
        const neighborColors = new Set();

        for (const [x, y] of region) {
          const neighbors = [
            [x + 1, y],
            [x - 1, y],
            [x, y + 1],
            [x, y - 1],
          ];

          for (const [nx, ny] of neighbors) {
            // Bỏ qua nếu ra ngoài bàn cờ
            if (nx < 0 || nx >= boardSize || ny < 0 || ny >= boardSize) {
              continue;
            }

            const neighborKey = `${nx},${ny}`;
            const neighborStone = stones[neighborKey];

            if (neighborStone === "B") {
              neighborColors.add("B");
            } else if (neighborStone === "W") {
              neighborColors.add("W");
            }
          }
        }

        // Theo luật Trung Quốc: Territory = vùng trống được bao quanh hoàn toàn bởi một màu
        if (neighborColors.size === 1) {
          owner = neighborColors.has("B") ? "B" : "W";
        } else {
          // Có cả 2 màu hoặc không có màu nào -> không phải territory
          return { region: null, owner: null };
        }

        return { region: region.map(([x, y]) => `${x},${y}`), owner };
      };

      // Duyệt tất cả các ô trống
      for (let x = 0; x < boardSize; x++) {
        for (let y = 0; y < boardSize; y++) {
          const key = `${x},${y}`;
          if (stones[key] || visited.has(key)) continue;

          const { region, owner } = floodFillTerritory(x, y);
          if (region && owner) {
            region.forEach((k) => visited.add(k));
            if (owner === "B") {
              territoryBlack += region.length;
            } else {
              territoryWhite += region.length;
            }
          }
        }
      }

      return { territoryBlack, territoryWhite };
    };

    const { territoryBlack, territoryWhite } = calculateTerritory();
    const komi = 7.5; // Komi cho quân trắng

    return {
      stonesBlack,
      stonesWhite,
      territoryBlack,
      territoryWhite,
      komi,
    };
  };

  // Helper function để load ELO cuối trận đấu
  const loadFinalElo = async (matchData) => {
    if (
      !matchData ||
      matchData.user_elo_change === null ||
      matchData.user_elo_change === undefined
    ) {
      return;
    }

    try {
      // Lấy ELO hiện tại từ statistics
      const statsResponse = await api.get("/statistics/me");
      if (statsResponse.data && statsResponse.data.elo_rating !== undefined) {
        const currentElo = statsResponse.data.elo_rating;
        const eloBefore = currentElo - matchData.user_elo_change;
        setFinalElo({
          before: eloBefore,
          after: currentElo,
          change: matchData.user_elo_change,
        });
        setEloChange(matchData.user_elo_change);
      }
    } catch (statsErr) {
      console.error("Failed to load statistics for ELO:", statsErr);
    }
  };

  const handleCancelMatch = async () => {
    if (!currentMatch) return;

    // Chỉ cho phép hủy PvP matches chưa có người chơi thứ 2
    if (currentMatch.ai_level !== null && currentMatch.ai_level !== undefined) {
      alert("Không thể hủy trận đấu với AI");
      return;
    }

    if (currentMatch.white_player_id) {
      alert(
        "Không thể hủy trận đấu đã có đủ người chơi. Vui lòng sử dụng chức năng đầu hàng."
      );
      return;
    }

    if (
      !confirm(
        "Bạn có chắc chắn muốn hủy bàn này? Người chơi khác sẽ không thể tham gia nữa."
      )
    )
      return;

    try {
      await api.delete(`/matches/${currentMatch.id}`);
      // Quay về trang chủ sau khi hủy thành công
      if (onBackToHome) {
        onBackToHome();
      } else {
        // Fallback: reset state
        setCurrentMatch(null);
        setBoardState({
          stones: {},
          boardSize: 9,
          lastMove: null,
          prisonersBlack: 0,
          prisonersWhite: 0,
          currentPlayer: "B",
        });
        await loadInitialData();
      }
    } catch (error) {
      alert(
        "Không thể hủy bàn: " + (error.response?.data?.detail || error.message)
      );
    }
  };

  const handleResign = async () => {
    if (!currentMatch) return;
    if (!confirm("Bạn có chắc chắn muốn đầu hàng?")) return;

    try {
      await api.post(`/matches/${currentMatch.id}/resign`);
      // Reload match để lấy result đầy đủ
      try {
        const matchResponse = await api.get(`/matches/${currentMatch.id}`);
        if (matchResponse.data) {
          await setGameOverState(matchResponse.data);
        }
      } catch (err) {
        console.error("Failed to load game result:", err);
        // Fallback: set game over với data hiện có
        setGameOver(true);
      }
      setCurrentMatch(null);
      setBoardState({
        stones: {},
        boardSize: 9,
        lastMove: null,
        prisonersBlack: 0,
        prisonersWhite: 0,
        currentPlayer: "B",
      });
      await loadInitialData();
    } catch (error) {
      alert(
        "Không thể đầu hàng: " + (error.response?.data?.detail || error.message)
      );
    }
  };

  const handleUndo = async () => {
    if (!currentMatch || isProcessing || gameOver) return;

    // Xác định thông báo phù hợp
    const isAIMatch =
      currentMatch.ai_level !== null && currentMatch.ai_level !== undefined;
    const confirmMessage = isAIMatch
      ? "Bạn có chắc chắn muốn hoàn tác? (Sẽ hoàn tác cả nước của AI và nước của bạn)"
      : "Bạn có chắc chắn muốn hoàn tác nước đi cuối cùng?";

    if (!confirm(confirmMessage)) return;

    setIsProcessing(true);
    try {
      const response = await api.post(`/matches/${currentMatch.id}/undo`);
      const result = response.data;

      // Cập nhật board state từ response
      if (result.board_position) {
        setBoardState((prev) => ({
          ...prev,
          stones: { ...result.board_position },
          currentPlayer: result.current_player || prev.currentPlayer,
          prisonersBlack: result.prisoners_black || prev.prisonersBlack,
          prisonersWhite: result.prisoners_white || prev.prisonersWhite,
          lastMove: null, // Clear last move highlight
        }));
      }

      // Reload match state để đảm bảo đồng bộ
      await loadMatchState(currentMatch.id);

      // Hiển thị thông báo về số moves đã undo
      if (result.undone_moves && result.undone_moves.length > 1) {
        console.log(
          `✅ Undo thành công: Đã hoàn tác ${result.undone_moves.length} nước đi (AI + User)`
        );
      } else {
        console.log(
          "✅ Undo thành công:",
          result.undone_move || result.undone_moves?.[0]
        );
      }
    } catch (error) {
      const errorMsg =
        error.response?.data?.detail || error.message || "Không thể hoàn tác";
      alert("Không thể hoàn tác: " + errorMsg);
      console.error("Undo error:", error);
    } finally {
      setIsProcessing(false);
    }
  };

  // Tên người chơi (PvP) để dùng cho hiển thị và kiểm tra độ dài
  const blackPlayerName = currentMatch?.black_player_username || "";
  const isBlackNameLong = blackPlayerName.length > 12;

  return (
    <div className="main-window">
      <header className="main-header">
        <div className="header-title">
          <FaCircle className="header-icon" />
          <h1>Cờ Vây - 囲碁</h1>
        </div>
        <div className="header-actions">
          {onBackToHome && (
            <button
              onClick={onBackToHome}
              className="btn btn-secondary"
              title="Về trang chủ"
            >
              <span>Trang chủ</span>
            </button>
          )}
          <button
            onClick={() => setShowMatchDialog(true)}
            className="btn btn-primary"
            title="Tạo trận đấu mới"
          >
            <span>Trận mới</span>
          </button>
          <button
            onClick={() => setShowSettingsDialog(true)}
            className="btn btn-secondary"
            title="Cài đặt"
          >
            <span>Cài đặt</span>
          </button>
          <button
            onClick={() => {
              console.log("🚪 Logout clicked");
              if (onLogout) {
                onLogout();
              } else {
                console.error("onLogout is not defined");
              }
            }}
            className="btn btn-secondary"
            title="Đăng xuất"
          >
            <span>Đăng xuất</span>
          </button>
        </div>
      </header>

      <div className="main-content">
        {/* Left sidebar - Game Info and Controls */}
        <div className="left-sidebar">
          {/* Player info and game status */}
          <div className="players-display">
            <div className="player-info player-black">
              {/* Hiển thị "Đen" và tên người chơi/AI bên cạnh */}
              <div className="player-label">
                <span className="player-color-name">Đen</span>
                {currentMatch?.ai_level ? (
                  // AI match: hiển thị Bạn hoặc AI dựa trên playerColor
                  <span className="player-player-name">
                    {playerColor === "B" ? " - Bạn" : " - AI"}
                  </span>
                ) : (
                  // PvP match
                  <span className="player-player-name">
                    {playerColor === "B" ? (
                      " - Bạn"
                    ) : currentMatch?.black_player_username ? (
                      <>
                        <span className="player-player-name-prefix"> - </span>
                        <span
                          className={
                            "player-player-name-text" +
                            (isBlackNameLong
                              ? " player-player-name-text-small"
                              : "")
                          }
                        >
                          {blackPlayerName}
                        </span>
                      </>
                    ) : (
                      ""
                    )}
                  </span>
                )}
              </div>
              {/* Hiển thị thời gian còn lại cho PvP matches */}
              {!currentMatch?.ai_level &&
                boardState.blackTimeRemaining !== null && (
                  <div
                    className={`time-display ${
                      boardState.currentPlayer === "B" ? "time-active" : ""
                    } ${
                      boardState.blackTimeRemaining <= 30 ? "time-warning" : ""
                    }`}
                  >
                    ⏱️ {formatTime(boardState.blackTimeRemaining)}
                  </div>
                )}
            </div>
            <div className="game-status">
              {gameOver ? (
                <div className="status-message game-over">
                  <span className="game-over-icon">🏁</span>
                  <span className="game-over-text">
                    {gameResult ? formatGameResult(gameResult) : "Kết thúc"}
                  </span>
                </div>
              ) : isProcessing ? (
                <div className="status-message processing">
                  <span className="spinner">⏳</span>
                  <span>Đang chờ...</span>
                </div>
              ) : (
                <div className="status-message">
                  {boardState.currentPlayer === "B" ? "Lượt Đen" : "Lượt Trắng"}
                </div>
              )}
            </div>
            <div className="player-info player-white">
              {/* Hiển thị "Trắng" và tên người chơi/AI bên cạnh */}
              <div className="player-label">
                <span className="player-color-name">Trắng</span>
                {currentMatch?.ai_level ? (
                  // AI match: hiển thị Bạn hoặc AI dựa trên playerColor
                  <span className="player-player-name">
                    {playerColor === "W" ? " - Bạn" : " - AI"}
                  </span>
                ) : (
                  // PvP match
                  <span className="player-player-name">
                    {playerColor === "W"
                      ? " - Bạn"
                      : currentMatch?.white_player_username
                      ? ` - ${currentMatch.white_player_username}`
                      : ""}
                  </span>
                )}
              </div>
              {/* Hiển thị thời gian còn lại cho PvP matches */}
              {!currentMatch?.ai_level &&
                boardState.whiteTimeRemaining !== null && (
                  <div
                    className={`time-display ${
                      boardState.currentPlayer === "W" ? "time-active" : ""
                    } ${
                      boardState.whiteTimeRemaining <= 30 ? "time-warning" : ""
                    }`}
                  >
                    ⏱️ {formatTime(boardState.whiteTimeRemaining)}
                  </div>
                )}
            </div>
          </div>

          {/* Room Code Display (for PvP matches) */}
          {currentMatch && currentMatch.room_code && (
            <div className="room-code-display-in-game">
              <div className="room-code-label">Mã bàn:</div>
              <div className="room-code-box-in-game">
                <span className="room-code-text-in-game">
                  {currentMatch.room_code}
                </span>
                <button
                  type="button"
                  onClick={() => {
                    navigator.clipboard.writeText(currentMatch.room_code);
                    setRoomCodeCopied(true);
                    setTimeout(() => setRoomCodeCopied(false), 2000);
                  }}
                  className="copy-button-in-game"
                  title="Sao chép mã bàn"
                >
                  {roomCodeCopied ? <FaCheck /> : <FaCopy />}
                </button>
              </div>
            </div>
          )}

          {/* Nút Hủy bàn - chỉ hiển thị khi chưa có người chơi thứ 2 */}
          {currentMatch &&
            !currentMatch.ai_level &&
            !currentMatch.white_player_id && (
              <div className="cancel-match-section">
                <div className="waiting-message">
                  <span className="waiting-icon">⏳</span>
                  <span>Đang chờ người chơi tham gia...</span>
                </div>
                <button
                  type="button"
                  onClick={handleCancelMatch}
                  className="btn btn-danger cancel-match-btn"
                  title="Hủy bàn này"
                >
                  <span>Hủy bàn</span>
                </button>
              </div>
            )}

          {/* Game Controls */}
          <GameControls
            onPass={handlePass}
            onResign={handleResign}
            onUndo={handleUndo}
            onHint={() => alert("Tính năng gợi ý - sắp ra mắt")}
            onAnalysis={() => alert("Tính năng phân tích - sắp ra mắt")}
            onReview={() => alert("Tính năng xem lại - sắp ra mắt")}
            disabled={isProcessing || gameOver}
            undoDisabled={!currentMatch || moveHistory.length === 0}
          />
        </div>

        {/* Center - Board */}
        <div className="center-panel">
          <div className="board-wrapper">
            {currentMatch ? (
              <Board
                boardSize={boardState.boardSize}
                stones={boardState.stones}
                onCellClick={handleBoardClick}
                lastMove={boardState.lastMove}
                disabled={isProcessing || gameOver}
                theme={settings.boardTheme}
              />
            ) : (
              <div className="no-match-message">
                <div style={{ fontSize: "3rem", marginBottom: "1rem" }}>
                  ⚫⚪
                </div>
                <p
                  style={{
                    fontSize: "1.2rem",
                    fontWeight: "600",
                    marginBottom: "0.5rem",
                  }}
                >
                  Chưa có trận đấu nào
                </p>
                <p style={{ fontSize: "0.9rem", color: "#666" }}>
                  Nhấn &quot;Trận mới&quot; để bắt đầu chơi
                </p>
              </div>
            )}
          </div>
        </div>

        {/* Right sidebar - Statistics and Move History */}
        <div className="right-sidebar">
          <div className="right-sidebar-content">
            <div className="right-sidebar-section">
              <StatisticsPanel statistics={statistics} compact={true} />
            </div>
            {currentMatch && (
              <div className="right-sidebar-section">
                <MoveHistory moves={moveHistory} />
              </div>
            )}
          </div>
        </div>
      </div>

      {showMatchDialog && (
        <MatchDialog
          onClose={() => {
            console.log("🔴 MatchDialog onClose called");
            setShowMatchDialog(false);
          }}
          onCreateMatch={async (matchType, level, boardSize, playerColor) => {
            await handleCreateMatch(matchType, level, boardSize, playerColor);
            // Ensure dialog closes after match is created
            setShowMatchDialog(false);
          }}
        />
      )}

      {showSettingsDialog && (
        <SettingsDialog
          isOpen={showSettingsDialog}
          onClose={() => setShowSettingsDialog(false)}
          settings={settings}
          onSettingsChange={(newSettings) => {
            setSettings(newSettings);
            localStorage.setItem("goGameSettings", JSON.stringify(newSettings));
          }}
        />
      )}

      {showKoDialog && (
        <KoDialog
          isOpen={showKoDialog}
          onClose={() => setShowKoDialog(false)}
          koPosition={koPosition}
        />
      )}

      {/* Opponent Pass Dialog */}
      {showOpponentPassDialog && (
        <div
          className="pass-dialog-overlay"
          onClick={() => setShowOpponentPassDialog(false)}
        >
          <div className="pass-dialog" onClick={(e) => e.stopPropagation()}>
            <div className="pass-dialog-header">
              <h3>Thông báo</h3>
              <button
                className="pass-dialog-close"
                onClick={() => setShowOpponentPassDialog(false)}
                title="Đóng"
              >
                <FaTimes />
              </button>
            </div>
            <div className="pass-dialog-body">
              <p>{opponentPassMessage || "Đối thủ đã bỏ lượt."}</p>
            </div>
          </div>
        </div>
      )}

      {/* Game Over Modal */}
      {showGameOverModal && gameOverMessage && (
        <div
          className="game-over-modal-overlay"
          onClick={() => setShowGameOverModal(false)}
        >
          <div className="game-over-modal" onClick={(e) => e.stopPropagation()}>
            <div className="game-over-modal-header">
              <h2>🎮 Game Over!</h2>
              <button
                className="game-over-modal-close"
                onClick={() => setShowGameOverModal(false)}
                title="Đóng"
              >
                <FaTimes />
              </button>
            </div>
            <div className="game-over-modal-content">
              <p>{gameOverMessage}</p>

              {/* Hiển thị chi tiết điểm số */}
              {gameScoreDetails && gameResult && !gameResult.endsWith("+R") && (
                <div className="game-over-score-details">
                  {gameResult.startsWith("W") ? (
                    // Quân trắng thắng
                    <div className="score-details-winner">
                      <div className="score-details-title">
                        🏆 Chi tiết điểm số - Quân Trắng thắng
                      </div>
                      <div className="score-details-content">
                        <div className="score-detail-item">
                          <span className="score-label">
                            Tổng số quân trên bàn cờ:
                          </span>
                          <span className="score-value">
                            {gameScoreDetails.stonesWhite}
                          </span>
                        </div>
                        <div className="score-detail-item">
                          <span className="score-label">
                            Số lãnh thổ đã chiếm:
                          </span>
                          <span className="score-value">
                            {gameScoreDetails.territoryWhite}
                          </span>
                        </div>
                        <div className="score-detail-item">
                          <span className="score-label">Điểm cộng Komi:</span>
                          <span className="score-value komi">
                            +{gameScoreDetails.komi}
                          </span>
                        </div>
                        <div className="score-detail-total">
                          <span className="score-label">Tổng điểm:</span>
                          <span className="score-value total">
                            {gameScoreDetails.stonesWhite +
                              gameScoreDetails.territoryWhite +
                              gameScoreDetails.komi}
                          </span>
                        </div>
                      </div>
                    </div>
                  ) : gameResult.startsWith("B") ? (
                    // Quân đen thắng
                    <div className="score-details-winner">
                      <div className="score-details-title">
                        🏆 Chi tiết điểm số - Quân Đen thắng
                      </div>
                      <div className="score-details-content">
                        <div className="score-detail-item">
                          <span className="score-label">
                            Tổng số quân trên bàn cờ:
                          </span>
                          <span className="score-value">
                            {gameScoreDetails.stonesBlack}
                          </span>
                        </div>
                        <div className="score-detail-item">
                          <span className="score-label">
                            Số lãnh thổ đã chiếm:
                          </span>
                          <span className="score-value">
                            {gameScoreDetails.territoryBlack}
                          </span>
                        </div>
                        <div className="score-detail-total">
                          <span className="score-label">Tổng điểm:</span>
                          <span className="score-value total">
                            {gameScoreDetails.stonesBlack +
                              gameScoreDetails.territoryBlack}
                          </span>
                        </div>
                      </div>
                    </div>
                  ) : null}
                </div>
              )}

              {/* Hiển thị ELO cuối trận đấu (chỉ cho PvP matches) */}
              {!currentMatch?.ai_level && finalElo && (
                <div className="game-over-elo-info">
                  <div className="elo-info-title">📊 ELO Rating</div>
                  <div className="elo-info-content">
                    <div className="elo-before">
                      <span className="elo-label">Trước trận:</span>
                      <span className="elo-value">{finalElo.before}</span>
                    </div>
                    <div className="elo-arrow">
                      {eloChange > 0 ? "↑" : eloChange < 0 ? "↓" : "→"}
                    </div>
                    <div className="elo-after">
                      <span className="elo-label">Sau trận:</span>
                      <span
                        className={`elo-value ${
                          eloChange > 0
                            ? "elo-positive"
                            : eloChange < 0
                            ? "elo-negative"
                            : ""
                        }`}
                      >
                        {finalElo.after}
                      </span>
                    </div>
                    <div className="elo-change">
                      <span
                        className={`elo-change-value ${
                          eloChange > 0
                            ? "elo-positive"
                            : eloChange < 0
                            ? "elo-negative"
                            : ""
                        }`}
                      >
                        {eloChange > 0 ? `+${eloChange}` : eloChange}
                      </span>
                    </div>
                  </div>
                </div>
              )}
            </div>
            <div className="game-over-modal-footer">
              <button
                className="game-over-modal-btn"
                onClick={() => setShowGameOverModal(false)}
              >
                Đóng
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Modal thông báo màu quân cờ cho PvP matches */}
      {showPlayerColorModal && playerColor && (
        <div
          className="player-color-modal-overlay"
          onClick={() => setShowPlayerColorModal(false)}
        >
          <div
            className="player-color-modal"
            onClick={(e) => e.stopPropagation()}
          >
            <div className="player-color-modal-header">
              <h2>Bạn chơi quân {playerColor === "B" ? "Đen" : "Trắng"}</h2>
              <button
                type="button"
                onClick={() => setShowPlayerColorModal(false)}
                className="player-color-modal-close"
                title="Đóng"
              >
                <FaTimes />
              </button>
            </div>
            <div className="player-color-modal-content">
              <div
                className={`player-color-badge ${
                  playerColor === "B" ? "badge-black" : "badge-white"
                }`}
              >
                {playerColor === "B" ? "⚫" : "⚪"}
                <span className="player-color-text">
                  {playerColor === "B" ? "Quân Đen" : "Quân Trắng"}
                </span>
              </div>
              <p className="player-color-info">
                {playerColor === "B"
                  ? "Bạn là người chơi Đen và sẽ đi trước."
                  : "Bạn là người chơi Trắng và sẽ đi sau."}
              </p>
            </div>
            <div className="player-color-modal-footer">
              <button
                className="player-color-modal-btn"
                onClick={() => setShowPlayerColorModal(false)}
              >
                Bắt đầu
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

MainWindow.propTypes = {
  onLogout: PropTypes.func,
  onBackToHome: PropTypes.func,
  initialMatch: PropTypes.object,
};

export default MainWindow;
