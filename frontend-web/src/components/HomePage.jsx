import { useState, useEffect } from 'react'
import PropTypes from 'prop-types'
import { useAuth } from '../contexts/AuthContext'
import { FaCog, FaSignOutAlt, FaGamepad, FaUser, FaHistory, FaInfoCircle, FaTimes, FaTrophy, FaGift } from 'react-icons/fa'
import api from '../services/api'
import MatchDialog from './MatchDialog'
import CoinDisplay from './CoinDisplay'
import PremiumBadge from './PremiumBadge'
import ShopDialog from './ShopDialog'
import PremiumDialog from './PremiumDialog'
import TransactionHistory from './TransactionHistory'
import EventDialog from './EventDialog'

// Force reload v3
console.log("🏠 HomePage.jsx loaded - version 3");
import MatchmakingDialog from "./MatchmakingDialog";
import MatchFoundDialog from "./MatchFoundDialog";
import MatchList from "./MatchList";
import StatisticsPanel from "./StatisticsPanel";
import SettingsDialog from "./SettingsDialog";
import Leaderboard from "./Leaderboard";
import LeaderboardPreview from "./LeaderboardPreview";
import GoTutorial from "./GoTutorial";
import InteractiveTutorial from "./InteractiveTutorial";
import "./HomePage.css";

const HomePage = ({ onStartMatch }) => {
  const { user, logout } = useAuth()
  const [statistics, setStatistics] = useState(null)
  const [recentMatches, setRecentMatches] = useState([])
  const [allMatches, setAllMatches] = useState([])
  const [showMatchDialog, setShowMatchDialog] = useState(false)
  const [showMatchmakingDialog, setShowMatchmakingDialog] = useState(false)
  const [matchmakingBoardSize, setMatchmakingBoardSize] = useState(19)
  const [showMatchFoundDialog, setShowMatchFoundDialog] = useState(false)
  const [foundMatch, setFoundMatch] = useState(null)
  const [showSettingsDialog, setShowSettingsDialog] = useState(false)
  const [showInfoPanel, setShowInfoPanel] = useState(false)
  const [showHistoryDialog, setShowHistoryDialog] = useState(false)
  const [showLeaderboard, setShowLeaderboard] = useState(false)
  const [showInteractiveTutorial, setShowInteractiveTutorial] = useState(false)
  const [hasCheckedTutorial, setHasCheckedTutorial] = useState(false)
  const [topPlayers, setTopPlayers] = useState([])
  const [loading, setLoading] = useState(true)
  const [showShopDialog, setShowShopDialog] = useState(false)
  const [showPremiumDialog, setShowPremiumDialog] = useState(false)
  const [showTransactionHistory, setShowTransactionHistory] = useState(false)
  const [showEventDialog, setShowEventDialog] = useState(false)
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

  useEffect(() => {
    let isMounted = true;

    const fetchData = async () => {
      if (isMounted) {
        await loadData();
      }
    };

    fetchData();

    return () => {
      isMounted = false;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const loadData = async () => {
    try {
      setLoading(true);
      const [matchesRes, statsRes, leaderboardRes] = await Promise.all([
        api.get("/matches/history?limit=1"), // Chỉ cần 1 để kiểm tra
        api.get("/statistics/me"),
        api.get("/statistics/leaderboard?limit=5"),
      ]);

      const matches = matchesRes.data || [];
      setRecentMatches(matches);
      setStatistics(statsRes.data);
      setTopPlayers(leaderboardRes.data || []);

      // Xác định key lưu trạng thái đã hoàn thành tutorial theo từng user
      const userIdFromStats = statsRes.data?.user_id;
      const userIdFromAuth = user?.id || user?.user_id;
      const tutorialUserId = userIdFromStats || userIdFromAuth || "guest";
      const tutorialKey = `interactiveTutorialCompleted_${tutorialUserId}`;
      const hasCompletedTutorial =
        typeof window !== "undefined" &&
        window.localStorage &&
        window.localStorage.getItem(tutorialKey) === "true";

      // Kiểm tra xem user có matches nào chưa
      // Chỉ hiện tutorial nếu:
      // - Chưa có trận đấu nào
      // - Và chưa hoàn thành hết các bài hướng dẫn tương tác
      if (
        !hasCheckedTutorial &&
        matches.length === 0 &&
        !hasCompletedTutorial
      ) {
        setShowInteractiveTutorial(true);
      }
      setHasCheckedTutorial(true);

      // Kiểm tra xem đã hiển thị event dialog chưa (chỉ hiển thị lần đầu sau khi đăng nhập trong session này)
      // Sử dụng sessionStorage để chỉ hiển thị khi thực sự là lần đầu đăng nhập, không phải refresh
      const eventDialogUserId = userIdFromStats || userIdFromAuth || "guest";
      const eventDialogSessionKey = `eventDialogShown_${eventDialogUserId}_session`;
      const hasShownEventDialogInSession =
        typeof window !== "undefined" &&
        window.sessionStorage &&
        window.sessionStorage.getItem(eventDialogSessionKey) === "true";
      
      // Chỉ hiển thị event dialog nếu chưa hiển thị trong session này và user đã đăng nhập
      if (!hasShownEventDialogInSession && eventDialogUserId !== "guest") {
        // Đánh dấu đã hiển thị trong session này
        try {
          if (typeof window !== "undefined" && window.sessionStorage) {
            window.sessionStorage.setItem(eventDialogSessionKey, "true");
          }
        } catch (e) {
          console.error("Failed to set event dialog session state:", e);
        }
        // Delay nhỏ để đảm bảo UI đã load xong
        setTimeout(() => {
          setShowEventDialog(true);
        }, 500);
      }

      // Load thêm matches để hiển thị trong recent matches
      if (matches.length > 0) {
        const allMatchesRes = await api.get("/matches/history?limit=3");
        setRecentMatches(allMatchesRes.data || []);
      }
    } catch (error) {
      console.error("Failed to load home data:", error);
      // Nếu lỗi, không hiển thị tutorial
      setHasCheckedTutorial(true);
    } finally {
      setLoading(false);
    }
  };

  const loadAllMatches = async () => {
    try {
      const response = await api.get("/matches/history");
      setAllMatches(response.data || []);
    } catch (error) {
      console.error("Failed to load all matches:", error);
      setAllMatches([]);
    }
  };

  const handleCreateMatch = async (
    matchType,
    option,
    boardSize,
    playerColor = "black"
  ) => {
    try {
      if (matchType === "matchmaking") {
        // Open matchmaking dialog và tự join queue với board size đã chọn
        if (boardSize) {
          setMatchmakingBoardSize(boardSize);
        }
        setShowMatchDialog(false);
        setShowMatchmakingDialog(true);
        return;
      }

      if (matchType === "pvp") {
        // Tạo trận PVP trực tiếp (tạo bàn mới) với thời gian đã chọn
        const timeControlMinutes = option || 10;
        try {
          const response = await api.post("/matches/pvp", {
            board_size: boardSize,
            time_control_minutes: timeControlMinutes,
          });
          const { match, join_code } = response.data;

          // Log mã bàn để chia sẻ (không hiện popup browser)
          if (join_code) {
            console.log(
              "Mã bàn của bạn (gửi cho đối thủ để họ tham gia):",
              join_code
            );
          }

          setShowMatchDialog(false);
          if (onStartMatch) {
            onStartMatch(match);
          }
        } catch (error) {
          alert(
            "Không thể tạo trận PVP: " +
              (error.response?.data?.detail || error.message)
          );
        }
        return;
      }

      if (matchType === "pvp-join") {
        // Tham gia bàn bằng mã
        const roomCode = playerColor; // playerColor được dùng để truyền roomCode
        if (!roomCode || roomCode.length !== 6) {
          alert("Mã bàn phải có 6 ký tự");
          return;
        }
        try {
          const response = await api.post("/matches/pvp/join", {
            room_code: roomCode.toUpperCase(),
          });
          setShowMatchDialog(false);
          if (onStartMatch) {
            onStartMatch(response.data);
          }
        } catch (error) {
          alert(
            "Không thể tham gia bàn: " +
              (error.response?.data?.detail || error.message)
          );
        }
        return;
      }

      // AI match - gửi player_color để backend biết người chơi muốn cầm quân gì
      console.log(
        "🎮 HomePage: Creating AI match with player_color:",
        playerColor
      );
      const response = await api.post("/matches/ai", {
        level: option,
        board_size: boardSize,
        player_color: playerColor,
      });
      const match = response.data;
      setShowMatchDialog(false);
      if (onStartMatch) {
        onStartMatch(match);
      }
    } catch (error) {
      alert(
        "Không thể tạo trận đấu: " +
          (error.response?.data?.detail || error.message)
      );
    }
  };

  const handleMatchFound = (match) => {
    console.log("🎮 [HomePage] handleMatchFound called with match:", match);
    setShowMatchmakingDialog(false);
    if (match && match.id) {
      setFoundMatch(match);
      setShowMatchFoundDialog(true);
      console.log("✅ [HomePage] MatchFoundDialog should be displayed now");
    } else {
      console.error("❌ [HomePage] Invalid match data:", match);
    }
  };

  const handleMatchStart = (match) => {
    console.log("🚀 [HomePage] handleMatchStart called with match:", match);
    setShowMatchFoundDialog(false);
    setFoundMatch(null);
    if (onStartMatch && match && match.id) {
      console.log("✅ [HomePage] Starting match:", match.id);
      onStartMatch(match);
    } else {
      console.error(
        "❌ [HomePage] Cannot start match - invalid match or onStartMatch:",
        {
          match,
          hasOnStartMatch: !!onStartMatch,
        }
      );
    }
  };

  const handleMatchFoundCancel = () => {
    setShowMatchFoundDialog(false);
    setFoundMatch(null);
    // Có thể quay lại matchmaking dialog hoặc home
  };

  const handleContinueMatch = (match) => {
    if (onStartMatch) {
      onStartMatch(match);
    }
  };

  const handleSettingsChange = (newSettings) => {
    setSettings(newSettings);
    localStorage.setItem("goGameSettings", JSON.stringify(newSettings));
  };

  const handleLogout = () => {
    if (window.confirm("Bạn có chắc muốn đăng xuất?")) {
      // Reset matchmaking related states before logging out to avoid leaking old match info
      setShowMatchFoundDialog(false);
      setFoundMatch(null);
      setShowMatchmakingDialog(false);
      logout();
    }
  };

  if (loading) {
    return (
      <div className="home-page loading">
        <div className="loading-spinner">Đang tải...</div>
      </div>
    );
  }

  return (
    <div className="home-page">
      {/* Top Left - User Info */}
      <div className="corner-panel top-left">
        <div className="user-display">
          <FaUser className="user-icon" />
          <div className="user-details">
            <span className="user-label">Người chơi</span>
            <span className="username">
              {statistics?.username || user?.username || "Khách"}
            </span>
          </div>
        </div>
      </div>

      {/* Top Right - Coins, Premium, Settings & Logout */}
      <div className="corner-panel top-right">
        <div className="top-right-content">
          <CoinDisplay 
            onShopClick={() => setShowShopDialog(true)}
            showShopButton={true}
          />
          <PremiumBadge 
            onPremiumClick={() => setShowPremiumDialog(true)}
            showButton={true}
          />
          <div className="top-right-divider"></div>
          <button 
            className="corner-btn"
            onClick={() => setShowTransactionHistory(true)}
            title="Lịch sử giao dịch"
          >
            <FaHistory />
          </button>
          <button 
            className="corner-btn event-btn"
            onClick={() => setShowEventDialog(true)}
            title="Xem sự kiện"
          >
            <FaGift />
          </button>
          <button 
            className="corner-btn"
            onClick={() => setShowSettingsDialog(true)}
            title="Cài đặt"
          >
            <FaCog />
          </button>
          <button 
            className="corner-btn"
            onClick={handleLogout}
            title="Đăng xuất"
          >
            <FaSignOutAlt />
          </button>
        </div>
      </div>

      {/* Middle Left - Leaderboard Preview (below user info) */}
      <div className="corner-panel middle-left">
        <LeaderboardPreview
          topPlayers={topPlayers}
          onViewAll={() => setShowLeaderboard(true)}
          loading={loading}
        />
      </div>

      {/* Bottom Left - Statistics */}
      <div className="corner-panel bottom-left">
        <StatisticsPanel statistics={statistics} compact={true} />
      </div>

      {/* Bottom Right - Recent Matches */}
      <div className="corner-panel bottom-right">
        <div className="compact-section">
          <div className="compact-header">
            <FaHistory className="compact-icon" />
            <span className="compact-title">Gần đây</span>
            <button
              className="view-all-btn"
              onClick={() => {
                loadAllMatches();
                setShowHistoryDialog(true);
              }}
              title="Xem tất cả"
            >
              Xem tất cả
            </button>
          </div>
          <div className="compact-content">
            {recentMatches.length > 0 ? (
              <MatchList
                matches={recentMatches.slice(0, 3)}
                onMatchClick={handleContinueMatch}
                compact={true}
              />
            ) : (
              <div className="empty-state-small">
                <p>Chưa có trận đấu</p>
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Info Icon - Bottom Center */}
      <div className="corner-panel bottom-center">
        <button
          className={`info-icon-btn ${showInfoPanel ? "active" : ""}`}
          onClick={(e) => {
            e.preventDefault();
            e.stopPropagation();
            console.log("Info button clicked, current state:", showInfoPanel);
            setShowInfoPanel((prev) => {
              const newState = !prev;
              console.log("Setting showInfoPanel to:", newState);
              return newState;
            });
          }}
          onMouseDown={(e) => {
            e.stopPropagation();
          }}
          title="Thông tin về cờ vây"
          type="button"
        >
          <FaInfoCircle className="info-icon" />
        </button>
      </div>

      {/* Go Tutorial */}
      <GoTutorial
        isOpen={showInfoPanel}
        onClose={() => setShowInfoPanel(false)}
      />

      {/* Interactive Tutorial - Tự động hiển thị cho người mới */}
      <InteractiveTutorial
        isOpen={showInteractiveTutorial}
        onClose={() => setShowInteractiveTutorial(false)}
        onCompleted={() => {
          const userIdFromStats = statistics?.user_id;
          const userIdFromAuth = user?.id || user?.user_id;
          const tutorialUserId = userIdFromStats || userIdFromAuth || "guest";
          const tutorialKey = `interactiveTutorialCompleted_${tutorialUserId}`;
          try {
            if (typeof window !== "undefined" && window.localStorage) {
              window.localStorage.setItem(tutorialKey, "true");
            }
          } catch (e) {
            // Nếu localStorage thất bại (ví dụ chế độ private), chỉ log và tiếp tục
            // eslint-disable-next-line no-console
            console.error("Failed to persist tutorial completion:", e);
          }
          setShowInteractiveTutorial(false);
        }}
      />

      {/* Center - Main Action Button */}
      <div className="center-section">
        <div className="center-content">
          <h1 className="main-title">Cờ Vây</h1>
          <button
            className="btn-main-action"
            onClick={() => setShowMatchDialog(true)}
          >
            <FaGamepad className="main-icon" />
            <span>Bắt đầu chơi</span>
          </button>
        </div>
      </div>

      {/* Dialogs */}
      {showMatchDialog && (
        <MatchDialog
          onClose={() => setShowMatchDialog(false)}
          onCreateMatch={handleCreateMatch}
        />
      )}

      {showMatchmakingDialog && (
        <MatchmakingDialog
          onClose={() => setShowMatchmakingDialog(false)}
          onMatchFound={handleMatchFound}
          initialBoardSize={matchmakingBoardSize}
          autoStart
        />
      )}

      {showMatchFoundDialog && foundMatch && (
        <MatchFoundDialog
          match={foundMatch}
          onStart={handleMatchStart}
          onCancel={handleMatchFoundCancel}
        />
      )}
      {showSettingsDialog && (
        <SettingsDialog
          isOpen={showSettingsDialog}
          onClose={() => setShowSettingsDialog(false)}
          settings={settings}
          onSettingsChange={handleSettingsChange}
        />
      )}

      {/* History Dialog */}
      {showHistoryDialog && (
        <div
          className="history-dialog-overlay"
          onClick={() => setShowHistoryDialog(false)}
        >
          <div className="history-dialog" onClick={(e) => e.stopPropagation()}>
            <div className="history-dialog-header">
              <h2>Lịch sử trận đấu</h2>
              <button
                className="history-close-btn"
                onClick={() => setShowHistoryDialog(false)}
              >
                <FaTimes />
              </button>
            </div>
            <div className="history-dialog-content">
              <MatchList
                matches={allMatches}
                onMatchClick={(match) => {
                  setShowHistoryDialog(false);
                  handleContinueMatch(match);
                }}
                compact={false}
              />
            </div>
          </div>
        </div>
      )}

      {/* Leaderboard Dialog */}
      <Leaderboard
        isOpen={showLeaderboard}
        onClose={() => setShowLeaderboard(false)}
      />

      {/* Shop Dialog */}
      <ShopDialog
        isOpen={showShopDialog}
        onClose={() => setShowShopDialog(false)}
        onPurchaseSuccess={() => {
          // Refresh balance sẽ tự động trong CoinDisplay
        }}
      />

      {/* Premium Dialog */}
      <PremiumDialog
        isOpen={showPremiumDialog}
        onClose={() => setShowPremiumDialog(false)}
        onSubscribeSuccess={() => {
          // Refresh sẽ tự động trong PremiumBadge
        }}
      />

      {/* Transaction History Dialog */}
      <TransactionHistory
        isOpen={showTransactionHistory}
        onClose={() => setShowTransactionHistory(false)}
      />

      {/* Event Dialog - Hiển thị lần đầu sau khi đăng nhập */}
      <EventDialog
        isOpen={showEventDialog}
        onClose={() => {
          // Không cần lưu vào localStorage nữa vì đã dùng sessionStorage
          // Chỉ đóng dialog
          setShowEventDialog(false);
        }}
        onBannerClick={() => {
          // Mở ShopDialog khi click vào banner
          setShowShopDialog(true);
        }}
      />
    </div>
  );
};

HomePage.propTypes = {
  onStartMatch: PropTypes.func,
};

export default HomePage;
