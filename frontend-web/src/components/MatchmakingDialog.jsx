import { useEffect, useRef, useState } from "react";
import PropTypes from "prop-types";
import api from "../services/api";
import "./MatchmakingDialog.css";

/**
 * MatchmakingDialog
 *
 * Props:
 *  - onClose(): đóng dialog
 *  - onMatchFound(match): gọi khi backend tạo được match
 *  - initialBoardSize?: số (9 | 13 | 19) – nếu truyền vào sẽ set mặc định
 *  - autoStart?: boolean – nếu true sẽ tự động join queue khi mở dialog
 *
 * Flow:
 *  - Chọn kích thước bàn cờ
 *  - Click "Tìm đối thủ" → POST /matchmaking/queue/join
 *  - Khi inQueue=true → polling mỗi 1 giây:
 *      + GET /matchmaking/queue/status (hiển thị thông tin hàng chờ)
 *      + GET /matchmaking/queue/match  (khi matched → onMatchFound)
 *  - Đóng dialog / Hủy tìm kiếm → POST /matchmaking/queue/leave
 */
const MatchmakingDialog = ({
  onClose,
  onMatchFound,
  initialBoardSize = 19,
  autoStart = false,
}) => {
  const [boardSize, setBoardSize] = useState(initialBoardSize || 19);
  const [inQueue, setInQueue] = useState(false);
  const [status, setStatus] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const pollingRef = useRef(null);

  const clearPolling = () => {
    if (pollingRef.current) {
      clearInterval(pollingRef.current);
      pollingRef.current = null;
    }
  };

  const leaveQueue = async () => {
    clearPolling();
    if (!inQueue) return;
    try {
      await api.post("/matchmaking/queue/leave");
    } catch (e) {
      // Không cần alert, chỉ log để debug
      // eslint-disable-next-line no-console
      console.error("Failed to leave matchmaking queue:", e);
    } finally {
      setInQueue(false);
    }
  };

  const handleClose = async () => {
    await leaveQueue();
    onClose?.();
  };

  const startPolling = () => {
    clearPolling();
    pollingRef.current = setInterval(async () => {
      try {
        // Cập nhật trạng thái queue
        const statusRes = await api.get("/matchmaking/queue/status");
        setStatus(statusRes.data);

        if (!statusRes.data?.in_queue) {
          // Nếu backend báo không còn trong queue thì dừng polling
          clearPolling();
          setInQueue(false);
          return;
        }

        // Kiểm tra xem đã có match chưa
        const matchRes = await api.get("/matchmaking/queue/match");
        if (matchRes.data && matchRes.data.matched && matchRes.data.match) {
          clearPolling();
          setInQueue(false);
          const matchData = matchRes.data.match;
          if (matchData) {
            onMatchFound?.(matchData);
          }
        }
      } catch (e) {
        // Lỗi mạng tạm thời: chỉ log, không thoát hàng chờ
        // eslint-disable-next-line no-console
        console.error("Error while polling matchmaking status:", e);
      }
    }, 1000);
  };

  const handleJoinQueue = async () => {
    setError("");
    setLoading(true);
    try {
      const res = await api.post("/matchmaking/queue/join", {
        board_size: boardSize,
      });
      setStatus(res.data);
      setInQueue(true);
      startPolling();
    } catch (e) {
      const statusCode = e.response?.status;
      // Nếu backend báo "đã có trong queue rồi" (409) thì chỉ cần coi như đang trong hàng chờ,
      // không hiển thị dòng lỗi khó chịu cho người dùng.
      if (statusCode === 409) {
        setInQueue(true);
        startPolling();
        setError("");
      } else {
        const message =
          e.response?.data?.detail || e.message || "Lỗi không xác định";
        setError(message);
      }
    } finally {
      setLoading(false);
    }
  };

  // Cập nhật board size khi prop initialBoardSize thay đổi
  useEffect(() => {
    if (initialBoardSize) {
      setBoardSize(initialBoardSize);
    }
  }, [initialBoardSize]);

  // Tự động join queue khi mở dialog (khi người dùng đã chọn "Ghép online")
  useEffect(() => {
    if (autoStart && !inQueue && !loading) {
      // Không cần await, chỉ fire-and-forget
      void handleJoinQueue();
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [autoStart, boardSize]);

  useEffect(() => {
    return () => {
      // Cleanup khi component unmount
      clearPolling();
      // Không await để tránh warning trong effect cleanup
      void leaveQueue();
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const handleOverlayClick = (e) => {
    if (e.target === e.currentTarget) {
      handleClose();
    }
  };

  return (
    <div className="matchmaking-dialog-overlay" onClick={handleOverlayClick}>
      <div className="matchmaking-dialog" onClick={(e) => e.stopPropagation()}>
        <div className="mm-header">
          <h2>Ghép người chơi online</h2>
          <button type="button" className="mm-close" onClick={handleClose}>
            ×
          </button>
        </div>

        <div className="mm-body">
          {!inQueue && (
            <div className="mm-section">
              <div className="mm-label">Chọn kích thước bàn cờ</div>
              <div className="mm-size-selector">
                {[9, 13, 19].map((size) => (
                  <button
                    key={size}
                    type="button"
                    className={
                      "mm-size-btn " + (boardSize === size ? "mm-size-btn-active" : "")
                    }
                    onClick={() => setBoardSize(size)}
                  >
                    {size}x{size}
                  </button>
                ))}
              </div>
            </div>
          )}

          <div className="mm-section">
            <div className="mm-label">
              {inQueue ? "Trạng thái hàng chờ" : "Thông tin"}
            </div>
            <div className="mm-status">
              {inQueue ? (
                <>
                  <div className="mm-loading-container">
                    <div className="mm-loading-spinner">
                      <div className="mm-spinner-ring"></div>
                      <div className="mm-spinner-ring"></div>
                      <div className="mm-spinner-ring"></div>
                    </div>
                    <div className="mm-loading-particles">
                      {[...Array(12)].map((_, i) => (
                        <div key={i} className="mm-particle" style={{
                          '--delay': `${i * 0.1}s`,
                          '--angle': `${i * 30}deg`
                        }}></div>
                      ))}
                    </div>
                  </div>
                  <div className="mm-status-header">
                    <div className="mm-status-icon">
                      <div className="mm-pulse-ring"></div>
                      <div className="mm-pulse-ring"></div>
                      <span className="mm-pulse-dot" />
                    </div>
                    <div className="mm-status-text">
                      <div className="mm-status-title">
                        <span className="mm-shimmer-text">Đang tìm đối thủ phù hợp</span>
                        <span className="mm-typing-dots">
                          <span>.</span>
                          <span>.</span>
                          <span>.</span>
                        </span>
                      </div>
                      <div className="mm-status-subtitle">
                        Bàn {boardSize}x{boardSize} · Hãy giữ ứng dụng mở để
                        không bỏ lỡ trận đấu.
                      </div>
                    </div>
                  </div>
                  <div className="mm-progress-bar">
                    <div className="mm-progress-fill"></div>
                  </div>

                  {status && (
                    <div className="mm-status-meta">
                      {status.queue_size != null && (
                        <span>
                          Người chơi trong hàng chờ: {status.queue_size}
                        </span>
                      )}
                      {status.wait_time != null && (
                        <span>Thời gian chờ: {status.wait_time}s</span>
                      )}
                    </div>
                  )}
                </>
              ) : (
                <div className="mm-status-empty">
                  <div>Chọn kích thước bàn cờ và nhấn "Tìm đối thủ" để bắt đầu</div>
                  <div style={{ fontSize: '12px', opacity: 0.6, marginTop: '8px' }}>
                    Hệ thống sẽ tự động ghép bạn với đối thủ phù hợp
                  </div>
                </div>
              )}
            </div>
          </div>

          {error && <div className="mm-error">Lỗi: {error}</div>}
        </div>

        <div className="mm-footer">
          {inQueue ? (
            <button
              type="button"
              className="mm-secondary-btn"
              onClick={handleClose}
              disabled={loading}
            >
              Hủy tìm kiếm
            </button>
          ) : (
            <button
              type="button"
              className="mm-primary-btn"
              onClick={handleJoinQueue}
              disabled={loading}
            >
              {loading ? "Đang tham gia..." : "Tìm đối thủ"}
            </button>
          )}
        </div>
      </div>
    </div>
  );
};

MatchmakingDialog.propTypes = {
  onClose: PropTypes.func,
  onMatchFound: PropTypes.func,
  initialBoardSize: PropTypes.number,
  autoStart: PropTypes.bool,
};

export default MatchmakingDialog;
