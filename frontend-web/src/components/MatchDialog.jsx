import { useState, useEffect } from "react";
import PropTypes from "prop-types";
import { FaUsers, FaRobot } from "react-icons/fa";
import StoneColorDialog from "./StoneColorDialog";
import "./MatchDialog.css";

/**
 * Thiết kế mới cho MatchDialog
 *
 * Props:
 *  - onClose(): đóng dialog
 *  - onCreateMatch(matchType, level, boardSize, playerColor?)
 *
 * Mapping:
 *  - PVE (Đấu với máy):  matchType = 'ai'
 *  - PVP + Mã tham gia:  matchType = 'pvp'
 *  - PVP + Ghép online:  matchType = 'matchmaking'
 */
const MatchDialog = ({ onClose, onCreateMatch }) => {
  const [expandedMode, setExpandedMode] = useState(null); // 'pvp' | 'pve' | null
  const [pvpSettings, setPvpSettings] = useState({
    matchType: null, // 'code' | 'online' | 'create'
    boardSize: null, // 9 | 13 | 19
    roomCode: '', // Mã bàn để tham gia (khi matchType === 'code')
    // Thời gian sẽ tự động gắn theo kích thước bàn cờ:
    // 9x9  -> 10 phút
    // 13x13 -> 20 phút
    // 19x19 -> 30 phút
    timeControlMinutes: 10,
  });
  const [pveSettings, setPveSettings] = useState({
    aiLevel: null, // 1–4
    boardSize: null, // 9 | 13 | 19
  });
  const [showStoneColorDialog, setShowStoneColorDialog] = useState(false);

  // Esc đóng dialog (hoặc đóng StoneColorDialog nếu đang mở)
  useEffect(() => {
    const handleEscape = (e) => {
      if (e.key === "Escape") {
        if (showStoneColorDialog) {
          setShowStoneColorDialog(false);
        } else {
          onClose();
        }
      }
    };

    document.addEventListener("keydown", handleEscape);
    return () => {
      document.removeEventListener("keydown", handleEscape);
    };
  }, [onClose, showStoneColorDialog]);

  const handleOverlayClick = (e) => {
    if (e.target === e.currentTarget) {
      onClose();
    }
  };

  const handleModeClick = (mode) => {
    if (expandedMode === mode) {
      setExpandedMode(null);
    } else {
      setExpandedMode(mode);
      if (mode === "pvp") {
        setPveSettings({ aiLevel: null, boardSize: null });
      } else {
        setPvpSettings({ matchType: null, boardSize: null });
      }
    }
  };

  const isPvpComplete = pvpSettings.matchType && 
    (pvpSettings.matchType === 'create' ? pvpSettings.boardSize : 
     pvpSettings.matchType === 'code' ? pvpSettings.roomCode.length === 6 :
     pvpSettings.matchType === 'online' ? pvpSettings.boardSize : false);
  const isPveComplete = pveSettings.aiLevel && pveSettings.boardSize;

  const handleCreateClick = () => {
    if (expandedMode === "pvp" && isPvpComplete) {
      if (pvpSettings.matchType === "code") {
        // Tham gia bàn bằng mã
        onCreateMatch("pvp-join", null, null, pvpSettings.roomCode);
      } else if (pvpSettings.matchType === "create") {
        // Tạo bàn mới
        const boardSize = pvpSettings.boardSize;
        const timeControlMinutes = pvpSettings.timeControlMinutes;
        onCreateMatch("pvp", timeControlMinutes, boardSize);
      } else if (pvpSettings.matchType === "online") {
        // Ghép online
        onCreateMatch("matchmaking", null, pvpSettings.boardSize);
      }
      onClose();
    } else if (expandedMode === "pve" && isPveComplete) {
      // PVE → mở dialog chọn màu
      setShowStoneColorDialog(true);
    }
  };

  const handleStoneColorSubmit = (color) => {
    if (!isPveComplete) {
      setShowStoneColorDialog(false);
      return;
    }
    const level = pveSettings.aiLevel;
    const boardSize = pveSettings.boardSize;
    onCreateMatch("ai", level, boardSize, color);
    setShowStoneColorDialog(false);
    onClose();
  };

  return (
    <div className="match-dialog-overlay" onClick={handleOverlayClick}>
      <div
        className="match-dialog match-dialog-new"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="mode-dialog">
          {/* Header */}
          <div className="mode-header">
            <h2>CHỌN CHẾ ĐỘ</h2>
            <button type="button" className="mode-close" onClick={onClose}>
              ×
            </button>
          </div>

          <div className="mode-divider" />

          {/* PVP Card */}
          <div
            className={
              "mode-card " + (expandedMode === "pvp" ? "mode-card-active" : "")
            }
          >
            {expandedMode === "pvp" && (
              <>
                <span className="sparkle" />
                <span className="sparkle" />
                <span className="sparkle" />
                <span className="sparkle" />
              </>
            )}
            <button
              type="button"
              className="mode-card-header"
              onClick={() => handleModeClick("pvp")}
            >
              <div className="mode-card-left">
                <div className="mode-icon mode-icon-pvp">
                  <FaUsers />
                </div>
                <div className="mode-text">
                  <div className="mode-title">Chế độ PVP</div>
                  <div className="mode-subtitle">Đấu với người chơi</div>
                </div>
              </div>
              {expandedMode === "pvp" && (
                <div className="mode-check">
                  <span>✓</span>
                </div>
              )}
            </button>

            {expandedMode === "pvp" && (
              <div className="mode-content fade-in">
                <div className="mode-section">
                  <div className="mode-section-label">Loại trận đấu</div>
                  <div className="mode-button-grid mode-button-grid-3">
                    <button
                      type="button"
                      className={
                        "mode-pill " +
                        (pvpSettings.matchType === "create"
                          ? "mode-pill-active"
                          : "")
                      }
                      onClick={() =>
                        setPvpSettings({ ...pvpSettings, matchType: "create", roomCode: "" })
                      }
                    >
                      <span>Tạo bàn</span>
                    </button>
                    <button
                      type="button"
                      className={
                        "mode-pill " +
                        (pvpSettings.matchType === "code"
                          ? "mode-pill-active"
                          : "")
                      }
                      onClick={() =>
                        setPvpSettings({ ...pvpSettings, matchType: "code", boardSize: null })
                      }
                    >
                      <span>Mã tham gia</span>
                    </button>
                    <button
                      type="button"
                      className={
                        "mode-pill " +
                        (pvpSettings.matchType === "online"
                          ? "mode-pill-active"
                          : "")
                      }
                      onClick={() =>
                        setPvpSettings({ ...pvpSettings, matchType: "online", roomCode: "" })
                      }
                    >
                      <span>Ghép online</span>
                    </button>
                  </div>
                </div>

                {pvpSettings.matchType === "code" ? (
                  <div className="mode-section">
                    <div className="mode-section-label">Nhập mã bàn (6 ký tự)</div>
                    <input
                      type="text"
                      value={pvpSettings.roomCode}
                      onChange={(e) => {
                        const value = e.target.value.toUpperCase().replace(/[^A-Z0-9]/g, '').slice(0, 6);
                        setPvpSettings({ ...pvpSettings, roomCode: value });
                      }}
                      placeholder="ABCD12"
                      maxLength={6}
                      className="mode-room-code-input"
                      autoFocus
                    />
                  </div>
                ) : (
                  <div className="mode-section">
                    <div className="mode-section-label">Kích thước bàn cờ</div>
                    <div className="mode-button-grid mode-button-grid-3">
                    {[
                      { label: "9x9", value: 9 },
                      { label: "13x13", value: 13 },
                      { label: "19x19", value: 19 },
                    ].map((size) => (
                      <button
                        key={size.value}
                        type="button"
                        className={
                          "mode-pill " +
                          (pvpSettings.boardSize === size.value
                            ? "mode-pill-active"
                            : "")
                        }
                        onClick={() => {
                          // Tự động gán thời gian theo kích thước bàn cờ
                          let minutes = 10;
                          if (size.value === 13) minutes = 20;
                          else if (size.value === 19) minutes = 30;

                          setPvpSettings({
                            ...pvpSettings,
                            boardSize: size.value,
                            timeControlMinutes: minutes,
                          });
                        }}
                      >
                        <span>{size.label}</span>
                      </button>
                    ))}
                    </div>
                  </div>
                )}

                {/* Thời gian cho mỗi người chơi được gán tự động theo kích thước bàn cờ
                    (9x9 → 10 phút, 13x13 → 20 phút, 19x19 → 30 phút).
                    Không hiển thị phần UI riêng để người chơi chỉnh. */}
              </div>
            )}
          </div>

          {/* PVE Card */}
          <div
            className={
              "mode-card " + (expandedMode === "pve" ? "mode-card-active" : "")
            }
          >
            {expandedMode === "pve" && (
              <>
                <span className="sparkle" />
                <span className="sparkle" />
                <span className="sparkle" />
                <span className="sparkle" />
              </>
            )}
            <button
              type="button"
              className="mode-card-header"
              onClick={() => handleModeClick("pve")}
            >
              <div className="mode-card-left">
                <div className="mode-icon mode-icon-pve">
                  <FaRobot />
                </div>
                <div className="mode-text">
                  <div className="mode-title">Chế độ PVE</div>
                  <div className="mode-subtitle">Đấu với máy</div>
                </div>
              </div>
              {expandedMode === "pve" && (
                <div className="mode-check">
                  <span>✓</span>
                </div>
              )}
            </button>

            {expandedMode === "pve" && (
              <div className="mode-content fade-in">
                <div className="mode-section">
                  <div className="mode-section-label">Cấp độ AI</div>
                  <div className="mode-button-grid mode-button-grid-2">
                    {[
                      { label: "Dễ", value: 1 },
                      { label: "Trung bình", value: 2 },
                      { label: "Khó", value: 3 },
                      { label: "Siêu khó", value: 4 },
                    ].map((level) => (
                      <button
                        key={level.value}
                        type="button"
                        className={
                          "mode-pill " +
                          (pveSettings.aiLevel === level.value
                            ? "mode-pill-active"
                            : "")
                        }
                        onClick={() =>
                          setPveSettings({
                            ...pveSettings,
                            aiLevel: level.value,
                          })
                        }
                      >
                        {level.label}
                      </button>
                    ))}
                  </div>
                </div>

                <div className="mode-section">
                  <div className="mode-section-label">Kích thước bàn cờ</div>
                  <div className="mode-button-grid mode-button-grid-3">
                    {[
                      { label: "9x9", value: 9 },
                      { label: "13x13", value: 13 },
                      { label: "19x19", value: 19 },
                    ].map((size) => (
                      <button
                        key={size.value}
                        type="button"
                        className={
                          "mode-pill " +
                          (pveSettings.boardSize === size.value
                            ? "mode-pill-active"
                            : "")
                        }
                        onClick={() =>
                          setPveSettings({
                            ...pveSettings,
                            boardSize: size.value,
                          })
                        }
                      >
                        <span>{size.label}</span>
                      </button>
                    ))}
                  </div>
                </div>
              </div>
            )}
          </div>

          {(isPvpComplete || isPveComplete) && (
            <div className="mode-footer">
              <button
                type="button"
                className="mode-create-button"
                onClick={handleCreateClick}
              >
                TẠO TRẬN ĐẤU
              </button>
            </div>
          )}
        </div>
      </div>

      {showStoneColorDialog && (
        <StoneColorDialog
          onClose={() => setShowStoneColorDialog(false)}
          onSubmit={handleStoneColorSubmit}
        />
      )}
    </div>
  );
};

export default MatchDialog;

MatchDialog.propTypes = {
  onClose: PropTypes.func,
  onCreateMatch: PropTypes.func,
};
