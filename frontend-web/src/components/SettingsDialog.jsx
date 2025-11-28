import { useState } from 'react'
import { FaTimes, FaCog, FaPalette, FaVolumeUp, FaVolumeMute, FaLanguage, FaChessBoard } from 'react-icons/fa'
import './SettingsDialog.css'

const SettingsDialog = ({ isOpen, onClose, settings, onSettingsChange }) => {
  const [localSettings, setLocalSettings] = useState(settings || {
    soundEnabled: true,
    showCoordinates: true,
    showLastMove: true,
    boardTheme: 'classic',
    animationSpeed: 'normal'
  })

  if (!isOpen) return null

  const handleChange = (key, value) => {
    const newSettings = { ...localSettings, [key]: value }
    setLocalSettings(newSettings)
    if (onSettingsChange) {
      onSettingsChange(newSettings)
    }
  }

  const handleSave = () => {
    // Lưu vào localStorage
    localStorage.setItem('goGameSettings', JSON.stringify(localSettings))
    if (onSettingsChange) {
      onSettingsChange(localSettings)
    }
    onClose()
  }

  return (
    <div className="settings-overlay" onClick={onClose}>
      <div className="settings-dialog" onClick={(e) => e.stopPropagation()}>
        <div className="settings-header">
          <div className="settings-title">
            <FaCog className="settings-icon" />
            <h2>Cài Đặt</h2>
          </div>
          <button className="settings-close" onClick={onClose}>
            <FaTimes />
          </button>
        </div>

        <div className="settings-content">
          {/* Âm thanh */}
          <div className="settings-section">
            <div className="settings-section-header">
              <FaVolumeUp className="section-icon" />
              <h3>Âm Thanh</h3>
            </div>
            <div className="settings-option">
              <label className="settings-switch">
                <input
                  type="checkbox"
                  checked={localSettings.soundEnabled}
                  onChange={(e) => handleChange('soundEnabled', e.target.checked)}
                />
                <span className="slider"></span>
                <span className="settings-label">Bật âm thanh</span>
              </label>
            </div>
          </div>

          {/* Hiển thị */}
          <div className="settings-section">
            <div className="settings-section-header">
              <FaChessBoard className="section-icon" />
              <h3>Hiển Thị</h3>
            </div>
            <div className="settings-option">
              <label className="settings-switch">
                <input
                  type="checkbox"
                  checked={localSettings.showCoordinates}
                  onChange={(e) => handleChange('showCoordinates', e.target.checked)}
                />
                <span className="slider"></span>
                <span className="settings-label">Hiển thị tọa độ</span>
              </label>
            </div>
            <div className="settings-option">
              <label className="settings-switch">
                <input
                  type="checkbox"
                  checked={localSettings.showLastMove}
                  onChange={(e) => handleChange('showLastMove', e.target.checked)}
                />
                <span className="slider"></span>
                <span className="settings-label">Đánh dấu nước đi cuối</span>
              </label>
            </div>
          </div>

          {/* Giao diện */}
          <div className="settings-section">
            <div className="settings-section-header">
              <FaPalette className="section-icon" />
              <h3>Giao Diện</h3>
            </div>
            <div className="settings-option">
              <label className="settings-label">Chủ đề bàn cờ</label>
              <select
                className="settings-select"
                value={localSettings.boardTheme}
                onChange={(e) => handleChange('boardTheme', e.target.value)}
              >
                <option value="classic">Cổ Điển</option>
                <option value="modern">Hiện Đại</option>
                <option value="natural-wood">Gỗ Tự Nhiên</option>
              </select>
            </div>
            <div className="settings-option">
              <label className="settings-label">Tốc độ animation</label>
              <select
                className="settings-select"
                value={localSettings.animationSpeed}
                onChange={(e) => handleChange('animationSpeed', e.target.value)}
              >
                <option value="slow">Chậm</option>
                <option value="normal">Bình thường</option>
                <option value="fast">Nhanh</option>
              </select>
            </div>
          </div>
        </div>

        <div className="settings-footer">
          <button className="btn btn-secondary" onClick={onClose}>
            Hủy
          </button>
          <button className="btn btn-primary" onClick={handleSave}>
            Lưu
          </button>
        </div>
      </div>
    </div>
  )
}

export default SettingsDialog

