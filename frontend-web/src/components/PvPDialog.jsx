import { useState, useEffect } from 'react'
import { FaTimes, FaUsers, FaGamepad, FaCopy, FaCheck } from 'react-icons/fa'
import api from '../services/api'
import './MatchDialog.css'

const PvPDialog = ({ onClose, onMatchCreated, onMatchJoined }) => {
  const [mode, setMode] = useState('create') // 'create' or 'join'
  const [boardSize, setBoardSize] = useState(9)
  const [timeControlMinutes, setTimeControlMinutes] = useState(10) // Mặc định 10 phút
  const [roomCode, setRoomCode] = useState('')
  const [createdRoomCode, setCreatedRoomCode] = useState('')
  const [copied, setCopied] = useState(false)
  const [error, setError] = useState(null)
  const [loading, setLoading] = useState(false)

  // Handle Escape key to close dialog
  useEffect(() => {
    const handleEscape = (e) => {
      if (e.key === 'Escape') {
        onClose()
      }
    }
    
    document.addEventListener('keydown', handleEscape)
    return () => {
      document.removeEventListener('keydown', handleEscape)
    }
  }, [onClose])

  const handleCreateMatch = async (e) => {
    e.preventDefault()
    setError(null)
    setLoading(true)
    
    try {
      const response = await api.post('/matches/pvp', {
        board_size: boardSize,
        time_control_minutes: timeControlMinutes
      })
      const { match, join_code } = response.data
      setCreatedRoomCode(join_code)
      if (onMatchCreated) {
        onMatchCreated(match)
      }
    } catch (error) {
      setError(error.response?.data?.detail || error.message || 'Không thể tạo bàn')
      console.error('Error creating PvP match:', error)
    } finally {
      setLoading(false)
    }
  }

  const handleJoinMatch = async (e) => {
    e.preventDefault()
    setError(null)
    
    if (!roomCode || roomCode.length !== 6) {
      setError('Mã bàn phải có 6 ký tự')
      return
    }
    
    setLoading(true)
    
    try {
      const response = await api.post('/matches/pvp/join', {
        room_code: roomCode.toUpperCase()
      })
      if (onMatchJoined) {
        onMatchJoined(response.data)
      }
    } catch (error) {
      setError(error.response?.data?.detail || error.message || 'Không thể tham gia bàn')
      console.error('Error joining PvP match:', error)
    } finally {
      setLoading(false)
    }
  }

  const handleCopyCode = () => {
    if (createdRoomCode) {
      navigator.clipboard.writeText(createdRoomCode)
      setCopied(true)
      setTimeout(() => setCopied(false), 2000)
    }
  }

  const handleOverlayClick = (e) => {
    if (e.target === e.currentTarget) {
      onClose()
    }
  }

  return (
    <div className="match-dialog-overlay" onClick={handleOverlayClick}>
      <div className="match-dialog" onClick={(e) => e.stopPropagation()}>
        <div className="match-dialog-header">
          <div className="match-dialog-title">
            <FaUsers className="dialog-icon" />
            <h2>Đấu với Người</h2>
          </div>
          <button
            type="button"
            onClick={onClose}
            className="match-dialog-close"
            title="Đóng (Esc)"
          >
            <FaTimes />
          </button>
        </div>

        {createdRoomCode ? (
          <div className="pvp-room-created">
            <div className="room-code-display">
              <h3>Mã bàn của bạn:</h3>
              <div className="room-code-box">
                <span className="room-code-text">{createdRoomCode}</span>
                <button
                  type="button"
                  onClick={handleCopyCode}
                  className="copy-button"
                  title="Sao chép mã"
                >
                  {copied ? <FaCheck /> : <FaCopy />}
                </button>
              </div>
              <p className="room-code-hint">Chia sẻ mã này với người chơi khác để họ tham gia</p>
            </div>
            <div className="dialog-actions">
              <button type="button" onClick={onClose} className="btn btn-primary">
                Đóng
              </button>
            </div>
          </div>
        ) : (
          <>
            <div className="pvp-mode-selector">
              <button
                type="button"
                className={`mode-button ${mode === 'create' ? 'active' : ''}`}
                onClick={() => {
                  setMode('create')
                  setError(null)
                }}
              >
                Tạo bàn
              </button>
              <button
                type="button"
                className={`mode-button ${mode === 'join' ? 'active' : ''}`}
                onClick={() => {
                  setMode('join')
                  setError(null)
                }}
              >
                Tham gia bàn
              </button>
            </div>

            {mode === 'create' ? (
              <form onSubmit={handleCreateMatch}>
                <div className="form-group">
                  <label>
                    <FaGamepad className="label-icon" />
                    Kích thước bàn cờ:
                  </label>
                  <select
                    value={boardSize}
                    onChange={(e) => setBoardSize(parseInt(e.target.value))}
                    disabled={loading}
                  >
                    <option value="9">9x9 (Nhanh)</option>
                    <option value="13">13x13 (Trung bình)</option>
                    <option value="19">19x19 (Chuẩn)</option>
                  </select>
                </div>

                <div className="form-group">
                  <label>
                    ⏱️ Thời gian cho mỗi người chơi (phút):
                  </label>
                  <select
                    value={timeControlMinutes}
                    onChange={(e) => setTimeControlMinutes(parseInt(e.target.value))}
                    disabled={loading}
                  >
                    <option value="5">5 phút</option>
                    <option value="10">10 phút</option>
                    <option value="15">15 phút</option>
                    <option value="20">20 phút</option>
                    <option value="30">30 phút</option>
                  </select>
                </div>

                {error && (
                  <div className="error-message">
                    {error}
                  </div>
                )}

                <div className="dialog-actions">
                  <button type="button" onClick={onClose} className="btn btn-secondary">
                    Hủy
                  </button>
                  <button type="submit" className="btn btn-primary" disabled={loading}>
                    {loading ? 'Đang tạo...' : 'Tạo bàn'}
                  </button>
                </div>
              </form>
            ) : (
              <form onSubmit={handleJoinMatch}>
                <div className="form-group">
                  <label>
                    <FaUsers className="label-icon" />
                    Nhập mã bàn (6 ký tự):
                  </label>
                  <input
                    type="text"
                    value={roomCode}
                    onChange={(e) => {
                      const value = e.target.value.toUpperCase().replace(/[^A-Z0-9]/g, '').slice(0, 6)
                      setRoomCode(value)
                      setError(null)
                    }}
                    placeholder="ABCD12"
                    maxLength={6}
                    disabled={loading}
                    className="room-code-input"
                    style={{
                      textAlign: 'center',
                      fontSize: '1.5rem',
                      letterSpacing: '0.5rem',
                      fontWeight: 'bold'
                    }}
                  />
                </div>

                {error && (
                  <div className="error-message">
                    {error}
                  </div>
                )}

                <div className="dialog-actions">
                  <button type="button" onClick={onClose} className="btn btn-secondary">
                    Hủy
                  </button>
                  <button type="submit" className="btn btn-primary" disabled={loading || roomCode.length !== 6}>
                    {loading ? 'Đang tham gia...' : 'Tham gia'}
                  </button>
                </div>
              </form>
            )}
          </>
        )}
      </div>
    </div>
  )
}

export default PvPDialog

