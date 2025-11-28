import { useState, useEffect } from 'react'
import { FaTimes, FaGamepad, FaRobot, FaUsers } from 'react-icons/fa'
import StoneColorDialog from './StoneColorDialog'
import './MatchDialog.css'

const MatchDialog = ({ onClose, onCreateMatch }) => {
  const [matchType, setMatchType] = useState('ai')
  const [aiLevel, setAiLevel] = useState(1)
  const [boardSize, setBoardSize] = useState(9)
  const [showStoneColorDialog, setShowStoneColorDialog] = useState(false)

  // Handle Escape key to close dialog
  useEffect(() => {
    const handleEscape = (e) => {
      if (e.key === 'Escape') {
        console.log('🔴 Escape key pressed - closing dialog')
        if (showStoneColorDialog) {
          setShowStoneColorDialog(false)
        } else {
          onClose()
        }
      }
    }
    
    document.addEventListener('keydown', handleEscape)
    return () => {
      document.removeEventListener('keydown', handleEscape)
    }
  }, [onClose, showStoneColorDialog])

  const handleSubmit = (e) => {
    e.preventDefault()
    if (matchType === 'ai') {
      // Show stone color dialog for AI matches
      setShowStoneColorDialog(true)
    } else {
      console.log('✅ Creating match:', { matchType, aiLevel, boardSize })
      onCreateMatch(matchType, aiLevel, boardSize)
    }
  }

  const handleStoneColorSubmit = (color) => {
    console.log('✅ Creating AI match with color:', color, { matchType, aiLevel, boardSize })
    console.log('🎨 Calling onCreateMatch with params:', matchType, aiLevel, boardSize, color)
    onCreateMatch(matchType, aiLevel, boardSize, color)
    setShowStoneColorDialog(false)
  }

  const handleOverlayClick = (e) => {
    // Only close if clicking directly on overlay, not on dialog content
    if (e.target === e.currentTarget) {
      console.log('🔴 Overlay clicked - closing dialog')
      onClose()
    }
  }

  return (
    <div className="match-dialog-overlay" onClick={handleOverlayClick}>
      <div className="match-dialog" onClick={(e) => e.stopPropagation()}>
        <div className="match-dialog-header">
          <div className="match-dialog-title">
            <FaGamepad className="dialog-icon" />
            <h2>Tạo Trận Đấu</h2>
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
        <form onSubmit={handleSubmit}>
          <div className="form-group">
            <label>
              <FaUsers className="label-icon" />
              Loại trận đấu:
            </label>
            <select
              value={matchType}
              onChange={(e) => setMatchType(e.target.value)}
            >
              <option value="ai">Đấu với AI</option>
              <option value="pvp">Đấu với người (Mã tham gia)</option>
              <option value="matchmaking">Ghép người chơi online</option>
            </select>
          </div>

          {matchType === 'ai' && (
            <div className="form-group">
              <label>
                <FaRobot className="label-icon" />
                Cấp độ AI:
              </label>
              <select
                value={aiLevel}
                onChange={(e) => setAiLevel(parseInt(e.target.value))}
                className="level-select"
              >
                <option value={1}>Dễ</option>
                <option value={2}>Trung bình</option>
                <option value={3}>Khó</option>
                <option value={4}>Siêu khó</option>
              </select>
            </div>
          )}

          <div className="form-group">
            <label>
              <FaGamepad className="label-icon" />
              Kích thước bàn cờ:
            </label>
            <select
              value={boardSize}
              onChange={(e) => setBoardSize(parseInt(e.target.value))}
            >
              <option value="9">9x9 (Nhanh)</option>
              <option value="13">13x13 (Trung bình)</option>
              <option value="19">19x19 (Chuẩn)</option>
            </select>
          </div>

          <div className="dialog-actions">
            <button type="button" onClick={onClose} className="btn btn-secondary">
              Hủy
            </button>
            <button type="submit" className="btn btn-primary">
              Tạo
            </button>
          </div>
        </form>
      </div>

      {showStoneColorDialog && (
        <StoneColorDialog
          onClose={() => setShowStoneColorDialog(false)}
          onSubmit={handleStoneColorSubmit}
        />
      )}
    </div>
  )
}

export default MatchDialog

