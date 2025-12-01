import { useEffect } from 'react'
import { FaExclamationTriangle, FaTimes } from 'react-icons/fa'
import './KoDialog.css'

const KoDialog = ({ isOpen, onClose, koPosition }) => {
  // Handle Escape key to close dialog
  useEffect(() => {
    if (!isOpen) return
    
    const handleEscape = (e) => {
      if (e.key === 'Escape') {
        onClose()
      }
    }
    
    document.addEventListener('keydown', handleEscape)
    return () => {
      document.removeEventListener('keydown', handleEscape)
    }
  }, [isOpen, onClose])

  if (!isOpen) return null

  const handleOverlayClick = (e) => {
    // Only close if clicking directly on overlay, not on dialog content
    if (e.target === e.currentTarget) {
      onClose()
    }
  }

  return (
    <div className="ko-dialog-overlay" onClick={handleOverlayClick}>
      <div className="ko-dialog" onClick={(e) => e.stopPropagation()}>
        <div className="ko-dialog-header">
          <div className="ko-dialog-title">
            <FaExclamationTriangle className="ko-dialog-icon" />
            <h2>Tình Trạng Cướp Cờ KO</h2>
          </div>
          <button
            type="button"
            onClick={onClose}
            className="ko-dialog-close"
            title="Đóng (Esc)"
          >
            <FaTimes />
          </button>
        </div>
        
        <div className="ko-dialog-content">
          <div className="ko-dialog-message">
            <p>
              <strong>Luật KO (Cấm cướp cờ):</strong>
            </p>
            <p>
              Đối thủ vừa ăn 1 quân của bạn tại vị trí ({koPosition ? `${koPosition[0]}, ${koPosition[1]}` : 'N/A'}).
            </p>
            <p>
              Bạn <strong>không được phép</strong> ngay lập tức ăn lại đúng quân vừa bắt tại đúng vị trí đó trong nước tiếp theo.
            </p>
            <p>
              Bạn phải đánh ở chỗ khác trước 1 nước, sau đó mới được quay lại ăn.
            </p>
          </div>
        </div>

        <div className="ko-dialog-actions">
          <button 
            type="button" 
            onClick={onClose} 
            className="btn btn-primary"
            autoFocus
          >
            Đã hiểu
          </button>
        </div>
      </div>
    </div>
  )
}

export default KoDialog

