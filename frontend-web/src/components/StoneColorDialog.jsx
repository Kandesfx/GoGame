import { useState } from 'react'
import { FaTimes, FaCheck } from 'react-icons/fa'
import './StoneColorDialog.css'

const StoneColorDialog = ({ onClose, onSubmit }) => {
  const [selectedColor, setSelectedColor] = useState('black')

  const handleSubmit = () => {
    console.log('🎨 StoneColorDialog: submitting color:', selectedColor)
    onSubmit(selectedColor)
  }

  const handleColorSelect = (color) => {
    console.log('🎨 StoneColorDialog: selected color:', color)
    setSelectedColor(color)
  }

  const handleOverlayClick = (e) => {
    if (e.target === e.currentTarget) {
      onClose()
    }
  }

  return (
    <div className="stone-color-overlay" onClick={handleOverlayClick}>
      <div className="stone-color-dialog">
        <button className="stone-color-close" onClick={onClose}>
          <FaTimes />
        </button>
        
        <h2 className="stone-color-title">CHỌN MÀU QUÂN CỜ</h2>
        
        <div className="stone-color-divider"></div>
        
        <div className="stone-color-options">
          <div 
            className={`stone-option-card ${selectedColor === 'black' ? 'active' : ''}`}
            onClick={() => handleColorSelect('black')}
          >
            {selectedColor === 'black' && (
              <div className="check-badge">
                <FaCheck />
              </div>
            )}
            <div className="stone-circle black-stone">
              <div className="stone-highlight"></div>
            </div>
            <div className="stone-info">
              <span className="stone-label">Quân Đen</span>
              <span className="stone-subtext">Đi trước</span>
            </div>
          </div>
          
          <div 
            className={`stone-option-card ${selectedColor === 'white' ? 'active' : ''}`}
            onClick={() => handleColorSelect('white')}
          >
            {selectedColor === 'white' && (
              <div className="check-badge">
                <FaCheck />
              </div>
            )}
            <div className="stone-circle white-stone">
              <div className="stone-glow"></div>
            </div>
            <div className="stone-info">
              <span className="stone-label">Quân Trắng</span>
              <span className="stone-subtext">Đi sau</span>
            </div>
          </div>
        </div>
        
        <div className="stone-color-actions">
          <button className="btn-cancel" onClick={onClose}>
            Hủy
          </button>
          <button className="btn-create" onClick={handleSubmit}>
            Tạo
          </button>
        </div>
      </div>
    </div>
  )
}

export default StoneColorDialog

