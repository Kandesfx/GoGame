import './GameControls.css'

const GameControls = ({ onPass, onResign, onUndo, onHint, onAnalysis, onReview, disabled = false, undoDisabled = false }) => {
  return (
    <div className="game-controls">
      <div className="controls-row">
        <button onClick={onPass} className="btn btn-primary" disabled={disabled} title="Bỏ lượt">
          <span>Bỏ lượt</span>
        </button>
        <button onClick={onResign} className="btn btn-danger" disabled={disabled} title="Đầu hàng">
          <span>Đầu hàng</span>
        </button>
        {onUndo && (
          <button onClick={onUndo} className="btn btn-secondary" disabled={disabled || undoDisabled} title="Hoàn tác nước đi">
            <span>Hoàn tác</span>
          </button>
        )}
      </div>
      <div className="controls-row">
        <button onClick={onHint} className="btn btn-warning" disabled={disabled} title="Gợi ý">
          <span>Gợi ý</span>
        </button>
        <button onClick={onAnalysis} className="btn btn-warning" disabled={disabled} title="Phân tích">
          <span>Phân tích</span>
        </button>
        <button onClick={onReview} className="btn btn-warning" disabled={disabled} title="Xem lại">
          <span>Xem lại</span>
        </button>
      </div>
    </div>
  )
}

export default GameControls

