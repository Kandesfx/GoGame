import './GameControls.css'

const GameControls = ({ onPass, onResign, onUndo, disabled = false, undoDisabled = false }) => {
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
    </div>
  )
}

export default GameControls

