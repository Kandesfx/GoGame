import { FaHistory } from 'react-icons/fa'
import './MoveHistory.css'

const MoveHistory = ({ moves = [] }) => {
  const formatMove = (move, index) => {
    if (!move) return null
    
    // Get color - ưu tiên từ move, nếu không có thì tính từ index
    const color = move.color || (index % 2 === 0 ? 'B' : 'W')
    
    // Pass move - kiểm tra position null hoặc undefined
    if (move.position === null || move.position === undefined) {
      return {
        number: move.number || (index + 1),
        color: color,
        text: 'Pass',
        isPass: true,
        captured: move.captured || []
      }
    }
    
    // Regular move - position có thể là array [x, y] hoặc object {x, y}
    let x, y
    if (Array.isArray(move.position) && move.position.length === 2) {
      [x, y] = move.position
    } else if (move.x !== undefined && move.y !== undefined) {
      x = move.x
      y = move.y
    } else {
      console.warn('Invalid move position format:', move)
      return null
    }
    
    const column = String.fromCharCode(65 + x) // A, B, C, ...
    const row = y + 1
    
    return {
      number: move.number || (index + 1),
      color: color,
      text: `${column}${row}`,
      isPass: false,
      captured: move.captured || []
    }
  }

  const formattedMoves = moves
    .map((move, index) => formatMove(move, index))
    .filter(move => move !== null)
    .reverse() // Hiển thị nước mới nhất ở trên

  return (
    <div className="move-history">
      <div className="move-history-header">
        <FaHistory className="move-history-icon" />
        <h3>Lịch sử nước đánh</h3>
      </div>
      <div className="move-history-list">
        {formattedMoves.length === 0 ? (
          <div className="move-history-empty">Chưa có nước đánh</div>
        ) : (
          formattedMoves.map((move) => (
            <div 
              key={move.number} 
              className={`move-history-item ${move.color === 'B' ? 'move-black' : 'move-white'}`}
            >
              <span className="move-number">{move.number}</span>
              <span className="move-color">{move.color}</span>
              <span className="move-text">{move.text}</span>
            </div>
          ))
        )}
      </div>
    </div>
  )
}

export default MoveHistory

