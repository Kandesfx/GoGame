import { useState, useEffect, useRef } from 'react'
import './Board.css'

// NOTE: In Go, stones are placed on INTERSECTIONS (giao điểm), not in squares.
// Each "cell" in this component represents an intersection point on the board.
const Board = ({ boardSize = 9, stones = {}, onCellClick, lastMove = null, disabled = false, theme = 'classic' }) => {
  const [hoverPos, setHoverPos] = useState(null)
  const boardRef = useRef(null)
  const [gridSize, setGridSize] = useState({ width: 0, height: 0 })
  const [actualCellSize, setActualCellSize] = useState(0)
  const [boardHeight, setBoardHeight] = useState(0) // Store board height for label positioning
  const [boardWidth, setBoardWidth] = useState(0) // Store board width for label positioning

  const handleCellClick = (x, y) => {
    if (disabled) {
      console.log('⚠️ Board is disabled, ignoring click')
      return
    }
    const key = `${x},${y}`
    if (!stones[key] && onCellClick) {
      onCellClick(x, y)
    }
  }

  const handleCellHover = (x, y) => {
    if (disabled) return
    const key = `${x},${y}`
    // Only show hover if cell is empty
    if (!stones[key]) {
      setHoverPos({ x, y })
    } else {
      setHoverPos(null)
    }
  }

  const handleCellLeave = () => {
    setHoverPos(null)
  }

  const getStoneColor = (x, y) => {
    const key = `${x},${y}`
    const color = stones[key]
    return color
  }

  // Calculate grid size and cell size for SVG grid lines - use actual cell dimensions
  useEffect(() => {
    const updateGridSize = () => {
      if (boardRef.current) {
        // Get the first cell to measure actual cell size
        const firstCell = boardRef.current.querySelector('.cell')
        if (firstCell) {
          const cellRect = firstCell.getBoundingClientRect()
          const rect = boardRef.current.getBoundingClientRect()
          
          // Use actual cell width/height (should be square)
          const cellSize = Math.min(cellRect.width, cellRect.height)
          const calculatedGridSize = cellSize * boardSize
          
          setActualCellSize(cellSize)
          setGridSize({
            width: calculatedGridSize,
            height: calculatedGridSize
          })
          setBoardHeight(rect.height) // Store board height for label positioning
          setBoardWidth(rect.width) // Store board width for label positioning
        } else {
          // Fallback: use board dimensions minus padding
          const rect = boardRef.current.getBoundingClientRect()
          const padding = 12
          const fallbackSize = rect.width - padding * 2
          const fallbackCellSize = fallbackSize / boardSize
          
          setActualCellSize(fallbackCellSize)
          setGridSize({
            width: fallbackSize,
            height: fallbackSize
          })
          setBoardHeight(rect.height) // Store board height for label positioning
          setBoardWidth(rect.width) // Store board width for label positioning
        }
      }
    }
    
    // Use ResizeObserver for more accurate updates
    const resizeObserver = new ResizeObserver(() => {
      // Small delay to ensure cells are rendered
      setTimeout(updateGridSize, 10)
    })
    
    if (boardRef.current) {
      resizeObserver.observe(boardRef.current)
      // Initial update with a small delay to ensure DOM is ready
      setTimeout(updateGridSize, 50)
    }
    
    window.addEventListener('resize', updateGridSize)
    
    return () => {
      resizeObserver.disconnect()
      window.removeEventListener('resize', updateGridSize)
    }
  }, [boardSize])

  // Debug: Log stones and boardHeight on mount and when they change
  useEffect(() => {
    if (Object.keys(stones).length > 0) {
      console.log('🔍 Board - Stones received:', stones)
      console.log('🔍 Board - Stone count:', Object.keys(stones).length)
      console.log('🔍 Board - Sample keys:', Object.keys(stones).slice(0, 5))
    }
    console.log('🔍 Board - boardHeight:', boardHeight, 'gridSize:', gridSize)
  }, [stones, boardHeight, gridSize])

  const isStarPoint = (x, y) => {
    if (boardSize === 9) {
      return (x === 2 && y === 2) || (x === 2 && y === 6) || (x === 6 && y === 2) || 
             (x === 6 && y === 6) || (x === 4 && y === 4)
    } else if (boardSize === 19) {
      return (x === 3 || x === 9 || x === 15) && (y === 3 || y === 9 || y === 15)
    } else if (boardSize === 13) {
      return (x === 3 || x === 6 || x === 9) && (y === 3 || y === 6 || y === 9)
    }
    return false
  }

  const isLastMove = (x, y) => {
    if (!lastMove) return false
    return lastMove.x === x && lastMove.y === y
  }

  // Calculate grid line positions - lines pass through intersection points (cell centers)
  // Each intersection is at the center of a cell, so lines are evenly spaced
  const padding = 12 // Must match CSS .board padding
  const gridLines = []
  
  // Use actual cell size if available, otherwise calculate from grid size
  const cellSize = actualCellSize > 0 ? actualCellSize : (gridSize.width > 0 ? gridSize.width / boardSize : 0)
  
  // Check theme
  const isModernTheme = theme === 'modern'
  const isNaturalWoodTheme = theme === 'natural-wood'
  
  // Get grid line color based on theme
  const getGridLineColor = () => {
    if (isModernTheme) {
      return '#00eaff' // Neon cyan for modern theme
    }
    if (isNaturalWoodTheme) {
      return 'rgba(107, 74, 47, 0.35)' // Burned ink on wood - low contrast
    }
    return 'rgba(139, 111, 71, 0.9)' // Classic brown
  }

  const getGridLineWidth = () => {
    if (isNaturalWoodTheme) return '1.6'
    return isModernTheme ? '1.6' : '1.5'
  }

  // Generate horizontal and vertical lines at exact intersection points (cell centers)
  // Lines are evenly spaced: first line at padding + 0.5*cellSize, last at padding + (boardSize-0.5)*cellSize
  if (cellSize > 0 && gridSize.width > 0 && gridSize.height > 0) {
    for (let i = 0; i < boardSize; i++) {
      // Calculate intersection position: center of each cell
      // Line i passes through the center of row/column i
      const intersectionOffset = (i + 0.5) * cellSize
      
      // Horizontal lines (y position) - pass through center of each row
      const y = padding + intersectionOffset
      const x1 = padding + 0.5 * cellSize // Start from first intersection
      const x2 = padding + (boardSize - 0.5) * cellSize // End at last intersection
      gridLines.push(
        <line
          key={`h-${i}`}
          x1={x1}
          y1={y}
          x2={x2}
          y2={y}
          stroke={getGridLineColor()}
          strokeWidth={getGridLineWidth()}
          vectorEffect="non-scaling-stroke"
        />
      )
      
      // Vertical lines (x position) - pass through center of each column
      const x = padding + intersectionOffset
      const y1 = padding + 0.5 * cellSize // Start from first intersection
      const y2 = padding + (boardSize - 0.5) * cellSize // End at last intersection
      gridLines.push(
        <line
          key={`v-${i}`}
          x1={x}
          y1={y1}
          x2={x}
          y2={y2}
          stroke={getGridLineColor()}
          strokeWidth={getGridLineWidth()}
          vectorEffect="non-scaling-stroke"
        />
      )
    }
  }

  // Generate coordinate labels
  const columnLabels = boardSize <= 9 ? ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I'] : 
                       boardSize <= 13 ? ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M'] :
                       ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T']
  // Row labels: 9, 8, 7, ..., 1 (from top to bottom)
  // In Go, row 9 is at the top, row 1 is at the bottom
  const rowLabels = Array.from({ length: boardSize }, (_, i) => boardSize - i) // 9, 8, 7, ..., 1

  // Calculate label positions based on actual cell size
  // Note: padding is already declared above (line 131)
  // Use the same calculation as grid lines to ensure perfect alignment
  const calculateLabelPosition = (index) => {
    if (actualCellSize > 0) {
      // Position at center of intersection: padding + (index + 0.5) * cellSize
      // This matches exactly with grid line positions
      return padding + (index + 0.5) * actualCellSize
    }
    // Fallback: use percentage if cellSize not available
    return null
  }
  
  // Calculate label position as percentage for row labels
  const calculateRowLabelPosition = (index) => {
    if (actualCellSize > 0 && boardHeight > 0) {
      // Position in pixels: padding + (index + 0.5) * actualCellSize
      // This matches exactly with grid line y position
      const positionPx = padding + (index + 0.5) * actualCellSize
      // Convert to percentage relative to board height (including padding)
      return (positionPx / boardHeight) * 100
    }
    // Fallback: use equal spacing
    return ((index + 0.5) / boardSize) * 100
  }
  
  // Calculate label position as percentage for column labels
  const calculateColumnLabelPosition = (index) => {
    if (actualCellSize > 0 && boardWidth > 0) {
      // Position in pixels: padding + (index + 0.5) * actualCellSize
      // This matches exactly with grid line x position
      const positionPx = padding + (index + 0.5) * actualCellSize
      // Convert to percentage relative to board width (including padding)
      return (positionPx / boardWidth) * 100
    }
    // Fallback: use equal spacing
    return ((index + 0.5) / boardSize) * 100
  }

  // Get theme classes
  const getContainerClass = () => {
    if (isModernTheme) return 'board-container-modern'
    if (isNaturalWoodTheme) return 'board-container-natural-wood'
    return ''
  }

  const getBoardClass = () => {
    if (isModernTheme) return 'board-modern'
    if (isNaturalWoodTheme) return 'board-natural-wood'
    return ''
  }

  const getCellClass = () => {
    if (isModernTheme) return 'cell-modern'
    if (isNaturalWoodTheme) return 'cell-natural-wood'
    return ''
  }

  const getStoneClass = () => {
    if (isModernTheme) return 'stone-modern'
    if (isNaturalWoodTheme) return 'stone-natural-wood'
    return ''
  }

  // Generate random rotation/scale for ink splash effect
  const getStoneStyle = (x, y) => {
    if (!isNaturalWoodTheme) return {}
    // Use position as seed for consistent randomness
    const seed = (x * 19 + y) * 13
    const rotation = ((seed % 30) - 15) // -15 to 15 degrees
    const scaleX = 0.95 + ((seed % 10) / 100) // 0.95 to 1.04
    const scaleY = 0.95 + (((seed * 7) % 10) / 100) // 0.95 to 1.04
    return {
      '--stone-rotation': `${rotation}deg`,
      '--stone-scale-x': scaleX,
      '--stone-scale-y': scaleY
    }
  }

  return (
    <div className={`board-container ${getContainerClass()}`}>
      {/* Column labels (top) */}
      <div className="coordinate-labels column-labels-top" style={{ width: boardWidth > 0 ? `${boardWidth}px` : '100%' }}>
        {columnLabels.slice(0, boardSize).map((label, i) => {
          const positionPercent = calculateColumnLabelPosition(i)
          return (
            <div 
              key={`col-top-${i}`} 
              className="coordinate-label"
              style={{ 
                position: 'absolute',
                left: `${positionPercent}%`,
                transform: 'translateX(-50%)'
              }}
            >
              {label}
            </div>
          )
        })}
      </div>
      
      <div className="board-with-labels">
        {/* Row labels (left) */}
        <div 
          className="coordinate-labels row-labels-left"
          style={{ height: boardHeight > 0 ? `${boardHeight}px` : '100%' }}
        >
          {rowLabels.map((label, i) => {
            // Calculate position to align with grid line at intersection i
            const positionPercent = calculateRowLabelPosition(i)
            return (
              <div 
                key={`row-left-${i}`} 
                className="coordinate-label"
                style={{ top: `${positionPercent}%` }}
              >
                {label}
              </div>
            )
          })}
        </div>
        
        <div 
          ref={boardRef}
          className={`board ${getBoardClass()}`}
          style={{ 
            gridTemplateColumns: `repeat(${boardSize}, 1fr)`,
            gridTemplateRows: `repeat(${boardSize}, 1fr)`,
            '--board-size': boardSize
          }}
        >
        {/* SVG overlay for precise grid lines */}
        {gridSize.width > 0 && (
          <svg
            className="grid-lines-overlay"
            style={{
              position: 'absolute',
              top: 0,
              left: 0,
              width: '100%',
              height: '100%',
              pointerEvents: 'none',
              zIndex: 0
            }}
          >
            {gridLines}
          </svg>
        )}
        {/* Render grid cells - each cell represents an intersection point */}
        {Array.from({ length: boardSize * boardSize }).map((_, index) => {
          const x = index % boardSize
          const y = Math.floor(index / boardSize)
          const key = `${x},${y}`
          const stoneColor = getStoneColor(x, y)
          const isHovered = hoverPos && hoverPos.x === x && hoverPos.y === y
          const isStar = isStarPoint(x, y)
          const isLast = isLastMove(x, y)

          return (
            <div
              key={key}
              className={`cell ${isStar ? 'star-point' : ''} ${disabled ? 'cell-disabled' : ''} ${getCellClass()}`}
              onClick={() => handleCellClick(x, y)}
              onMouseEnter={() => !disabled && handleCellHover(x, y)}
              onMouseLeave={handleCellLeave}
              style={{ 
                cursor: disabled ? 'not-allowed' : 'pointer', 
                opacity: disabled ? 0.5 : 1 
              }}
              data-intersection-position={key}
              title={`Intersection at (${x}, ${y})`}
            >
              {/* Stone placed at intersection (center of cell) */}
              {stoneColor ? (
                <div 
                  className={`stone stone-${stoneColor.toLowerCase()} ${isLast ? 'stone-last-move' : ''} ${getStoneClass()}`}
                  style={getStoneStyle(x, y)}
                  data-stone-color={stoneColor}
                  data-position={key}
                  title={`Stone at ${key}, color: ${stoneColor}`}
                />
              ) : null}
              {/* Hover indicator at intersection point */}
              {isHovered && !stoneColor && (
                <div className="hover-indicator" />
              )}
            </div>
          )
               })}
             </div>
            
            {/* Row labels (right) */}
            <div 
              className="coordinate-labels row-labels-right"
              style={{ height: boardHeight > 0 ? `${boardHeight}px` : '100%' }}
            >
              {rowLabels.map((label, i) => {
                const positionPercent = calculateRowLabelPosition(i)
                return (
                  <div 
                    key={`row-right-${i}`} 
                    className="coordinate-label"
                    style={{ top: `${positionPercent}%` }}
                  >
                    {label}
                  </div>
                )
              })}
            </div>
          </div>
          
          {/* Column labels (bottom) */}
          <div className="coordinate-labels column-labels-bottom" style={{ width: boardWidth > 0 ? `${boardWidth}px` : '100%' }}>
            {columnLabels.slice(0, boardSize).map((label, i) => {
              const positionPercent = calculateColumnLabelPosition(i)
              return (
                <div 
                  key={`col-bottom-${i}`} 
                  className="coordinate-label"
                  style={{ 
                    position: 'absolute',
                    left: `${positionPercent}%`,
                    transform: 'translateX(-50%)'
                  }}
                >
                  {label}
                </div>
              )
            })}
          </div>
        </div>
  )
}

export default Board


