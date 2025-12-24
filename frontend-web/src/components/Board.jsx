import { useState, useEffect, useRef } from 'react'
import './Board.css'

/**
 * ============================================================================
 * ML VISUALIZATION HELPER FUNCTION
 * ============================================================================
 * 
 * Render ML analysis visualization trực tiếp trong Board component để đảm bảo:
 * - Khớp chính xác với board cells (dùng cùng coordinate system)
 * - Tự động sync khi board resize
 * - Hiệu năng tốt (render trong cùng SVG với grid lines)
 * 
 * @param {Object} mlAnalysisData - Data từ ML analysis service
 * @param {string} mlVisualizationMode - Mode: 'threats', 'attacks', 'intent'
 * @param {number} cellSize - Kích thước cell (từ Board component)
 * @param {number} padding - Padding của board (12px)
 * @param {number} boardSize - Kích thước board (9, 13, 19)
 * @returns {Array|null} Array of SVG elements hoặc null nếu không có data
 */
const renderMLVisualization = (mlAnalysisData, mlVisualizationMode, cellSize, padding, boardSize) => {
  // ========================================================================
  // VALIDATION: Kiểm tra input parameters
  // ========================================================================
  if (!mlAnalysisData) {
    console.warn('ML Visualization: No analysis data provided');
    return null;
  }
  
  if (!cellSize || cellSize <= 0) {
    console.warn('ML Visualization: Invalid cell size:', cellSize);
    return null;
  }
  
  if (!boardSize || boardSize <= 0) {
    console.warn('ML Visualization: Invalid board size:', boardSize);
    return null;
  }
  
  if (!['threats', 'attacks', 'intent'].includes(mlVisualizationMode)) {
    console.warn('ML Visualization: Invalid mode:', mlVisualizationMode);
    return null;
  }

  // ========================================================================
  // GET HEATMAP DATA: Lấy heatmap data theo mode
  // ========================================================================
  const getHeatmapData = () => {
    switch (mlVisualizationMode) {
      case 'threats':
        // Threats: Mối đe dọa - quân của mình bị đe dọa
        if (!mlAnalysisData.threats || !mlAnalysisData.threats.heatmap) {
          console.warn('ML Visualization: No threats heatmap data');
          return [];
        }
        return mlAnalysisData.threats.heatmap;
        
      case 'attacks':
        // Attacks: Cơ hội tấn công - nơi có thể tấn công đối thủ
        if (!mlAnalysisData.attacks || !mlAnalysisData.attacks.heatmap) {
          console.warn('ML Visualization: No attacks heatmap data');
          return [];
        }
        return mlAnalysisData.attacks.heatmap;
        
      case 'intent':
        // Intent: Ý định chiến lược - nơi AI muốn đánh
        if (!mlAnalysisData.intent || !mlAnalysisData.intent.heatmap) {
          console.warn('ML Visualization: No intent heatmap data');
          return [];
        }
        return mlAnalysisData.intent.heatmap;
        
      default:
        console.warn('ML Visualization: Unknown mode:', mlVisualizationMode);
        return [];
    }
  };

  // ========================================================================
  // GET REGIONS: Lấy regions data (threats/attacks regions)
  // ========================================================================
  const getRegions = () => {
    switch (mlVisualizationMode) {
      case 'threats':
        // Threats regions: Nhóm quân bị đe dọa
        if (!mlAnalysisData.threats || !mlAnalysisData.threats.regions) {
          return [];
        }
        return mlAnalysisData.threats.regions;
        
      case 'attacks':
        // Attacks opportunities: Cơ hội tấn công cụ thể
        if (!mlAnalysisData.attacks || !mlAnalysisData.attacks.opportunities) {
          return [];
        }
        // Convert opportunities format để render
        return (mlAnalysisData.attacks.opportunities || []).map(opp => ({
          ...opp,
          positions: opp.position ? [opp.position] : [],
          severity: opp.confidence || 0.5
        }));
        
      case 'intent':
        // Intent không có regions (chỉ có heatmap)
        return [];
        
      default:
        return [];
    }
  };

  // ========================================================================
  // GET SEVERITY COLOR: Màu sắc theo giá trị và mode
  // ========================================================================
  const getSeverityColor = (value) => {
    if (mlVisualizationMode === 'threats') {
      // Threats: Đỏ (nguy hiểm) -> Cam -> Vàng
      if (value > 0.8) return 'rgba(239, 68, 68, 0.6)';      // Critical: Đỏ đậm
      if (value > 0.5) return 'rgba(249, 115, 22, 0.5)';   // Moderate: Cam
      return 'rgba(234, 179, 8, 0.4)';                      // Low: Vàng
      
    } else if (mlVisualizationMode === 'attacks') {
      // Attacks: Xanh lá (tốt) -> Xanh dương -> Xanh nhạt
      if (value > 0.7) return 'rgba(34, 197, 94, 0.6)';    // High value: Xanh lá
      if (value > 0.4) return 'rgba(59, 130, 246, 0.5)';   // Medium: Xanh dương
      return 'rgba(147, 197, 253, 0.3)';                    // Low: Xanh nhạt
      
    } else {
      // Intent: Tím (ý định chiến lược)
      return 'rgba(168, 85, 247, 0.4)';
    }
  };

  // ========================================================================
  // GET DATA: Lấy heatmap và regions data
  // ========================================================================
  const heatmap = getHeatmapData();
  const regions = getRegions();
  
  // Validation: Kiểm tra heatmap có đúng format không
  if (!Array.isArray(heatmap) || heatmap.length === 0) {
    console.warn('ML Visualization: Invalid or empty heatmap data');
    return null;
  }
  
  // Validation: Kiểm tra heatmap có đúng kích thước không
  if (heatmap.length !== boardSize) {
    console.warn(`ML Visualization: Heatmap size mismatch. Expected ${boardSize}, got ${heatmap.length}`);
    return null;
  }
  
  const elements = [];

  // ========================================================================
  // RENDER HEATMAP: Vẽ heatmap overlay trên board
  // ========================================================================
  // Coordinate system:
  // - Backend trả về: heatmap[y][x] = heatmap[row][col]
  // - Board component: x = index % boardSize (col), y = Math.floor(index / boardSize) (row)
  // - Vị trí render: padding + colIndex * cellSize (khớp với cells)
  if (heatmap.length > 0) {
    // Threshold khác nhau cho từng mode để tối ưu visualization
    const threshold = mlVisualizationMode === 'intent' ? 0.1 : 0.2; // Intent cần threshold thấp hơn
    
    heatmap.forEach((row, rowIndex) => {
      // Validation: Kiểm tra row có đúng format không
      if (!Array.isArray(row) || row.length !== boardSize) {
        console.warn(`ML Visualization: Invalid row at index ${rowIndex}. Expected length ${boardSize}, got ${row.length}`);
        return; // Skip invalid row
      }
      
      row.forEach((value, colIndex) => {
        // Validation: Kiểm tra value có hợp lệ không
        if (typeof value !== 'number' || isNaN(value) || value < 0) {
          return; // Skip invalid values
        }
        
        // Chỉ render nếu value >= threshold
        if (value < threshold) return;
        
        // Lấy màu theo severity
        const color = getSeverityColor(value);
        
        // Tính vị trí chính xác: padding + colIndex * cellSize
        // Đảm bảo khớp với cách Board render cells
        const xPos = padding + colIndex * cellSize;
        const yPos = padding + rowIndex * cellSize;
        
        // Opacity: Intent cần opacity cao hơn để dễ nhìn
        const opacity = mlVisualizationMode === 'intent' 
          ? Math.min(value * 0.8, 0.6)  // Intent: opacity cao hơn
          : Math.min(value, 0.7);        // Threats/Attacks: opacity bình thường
        
        // Render rectangle cho heatmap cell
        elements.push(
          <rect
            key={`heat-${colIndex}-${rowIndex}`}
            x={xPos}
            y={yPos}
            width={cellSize}
            height={cellSize}
            fill={color}
            opacity={opacity}
          />
        );
      });
    });
  }

  // ========================================================================
  // RENDER REGIONS: Vẽ regions (threats boxes, attack markers)
  // ========================================================================
  if (Array.isArray(regions) && regions.length > 0) {
    regions.forEach((region, idx) => {
      // Validation: Kiểm tra region có hợp lệ không
      if (!region || typeof region !== 'object') {
        console.warn(`ML Visualization: Invalid region at index ${idx}`);
        return; // Skip invalid region
      }
      
      // ====================================================================
      // THREATS REGIONS: Vẽ box bao quanh nhóm quân bị đe dọa
      // ====================================================================
      if (mlVisualizationMode === 'threats' && region.positions && Array.isArray(region.positions) && region.positions.length > 0) {
        // Backend trả về positions là [x, y] = [col, row]
        const xs = region.positions.map(p => {
          if (Array.isArray(p) && p.length >= 2) return p[0];
          if (p && typeof p === 'object' && p.x !== undefined) return p.x;
          return 0;
        }).filter(x => typeof x === 'number' && x >= 0 && x < boardSize);
        
        const ys = region.positions.map(p => {
          if (Array.isArray(p) && p.length >= 2) return p[1];
          if (p && typeof p === 'object' && p.y !== undefined) return p.y;
          return 0;
        }).filter(y => typeof y === 'number' && y >= 0 && y < boardSize);
        
        // Validation: Kiểm tra có positions hợp lệ không
        if (xs.length === 0 || ys.length === 0) {
          console.warn(`ML Visualization: Invalid threat region positions at index ${idx}`);
          return; // Skip invalid region
        }
        
        const minX = Math.min(...xs);
        const maxX = Math.max(...xs);
        const minY = Math.min(...ys);
        const maxY = Math.max(...ys);
        
        // Tính bounding box: padding + minIndex * cellSize
        const boxX = padding + minX * cellSize - 3;  // -3 để có padding
        const boxY = padding + minY * cellSize - 3;
        const boxWidth = (maxX - minX + 1) * cellSize + 6;   // +6 để có padding cả 2 bên
        const boxHeight = (maxY - minY + 1) * cellSize + 6;
        
        // Màu theo severity
        const severity = typeof region.severity === 'number' ? region.severity : 0.5;
        const color = severity > 0.8 ? '#ef4444' : severity > 0.5 ? '#f97316' : '#eab308';
        
        // Render threat box với icon cảnh báo
        elements.push(
          <g key={`threat-${idx}`}>
            {/* Dashed box bao quanh nhóm quân bị đe dọa */}
            <rect
              x={boxX}
              y={boxY}
              width={boxWidth}
              height={boxHeight}
              stroke={color}
              strokeWidth={3}
              fill="none"
              strokeDasharray="5,5"
            />
            {/* Icon cảnh báo ở trên box */}
            <circle
              cx={boxX + boxWidth / 2}
              cy={boxY - 12}
              r={10}
              fill={color}
            />
            <text
              x={boxX + boxWidth / 2}
              y={boxY - 8}
              textAnchor="middle"
              fontSize={12}
              fill="white"
            >
              ⚠️
            </text>
          </g>
        );
        
      // ====================================================================
      // ATTACKS REGIONS: Vẽ marker cho cơ hội tấn công
      // ====================================================================
      } else if (mlVisualizationMode === 'attacks') {
        // Attacks có format: { position: [x, y], confidence: number }
        const position = region.position || (region.positions && Array.isArray(region.positions) && region.positions[0]);
        
        // Validation: Kiểm tra position có hợp lệ không
        if (!position || !Array.isArray(position) || position.length !== 2) {
          console.warn(`ML Visualization: Invalid attack position at index ${idx}`);
          return; // Skip invalid attack
        }
        
        const [x, y] = position; // [col, row]
        
        // Validation: Kiểm tra coordinates có trong board không
        if (typeof x !== 'number' || typeof y !== 'number' || 
            x < 0 || x >= boardSize || y < 0 || y >= boardSize) {
          console.warn(`ML Visualization: Attack position out of bounds at index ${idx}: [${x}, ${y}]`);
          return; // Skip invalid position
        }
        
        // Tính center của cell: padding + index * cellSize + cellSize/2
        const centerX = padding + x * cellSize + cellSize / 2;
        const centerY = padding + y * cellSize + cellSize / 2;
        
        // Màu theo confidence
        const confidence = typeof region.confidence === 'number' ? region.confidence : 0.5;
        const color = confidence > 0.7 ? '#22c55e' : '#3b82f6';
        
        // Render attack marker (circle + arrow)
        elements.push(
          <g key={`attack-${idx}`}>
            {/* Circle background */}
            <circle
              cx={centerX}
              cy={centerY}
              r={cellSize * 0.4}
              fill={color}
              opacity={0.3}
            />
            {/* Arrow pointing down (attack direction) */}
            <polygon
              points={`${centerX},${centerY - cellSize * 0.3} ${centerX - cellSize * 0.2},${centerY + cellSize * 0.2} ${centerX + cellSize * 0.2},${centerY + cellSize * 0.2}`}
              fill={color}
            />
          </g>
        );
      }
      // Intent không có regions (chỉ có heatmap)
    });
  }

  // ========================================================================
  // RETURN: Trả về array of SVG elements
  // ========================================================================
  // Nếu không có elements nào, trả về null (không render gì)
  if (elements.length === 0) {
    console.warn('ML Visualization: No elements to render');
    return null;
  }
  
  return elements;
};

/**
 * ============================================================================
 * HINT VISUALIZATION HELPER FUNCTION
 * ============================================================================
 * Render hints visualization trên board - hiển thị các nước đi gợi ý
 */
const renderHintVisualization = (hints, cellSize, padding, boardSize) => {
  console.log('renderHintVisualization called:', { hints, cellSize, padding, boardSize });
  
  if (!hints || !Array.isArray(hints) || hints.length === 0) {
    console.warn('renderHintVisualization: Invalid hints', hints);
    return null;
  }
  
  if (!cellSize || cellSize <= 0 || !boardSize || boardSize <= 0) {
    console.warn('renderHintVisualization: Invalid dimensions', { cellSize, boardSize });
    return null;
  }
  
  const elements = [];
  
  hints.forEach((hint, index) => {
    // Hints có format: { move: [x, y], confidence: number, is_pass: boolean }
    const move = hint.move;
    
    if (!move || hint.is_pass || !Array.isArray(move) || move.length !== 2) {
      console.log(`Skipping hint ${index}:`, { move, is_pass: hint.is_pass });
      return;
    }
    
    const [x, y] = move;
    if (x < 0 || x >= boardSize || y < 0 || y >= boardSize) {
      return;
    }
    
    const confidence = hint.confidence || 0;
    const centerX = padding + x * cellSize + cellSize / 2;
    const centerY = padding + y * cellSize + cellSize / 2;
    
    // Màu theo confidence
    let bgColor = '#fbbf24'; // Vàng mặc định
    if (confidence >= 0.7) {
      bgColor = '#10b981'; // Xanh lá - tốt
    } else if (confidence >= 0.5) {
      bgColor = '#3b82f6'; // Xanh dương - trung bình
    } else {
      bgColor = '#f59e0b'; // Cam - thấp
    }
    
    elements.push(
      <g key={`hint-${index}`}>
        <circle
          cx={centerX}
          cy={centerY}
          r={cellSize * 0.35}
          fill={bgColor}
          fillOpacity={0.9}
          stroke="white"
          strokeWidth={2}
        />
        <text
          x={centerX}
          y={centerY}
          textAnchor="middle"
          dominantBaseline="middle"
          fill="white"
          fontSize={cellSize * 0.25}
          fontWeight="bold"
        >
          {index + 1}
        </text>
        <text
          x={centerX}
          y={centerY + cellSize * 0.5}
          textAnchor="middle"
          dominantBaseline="top"
          fill={bgColor}
          fontSize={cellSize * 0.15}
          fontWeight="600"
        >
          {Math.round(confidence * 100)}%
        </text>
      </g>
    );
  });
  
  return elements.length > 0 ? elements : null;
};

/**
 * ============================================================================
 * REVIEW VISUALIZATION HELPER FUNCTION
 * ============================================================================
 * Render review visualization trên board - hiển thị mistakes và key moments
 */
const renderReviewVisualization = (reviewData, cellSize, padding, boardSize) => {
  console.log('renderReviewVisualization called:', { reviewData, cellSize, padding, boardSize });
  
  if (!reviewData) {
    console.warn('renderReviewVisualization: No reviewData');
    return null;
  }
  
  if (!cellSize || cellSize <= 0 || !boardSize || boardSize <= 0) {
    console.warn('renderReviewVisualization: Invalid dimensions', { cellSize, boardSize });
    return null;
  }
  
  const details = reviewData.details || reviewData;
  if (!details || typeof details !== 'object') {
    return null;
  }
  
  const mistakes = details.mistakes || [];
  const key_moments = details.key_moments || [];
  
  const elements = [];
  
  // Render mistakes (màu đỏ/cam)
  mistakes.forEach((mistake, index) => {
    if (!mistake.position || !Array.isArray(mistake.position)) {
      return;
    }
    
    const [x, y] = mistake.position;
    if (x < 0 || x >= boardSize || y < 0 || y >= boardSize) {
      return;
    }
    
    const centerX = padding + x * cellSize + cellSize / 2;
    const centerY = padding + y * cellSize + cellSize / 2;
    const severity = mistake.severity === 'major' ? '#ef4444' : '#f59e0b';
    
    elements.push(
      <g key={`mistake-${index}`}>
        <circle
          cx={centerX}
          cy={centerY}
          r={cellSize * 0.3}
          fill={severity}
          fillOpacity={0.7}
          stroke="white"
          strokeWidth={2}
        />
        <text
          x={centerX}
          y={centerY}
          textAnchor="middle"
          dominantBaseline="middle"
          fill="white"
          fontSize={cellSize * 0.2}
          fontWeight="bold"
        >
          ⚠
        </text>
        <text
          x={centerX}
          y={centerY + cellSize * 0.5}
          textAnchor="middle"
          dominantBaseline="top"
          fill={severity}
          fontSize={cellSize * 0.12}
        >
          #{mistake.move_number}
        </text>
      </g>
    );
  });
  
  // Render key moments (màu xanh/đỏ)
  key_moments.forEach((moment, index) => {
    if (!moment.position || !Array.isArray(moment.position)) {
      return;
    }
    
    const [x, y] = moment.position;
    if (x < 0 || x >= boardSize || y < 0 || y >= boardSize) {
      return;
    }
    
    const centerX = padding + x * cellSize + cellSize / 2;
    const centerY = padding + y * cellSize + cellSize / 2;
    const color = moment.type === 'advantage_gain' ? '#10b981' : '#ef4444';
    
    elements.push(
      <g key={`moment-${index}`}>
        <circle
          cx={centerX}
          cy={centerY}
          r={cellSize * 0.25}
          fill={color}
          fillOpacity={0.6}
          stroke="white"
          strokeWidth={2}
          strokeDasharray="3,3"
        />
        <text
          x={centerX}
          y={centerY}
          textAnchor="middle"
          dominantBaseline="middle"
          fill="white"
          fontSize={cellSize * 0.18}
          fontWeight="bold"
        >
          {moment.type === 'advantage_gain' ? '↑' : '↓'}
        </text>
        <text
          x={centerX}
          y={centerY + cellSize * 0.5}
          textAnchor="middle"
          dominantBaseline="top"
          fill={color}
          fontSize={cellSize * 0.12}
        >
          #{moment.move_number}
        </text>
      </g>
    );
  });
  
  return elements.length > 0 ? elements : null;
};

// NOTE: In Go, stones are placed on INTERSECTIONS (giao điểm), not in squares.
// Each "cell" in this component represents an intersection point on the board.
const Board = ({ boardSize = 9, stones = {}, onCellClick, lastMove = null, disabled = false, theme = 'classic', mlAnalysisData = null, mlVisualizationMode = 'threats', hints = null, reviewData = null, showHintsViz = false, showReviewViz = false }) => {
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
            {/* ML Analysis Visualization Layer - render trực tiếp trong Board để đảm bảo khớp chính xác */}
            {mlAnalysisData && actualCellSize > 0 && renderMLVisualization(mlAnalysisData, mlVisualizationMode, actualCellSize, padding, boardSize)}
            {/* Hint Visualization Layer */}
            {showHintsViz && hints && Array.isArray(hints) && hints.length > 0 && actualCellSize > 0 && renderHintVisualization(hints, actualCellSize, padding, boardSize)}
            {/* Review Visualization Layer */}
            {showReviewViz && reviewData && actualCellSize > 0 && renderReviewVisualization(reviewData, actualCellSize, padding, boardSize)}
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


