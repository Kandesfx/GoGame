import React, { useState, useEffect } from 'react';
import { FaChevronLeft, FaChevronRight, FaTimes, FaCheck, FaExclamationTriangle } from 'react-icons/fa';
import './InteractiveTutorial.css';

// Component bàn cờ 9x9 tương tác
const InteractiveBoard = ({ boardSize = 9, stones, onCellClick, disabled, highlightCells = [], errorMessage = null }) => {
  const [hoverPos, setHoverPos] = useState(null);
  const cellSize = 35;
  const padding = 15;
  const boardSizePx = cellSize * (boardSize - 1) + padding * 2;

  const handleCellClick = (x, y) => {
    if (disabled) return;
    const key = `${x},${y}`;
    if (!stones[key] && onCellClick) {
      onCellClick(x, y);
    }
  };

  const handleCellHover = (x, y) => {
    if (disabled) return;
    const key = `${x},${y}`;
    if (!stones[key]) {
      setHoverPos({ x, y });
    } else {
      setHoverPos(null);
    }
  };

  const handleCellLeave = () => {
    setHoverPos(null);
  };

  const isHighlighted = (x, y) => {
    return highlightCells.some(cell => cell.x === x && cell.y === y);
  };

  return (
    <div className="interactive-board-container">
      {errorMessage && (
        <div className="tutorial-error-message">
          <FaExclamationTriangle /> {errorMessage}
        </div>
      )}
      <svg width={boardSizePx} height={boardSizePx} className="interactive-board-svg">
        {/* Nền gỗ */}
        <rect width={boardSizePx} height={boardSizePx} fill="#DEB887" />
        
        {/* Lưới bàn cờ */}
        {Array.from({ length: boardSize }).map((_, i) => (
          <g key={`grid-${i}`}>
            <line
              x1={padding}
              y1={padding + i * cellSize}
              x2={boardSizePx - padding}
              y2={padding + i * cellSize}
              stroke="#8B4513"
              strokeWidth="1.5"
            />
            <line
              x1={padding + i * cellSize}
              y1={padding}
              x2={padding + i * cellSize}
              y2={boardSizePx - padding}
              stroke="#8B4513"
              strokeWidth="1.5"
            />
          </g>
        ))}

        {/* Các điểm sao (star points) cho bàn 9x9 */}
        {[
          [2, 2], [2, 6], [6, 2], [6, 6], [4, 4]
        ].map(([x, y], i) => (
          <circle
            key={`star-${i}`}
            cx={padding + x * cellSize}
            cy={padding + y * cellSize}
            r="3"
            fill="#8B4513"
          />
        ))}

        {/* Highlight cells - Vị trí cần đánh */}
        {highlightCells.map((cell, i) => {
          const cx = padding + cell.x * cellSize;
          const cy = padding + cell.y * cellSize;
          return (
            <g key={`highlight-${i}`}>
              {/* Vòng tròn highlight với animation pulse */}
              <circle
                cx={cx}
                cy={cy}
                r="16"
                fill="rgba(255, 215, 0, 0.4)"
                stroke="rgba(255, 165, 0, 1)"
                strokeWidth="3"
                className="highlight-pulse"
              />
              {/* Vòng tròn ngoài với animation rotate */}
              <circle
                cx={cx}
                cy={cy}
                r="20"
                fill="none"
                stroke="rgba(255, 215, 0, 0.6)"
                strokeWidth="2"
                strokeDasharray="4,4"
                className="highlight-rotate"
              />
              {/* Icon mũi tên chỉ vào với animation bounce */}
              <text
                x={cx}
                y={cy - 25}
                fontSize="20"
                fill="rgba(255, 140, 0, 1)"
                textAnchor="middle"
                dominantBaseline="middle"
                fontWeight="bold"
                className="highlight-bounce"
              >
                ↓
              </text>
              {/* Vòng tròn nhấn mạnh */}
              <circle
                cx={cx}
                cy={cy}
                r="8"
                fill="rgba(255, 215, 0, 0.8)"
                stroke="rgba(255, 140, 0, 1)"
                strokeWidth="2"
              />
            </g>
          );
        })}

        {/* Hover indicator */}
        {hoverPos && !stones[`${hoverPos.x},${hoverPos.y}`] && (
          <circle
            cx={padding + hoverPos.x * cellSize}
            cy={padding + hoverPos.y * cellSize}
            r="13"
            fill="rgba(0, 0, 0, 0.1)"
            stroke="rgba(0, 0, 0, 0.3)"
            strokeWidth="2"
            strokeDasharray="3,3"
          />
        )}

        {/* Các quân cờ */}
        {Object.entries(stones).map(([key, color]) => {
          const [x, y] = key.split(',').map(Number);
          const cx = padding + x * cellSize;
          const cy = padding + y * cellSize;
          return (
            <g key={`stone-${key}`}>
              <circle
                cx={cx}
                cy={cy}
                r="14"
                fill={color === 'black' ? '#000' : '#fff'}
                stroke={color === 'black' ? '#333' : '#ccc'}
                strokeWidth="1.5"
              />
              {color === 'white' && (
                <circle
                  cx={cx}
                  cy={cy}
                  r="14"
                  fill="none"
                  stroke="#ddd"
                  strokeWidth="1"
                />
              )}
            </g>
          );
        })}

        {/* Clickable cells */}
        {Array.from({ length: boardSize }).map((_, y) =>
          Array.from({ length: boardSize }).map((_, x) => (
            <rect
              key={`cell-${x}-${y}`}
              x={padding + x * cellSize - cellSize / 2}
              y={padding + y * cellSize - cellSize / 2}
              width={cellSize}
              height={cellSize}
              fill="transparent"
              style={{ cursor: disabled ? 'not-allowed' : 'pointer' }}
              onClick={() => handleCellClick(x, y)}
              onMouseEnter={() => handleCellHover(x, y)}
              onMouseLeave={handleCellLeave}
            />
          ))
        )}
      </svg>
    </div>
  );
};

// Hàm kiểm tra khí của một nhóm quân
const getGroupLiberties = (stones, x, y, boardSize, visited = new Set()) => {
  const key = `${x},${y}`;
  if (visited.has(key) || x < 0 || x >= boardSize || y < 0 || y >= boardSize) {
    return [];
  }
  
  const color = stones[key];
  if (!color) {
    return [{ x, y }];
  }
  
  visited.add(key);
  const liberties = [];
  
  const neighbors = [
    { x: x - 1, y },
    { x: x + 1, y },
    { x, y: y - 1 },
    { x, y: y + 1 }
  ];
  
  for (const neighbor of neighbors) {
    const neighborKey = `${neighbor.x},${neighbor.y}`;
    if (!visited.has(neighborKey)) {
      if (!stones[neighborKey]) {
        liberties.push({ x: neighbor.x, y: neighbor.y });
      } else if (stones[neighborKey] === color) {
        liberties.push(...getGroupLiberties(stones, neighbor.x, neighbor.y, boardSize, visited));
      }
    }
  }
  
  return liberties;
};

// Hàm lấy tất cả các quân trong một nhóm
const getGroup = (stones, x, y, boardSize, visited = new Set()) => {
  const key = `${x},${y}`;
  if (visited.has(key) || x < 0 || x >= boardSize || y < 0 || y >= boardSize) {
    return [];
  }
  
  const color = stones[key];
  if (!color) {
    return [];
  }
  
  visited.add(key);
  const group = [{ x, y }];
  
  const neighbors = [
    { x: x - 1, y },
    { x: x + 1, y },
    { x, y: y - 1 },
    { x, y: y + 1 }
  ];
  
  for (const neighbor of neighbors) {
    const neighborKey = `${neighbor.x},${neighbor.y}`;
    if (!visited.has(neighborKey) && stones[neighborKey] === color) {
      group.push(...getGroup(stones, neighbor.x, neighbor.y, boardSize, visited));
    }
  }
  
  return group;
};

// Hàm kiểm tra và xóa các nhóm quân bị ăn
const removeCapturedGroups = (stones, x, y, color, boardSize) => {
  const newStones = { ...stones, [`${x},${y}`]: color };
  const opponentColor = color === 'black' ? 'white' : 'black';
  const captured = [];
  
  // Kiểm tra các nhóm đối phương xung quanh
  const neighbors = [
    { x: x - 1, y },
    { x: x + 1, y },
    { x, y: y - 1 },
    { x, y: y + 1 }
  ];
  
  for (const neighbor of neighbors) {
    const neighborKey = `${neighbor.x},${neighbor.y}`;
    if (newStones[neighborKey] === opponentColor) {
      const liberties = getGroupLiberties(newStones, neighbor.x, neighbor.y, boardSize);
      if (liberties.length === 0) {
        // Nhóm này bị ăn
        const group = getGroup(newStones, neighbor.x, neighbor.y, boardSize);
        captured.push(...group);
      }
    }
  }
  
  // Xóa các quân bị ăn
  for (const stone of captured) {
    const key = `${stone.x},${stone.y}`;
    delete newStones[key];
  }
  
  return { newStones, capturedCount: captured.length };
};

// Hàm kiểm tra nước đi có hợp lệ không
const isValidMove = (stones, x, y, color, boardSize) => {
  // Kiểm tra vị trí có trống không
  const key = `${x},${y}`;
  if (stones[key]) {
    return { valid: false, message: 'Vị trí này đã có quân cờ!' };
  }
  
  // Kiểm tra biên
  if (x < 0 || x >= boardSize || y < 0 || y >= boardSize) {
    return { valid: false, message: 'Nước đi ngoài bàn cờ!' };
  }
  
  // Tạo bản sao để kiểm tra
  const testStones = { ...stones, [key]: color };
  
  // Kiểm tra xem có ăn được quân đối phương không
  const { newStones: afterCapture } = removeCapturedGroups(stones, x, y, color, boardSize);
  
  // Kiểm tra khí của quân vừa đặt sau khi ăn
  const liberties = getGroupLiberties(afterCapture, x, y, boardSize);
  if (liberties.length === 0) {
    return { valid: false, message: 'Quân cờ phải có ít nhất 1 khí!' };
  }
  
  return { valid: true, message: null };
};

const InteractiveTutorial = ({ isOpen, onClose }) => {
  const [currentLesson, setCurrentLesson] = useState(0);
  const [stones, setStones] = useState({});
  const [currentPlayer, setCurrentPlayer] = useState('black');
  const [errorMessage, setErrorMessage] = useState(null);
  const [highlightCells, setHighlightCells] = useState([]);
  const [lessonCompleted, setLessonCompleted] = useState(false);
  const [boardHistory, setBoardHistory] = useState([]); // Lưu lịch sử để phát hiện KO
  const [koDetected, setKoDetected] = useState(false);

  const lessons = [
    {
      title: "Quân đen đi trước, quân trắng đi sau",
      description: "Trong cờ vây, quân đen luôn đi trước. Hãy đặt quân đen đầu tiên vào bàn cờ.",
      instruction: "Click vào một giao điểm trên bàn cờ để đặt quân đen.",
      goal: (stones, currentPlayer) => {
        const blackCount = Object.values(stones).filter(c => c === 'black').length;
        return blackCount >= 1;
      },
      initialStones: {},
      initialPlayer: 'black',
      highlightCells: []
    },
    {
      title: "Quân cờ phải có khí",
      description: "Mỗi quân cờ cần có 'khí' (các điểm trống liền kề) để tồn tại. Hãy đặt quân trắng vào một vị trí có khí.",
      instruction: "Đặt quân trắng vào một vị trí có ít nhất 1 điểm trống liền kề.",
      goal: (stones, currentPlayer) => {
        const whiteCount = Object.values(stones).filter(c => c === 'white').length;
        return whiteCount >= 1;
      },
      initialStones: { '4,4': 'black' },
      initialPlayer: 'white',
      highlightCells: [
        { x: 3, y: 4 }, { x: 5, y: 4 }, { x: 4, y: 3 }, { x: 4, y: 5 }
      ]
    },
    {
      title: "Cách ăn quân",
      description: "Khi bao vây quân địch và lấy hết khí của chúng, bạn có thể ăn quân. Ở đây, quân đen đã bị bao vây 3 phía, chỉ còn 1 khí. Hãy đặt quân trắng vào vị trí còn lại để ăn quân đen.",
      instruction: "Đặt quân trắng vào vị trí được đánh dấu (khí cuối cùng) để ăn quân đen.",
      goal: (stones, currentPlayer) => {
        // Kiểm tra xem không còn quân đen nào trên bàn cờ (đã bị ăn)
        const blackStones = Object.values(stones).filter(c => c === 'black');
        return blackStones.length === 0;
      },
      initialStones: { 
        '4,4': 'black',  // Quân đen ở giữa
        '3,4': 'white',  // Đã bao vây bên trái
        '5,4': 'white',  // Đã bao vây bên phải
        '4,3': 'white'   // Đã bao vây bên trên
      },
      initialPlayer: 'white',
      highlightCells: [
        { x: 4, y: 5 }  // Chỉ còn 1 khí ở dưới
      ]
    },
    {
      title: "Luật cướp cờ (KO)",
      description: "Luật KO ngăn việc lặp lại vô tận cùng một nước đi. Ở đây, quân trắng có thể ăn quân đen. Sau đó, nếu đen muốn ăn lại ngay ở cùng vị trí sẽ tạo ra KO. Hãy đánh vài nước để hiểu luật này.",
      instruction: "Đặt quân trắng vào vị trí được đánh dấu để ăn quân đen. Sau đó đen sẽ đánh lại ở cùng vị trí - đây chính là tình huống KO!",
      goal: (stones, currentPlayer, koDetected) => {
        // Hoàn thành khi phát hiện KO
        return koDetected;
      },
      initialStones: {
        // Setup theo tọa độ từ GoTutorial.jsx
        '4,2': 'black',
        '5,2': 'white',
        '3,3': 'black',
        '5,3': 'black',
        '6,3': 'white',
        '4,4': 'black',
        '5,4': 'white'
        // Vị trí (4,3) trống - đây là nơi có thể tạo KO
      },
      initialPlayer: 'white',
      highlightCells: [
        { x: 4, y: 3 }  // Vị trí có thể ăn quân đen (4,4) - đánh ở đây để tạo tình huống KO
      ]
    }
  ];

  useEffect(() => {
    if (isOpen) {
      resetLesson();
    }
  }, [isOpen, currentLesson]);

  const resetLesson = () => {
    const lesson = lessons[currentLesson];
    setStones(lesson.initialStones || {});
    setCurrentPlayer(lesson.initialPlayer || 'black');
    setErrorMessage(null);
    setHighlightCells(lesson.highlightCells || []);
    setLessonCompleted(false);
    setBoardHistory([]);
    setKoDetected(false);
  };

  // Hàm chuyển đổi stones thành string để so sánh
  // Sắp xếp keys để đảm bảo so sánh chính xác
  const stonesToString = (stones) => {
    const sortedKeys = Object.keys(stones).sort();
    const sortedStones = {};
    sortedKeys.forEach(key => {
      sortedStones[key] = stones[key];
    });
    return JSON.stringify(sortedStones);
  };

  // Hàm kiểm tra KO - tình huống KO xảy ra khi trạng thái bàn cờ lặp lại sau 2 nước
  // Phát hiện ngay khi lặp lại lần thứ 2 (nước thứ 2 giống nước thứ 0)
  const checkKo = (newStones, history) => {
    const currentState = stonesToString(newStones);
    // KO pattern: state0 -> state1 -> state2 (nếu state2 === state0, đây là KO)
    // history chứa các trạng thái TRƯỚC khi đánh nước hiện tại
    // Ví dụ: 
    //   - Nước 1: stones = state0, newStones = state1, history = [state0]
    //   - Nước 2: stones = state1, newStones = state2, history = [state0, state1]
    //   - Nếu state2 === state0, đây là KO
    
    // Kiểm tra với tất cả các state trong history (trừ state ngay trước)
    // Vì state ngay trước là state trước khi đánh nước hiện tại
    if (history.length >= 2) {
      // Kiểm tra với state cách đó 2 nước (đây là pattern KO điển hình)
      const twoMovesAgo = history[history.length - 2];
      if (currentState === twoMovesAgo) {
        console.log('KO detected: currentState matches two moves ago');
        return true; // Phát hiện KO
      }
    }
    
    // Kiểm tra với tất cả các state trước đó (trừ state ngay trước) để đảm bảo
    if (history.length >= 1) {
      for (let i = 0; i < history.length - 1; i++) {
        if (currentState === history[i]) {
          console.log(`KO detected: currentState matches history[${i}]`);
          return true; // Phát hiện lặp lại - đây là KO
        }
      }
    }
    
    return false;
  };

  const handleCellClick = (x, y) => {
    const lesson = lessons[currentLesson];
    const validation = isValidMove(stones, x, y, currentPlayer, 9);
    
    if (!validation.valid) {
      setErrorMessage(validation.message);
      setTimeout(() => setErrorMessage(null), 3000);
      return;
    }

    setErrorMessage(null);
    
    // Đặt quân và xóa các nhóm bị ăn
    const { newStones, capturedCount } = removeCapturedGroups(stones, x, y, currentPlayer, 9);
    
    // Lưu trạng thái TRƯỚC khi đánh vào lịch sử (để so sánh với trạng thái SAU khi đánh)
    const newHistory = [...boardHistory, stonesToString(stones)];
    
    // Kiểm tra KO (chỉ cho bài học KO) - phải kiểm tra TRƯỚC khi update state
    if (currentLesson === 3) {
      console.log('Checking KO:', {
        currentState: stonesToString(newStones),
        history: newHistory,
        historyLength: newHistory.length
      });
      const isKo = checkKo(newStones, newHistory);
      if (isKo) {
        console.log('KO DETECTED!');
        setKoDetected(true);
        setErrorMessage(null);
        setLessonCompleted(true);
        setBoardHistory(newHistory); // Update history
        setStones(newStones); // Update stones
        return; // Dừng lại, không đổi lượt
      }
    }
    
    // Update history và stones
    setBoardHistory(newHistory);
    
    setStones(newStones);

    // Kiểm tra xem đã hoàn thành bài học chưa
    if (lesson.goal(newStones, currentPlayer, koDetected || (currentLesson === 3 && checkKo(newStones, newHistory)))) {
      setLessonCompleted(true);
    } else {
      // Đổi lượt chơi
      setCurrentPlayer(currentPlayer === 'black' ? 'white' : 'black');
    }
  };

  const nextLesson = () => {
    if (currentLesson < lessons.length - 1) {
      setCurrentLesson(currentLesson + 1);
    } else {
      // Hoàn thành tất cả bài học
      onClose();
    }
  };

  const prevLesson = () => {
    if (currentLesson > 0) {
      setCurrentLesson(currentLesson - 1);
    }
  };

  const currentLessonData = lessons[currentLesson];

  if (!isOpen) {
    return null;
  }

  return (
    <div className="interactive-tutorial-overlay" onClick={onClose}>
      <div className="interactive-tutorial-dialog" onClick={(e) => e.stopPropagation()}>
        {/* Header */}
        <div className="interactive-tutorial-header">
          <h2 className="interactive-tutorial-title">Hướng Dẫn Chơi Cờ Vây - Bàn Cờ Ảo</h2>
          <button
            onClick={onClose}
            className="interactive-tutorial-close-btn"
          >
            <FaTimes size={24} />
          </button>
        </div>

        <p className="interactive-tutorial-step-indicator">
          Bài học {currentLesson + 1} / {lessons.length}
        </p>

        {/* Progress Bar */}
        <div className="interactive-tutorial-progress-bar">
          <div
            className="interactive-tutorial-progress-fill"
            style={{ width: `${((currentLesson + 1) / lessons.length) * 100}%` }}
          />
        </div>

        {/* Content */}
        <div className="interactive-tutorial-content">
          <h3 className="interactive-tutorial-step-title">
            {currentLessonData.title}
          </h3>
          
          {/* Main layout: Bàn cờ bên trái, mô tả bên phải */}
          <div className="interactive-tutorial-main-layout">
            {/* Bàn cờ tương tác - Bên trái */}
            <div className="interactive-tutorial-board-wrapper">
              <InteractiveBoard
                boardSize={9}
                stones={stones}
                onCellClick={handleCellClick}
                disabled={lessonCompleted}
                highlightCells={highlightCells}
                errorMessage={errorMessage}
              />
              
              <div className="interactive-tutorial-player-info">
                <div className={`player-indicator ${currentPlayer}`}>
                  <div className="player-stone-preview"></div>
                  <span>Lượt: {currentPlayer === 'black' ? 'Đen' : 'Trắng'}</span>
                </div>
              </div>
            </div>

            {/* Mô tả - Bên phải */}
            <div className="interactive-tutorial-text-content">
              <p className="interactive-tutorial-description">
                {currentLessonData.description}
              </p>

              <div className="interactive-tutorial-instruction">
                <FaCheck className="instruction-icon" />
                <span>{currentLessonData.instruction}</span>
              </div>

              {koDetected && currentLesson === 3 && (
                <div className="interactive-tutorial-ko-notice">
                  <FaExclamationTriangle className="ko-icon" />
                  <div>
                    <strong>Tình huống KO đã được phát hiện!</strong>
                    <p>Bạn đã tạo ra tình huống cướp cờ (KO). Luật KO ngăn việc lặp lại vô tận cùng một nước đi. Sau khi bị ăn, bạn không thể ngay lập tức ăn lại ở cùng vị trí mà phải đánh ở chỗ khác trước.</p>
                  </div>
                </div>
              )}

              {lessonCompleted && !koDetected && (
                <div className="interactive-tutorial-completion">
                  <FaCheck className="completion-icon" />
                  <span>Hoàn thành! Bạn đã hiểu bài học này.</span>
                </div>
              )}
            </div>
          </div>
        </div>

        {/* Navigation */}
        <div className="interactive-tutorial-navigation">
          <button
            onClick={prevLesson}
            disabled={currentLesson === 0}
            className={`interactive-tutorial-nav-btn interactive-tutorial-nav-prev ${
              currentLesson === 0 ? 'disabled' : ''
            }`}
          >
            <FaChevronLeft size={20} />
            Trước
          </button>

          <button
            onClick={resetLesson}
            className="interactive-tutorial-reset-btn"
            title="Làm lại bài học này"
          >
            Làm lại
          </button>

          <button
            onClick={nextLesson}
            disabled={!lessonCompleted && currentLesson < lessons.length - 1}
            className={`interactive-tutorial-nav-btn interactive-tutorial-nav-next ${
              !lessonCompleted && currentLesson < lessons.length - 1 ? 'disabled' : ''
            }`}
          >
            {currentLesson === lessons.length - 1 ? 'Hoàn thành' : 'Tiếp'}
            {currentLesson < lessons.length - 1 && <FaChevronRight size={20} />}
          </button>
        </div>
      </div>
    </div>
  );
};

export default InteractiveTutorial;

