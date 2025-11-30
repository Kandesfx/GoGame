import React, { useState } from 'react';
import { FaChevronLeft, FaChevronRight, FaTimes } from 'react-icons/fa';
import './GoTutorial.css';

// Component vẽ bàn cờ với các quân cờ
const GoBoard = ({ stones, marks, size = 9 }) => {
  const cellSize = 28; // Giảm 30% từ 40
  const padding = 14; // Giảm 30% từ 20
  const boardSize = cellSize * (size - 1) + padding * 2;

  return (
    <svg width={boardSize} height={boardSize} className="mx-auto">
      {/* Nền gỗ */}
      <rect width={boardSize} height={boardSize} fill="#DEB887" />
      
      {/* Lưới bàn cờ */}
      {Array.from({ length: size }).map((_, i) => (
        <g key={`grid-${i}`}>
          <line
            x1={padding}
            y1={padding + i * cellSize}
            x2={boardSize - padding}
            y2={padding + i * cellSize}
            stroke="#8B4513"
            strokeWidth="1"
          />
          <line
            x1={padding + i * cellSize}
            y1={padding}
            x2={padding + i * cellSize}
            y2={boardSize - padding}
            stroke="#8B4513"
            strokeWidth="1"
          />
        </g>
      ))}

      {/* Các điểm sao (star points) */}
      {size === 9 && [
        [2, 2], [2, 6], [6, 2], [6, 6], [4, 4]
      ].map(([x, y], i) => (
        <circle
          key={`star-${i}`}
          cx={padding + x * cellSize}
          cy={padding + y * cellSize}
          r="2.1"
          fill="#8B4513"
        />
      ))}

      {/* Các quân cờ */}
      {stones?.map((stone, i) => {
        const x = padding + stone.x * cellSize;
        const y = padding + stone.y * cellSize;
        return (
          <g key={`stone-${i}`}>
            <circle
              cx={x}
              cy={y}
              r="11.2"
              fill={stone.color === 'black' ? '#000' : '#fff'}
              stroke={stone.color === 'black' ? '#333' : '#ccc'}
              strokeWidth="1"
            />
            {stone.color === 'white' && (
              <circle
                cx={x}
                cy={y}
                r="11.2"
                fill="none"
                stroke="#ddd"
                strokeWidth="1"
              />
            )}
          </g>
        );
      })}

      {/* Các ký hiệu đánh dấu */}
      {marks?.map((mark, i) => {
        const x = padding + mark.x * cellSize;
        const y = padding + mark.y * cellSize;
        
        if (mark.type === 'x') {
          return (
            <g key={`mark-${i}`}>
              <line x1={x-5.6} y1={y-5.6} x2={x+5.6} y2={y+5.6} stroke="#000" strokeWidth="1.4" />
              <line x1={x-5.6} y1={y+5.6} x2={x+5.6} y2={y-5.6} stroke="#000" strokeWidth="1.4" />
            </g>
          );
        }
        
        if (mark.type === 'triangle') {
          return (
            <g key={`mark-${i}`}>
              <polygon
                points={`${x},${y-7} ${x-6.3},${y+4.9} ${x+6.3},${y+4.9}`}
                fill="none"
                stroke="#000"
                strokeWidth="1.4"
              />
            </g>
          );
        }
        
        if (mark.type === 'number') {
          return (
            <text
              key={`mark-${i}`}
              x={x}
              y={y}
              fontSize="11.2"
              fontWeight="bold"
              textAnchor="middle"
              dominantBaseline="middle"
              fill="#000"
            >
              {mark.label}
            </text>
          );
        }
        
        if (mark.type === 'letter') {
          return (
            <text
              key={`mark-${i}`}
              x={x}
              y={y}
              fontSize="12.6"
              fontWeight="bold"
              textAnchor="middle"
              dominantBaseline="middle"
              fill="#fff"
            >
              {mark.label}
            </text>
          );
        }
        
        return null;
      })}
    </svg>
  );
};

const GoTutorial = ({ isOpen, onClose }) => {
  const [currentStep, setCurrentStep] = useState(0);

  const tutorialSteps = [
    {
      title: "Cách Vận Hành Trò Chơi",
      description: "Đen luôn đánh trước và Trắng luôn đánh sau. Hai người chơi lần lượt đặt quân cờ lên các giao điểm của bàn cờ.",
      board: {
        stones: [
          { x: 3, y: 4, color: 'black' },
          { x: 5, y: 4, color: 'white' }
        ],
        marks: []
      }
    },
    {
      title: "Khí Của Quân Cờ - Phần 1",
      description: "Mỗi quân cờ cần có 'khí' để tồn tại. Khí là các điểm trống liền kề theo chiều ngang hoặc dọc. Các dấu X đánh dấu vị trí khí của quân cờ.",
      board: {
        stones: [
          { x: 4, y: 4, color: 'black' }
        ],
        marks: [
          { x: 3, y: 4, type: 'x' },
          { x: 5, y: 4, type: 'x' },
          { x: 4, y: 3, type: 'x' },
          { x: 4, y: 5, type: 'x' }
        ]
      }
    },
    {
      title: "Khí Của Quân Cờ - Phần 2",
      description: "Các quân cờ cùng màu nằm liền kề nhau tạo thành một nhóm và chia sẻ chung khí.",
      board: {
        stones: [
          { x: 3, y: 3, color: 'white' },
          { x: 4, y: 3, color: 'white' },
          { x: 5, y: 3, color: 'white' },
          { x: 3, y: 4, color: 'white' },
          { x: 4, y: 4, color: 'black' },
          { x: 5, y: 4, color: 'white' },
          { x: 3, y: 5, color: 'white' },
          { x: 4, y: 5, color: 'black' },
          { x: 5, y: 5, color: 'white' }
        ],
        marks: [
          { x: 3, y: 6, type: 'number', label: '1' },
          { x: 2, y: 5, type: 'number', label: '2' },
          { x: 2, y: 4, type: 'number', label: '3' },
          { x: 2, y: 3, type: 'number', label: '4' },
          { x: 3, y: 2, type: 'number', label: '5' },
          { x: 4, y: 2, type: 'number', label: '6' },
          { x: 5, y: 2, type: 'number', label: '7' },
          { x: 6, y: 3, type: 'number', label: '8' },
          { x: 6, y: 4, type: 'number', label: '9' },
          { x: 6, y: 5, type: 'number', label: '10' },
          { x: 5, y: 6, type: 'number', label: '11' },
          { x: 4, y: 6, type: 'number', label: '1' },
        ]
      }
    },
    {
      title: "Ăn Quân Địch",
      description: "Có thể đánh vào điểm hết khí của quân địch(A) để bắt quân. Khi một quân hoặc nhóm quân hết khí, chúng sẽ bị bắt và loại khỏi bàn cờ.",
      board: {
        stones: [
          { x: 1, y: 1, color: 'black' },
          { x: 2, y: 1, color: 'black' },
          { x: 3, y: 1, color: 'white' },
          { x: 4, y: 1, color: 'white' },
          { x: 4, y: 2, color: 'white' },
          { x: 4, y: 3, color: 'white' },
          { x: 1, y: 2, color: 'black' },
          { x: 2, y: 2, color: 'white' },
          { x: 3, y: 2, color: 'black' },
          { x: 3, y: 3, color: 'black' },
          { x: 1, y: 3, color: 'black' },
          { x: 1, y: 4, color: 'black' },
          { x: 2, y: 4, color: 'black' },
          { x: 3, y: 4, color: 'white' },
          { x: 4, y: 4, color: 'white' }
        ],
        marks: [
          { x: 2, y: 3, type: 'letter', label: 'A'  },

        ]
      }
    },
    {
      title: "Luật Cướp Cờ KO - Phần 1",
      description: "Tình huống 'Ko' xảy ra khi hai bên có thể ăn lẫn nhau vô tận ở cùng một vị trí. Trong hình này, nếu quân đen ở giữa bị bắt, quân đen không được ngay lập tức bắt lại.",
      board: {
        stones: [
          { x: 4, y: 2, color: 'black' },
          { x: 5, y: 2, color: 'white' },
          { x: 3, y: 3, color: 'black' },
          { x: 5, y: 3, color: 'black' },
          { x: 6, y: 3, color: 'white' },
          { x: 4, y: 4, color: 'black' },
          { x: 5, y: 4, color: 'white' }
        ],
        marks: []
      }
    },
    {
      title: "Luật Cướp Cờ KO - Phần 2",
      description: "Sau khi quân đen bị bắt . Đen không được ngay lập tức bắt lại mà phải đánh ở chỗ khác trước. Sau đó lượt tiếp theo mới được phép ăn lại.",
      board: {
        stones: [
          { x: 4, y: 2, color: 'black' },
          { x: 5, y: 2, color: 'white' },
          { x: 3, y: 3, color: 'black' },
          { x: 4, y: 3, color: 'white' },
          { x: 6, y: 3, color: 'white' },
          { x: 4, y: 4, color: 'black' },
          { x: 5, y: 4, color: 'white' },
          { x: 7, y: 5, color: 'black' },

        ],
        marks: [
          
        ]
      }
    },
    {
      title: "Chiếm Lãnh Thổ",
      description: "Mục tiêu của cờ vây là chiếm lãnh thổ. Các tam giác đánh dấu lãnh thổ của mỗi bên. Người có lãnh thổ nhiều hơn sẽ thắng cuộc.",
      board: {
        stones: [
          // Khu vực đen góc trên trái
          { x: 1, y: 0, color: 'black' },
          { x: 0, y: 1, color: 'black' },
          { x: 1, y: 1, color: 'black' },
          
          // Khu vực đen góc trên giữa
          { x: 3, y: 0, color: 'black' },
          { x: 5, y: 0, color: 'black' },
          { x: 3, y: 1, color: 'black' },
          { x: 4, y: 1, color: 'black' },
          { x: 5, y: 1, color: 'black' },
          
          // Khu vực đen ở giữa
          { x: 3, y: 3, color: 'black' },
          { x: 4, y: 3, color: 'black' },
          { x: 2, y: 4, color: 'black' },
          { x: 4, y: 4, color: 'black' },
          { x: 2, y: 5, color: 'black' },
          { x: 3, y: 5, color: 'black' },
          { x: 4, y: 5, color: 'black' },
          
          // Khu vực trắng bên phải
          { x: 6, y: 4, color: 'white' },
          { x: 7, y: 4, color: 'white' },
          { x: 8, y: 4, color: 'white' },
          { x: 6, y: 5, color: 'white' },
          { x: 6, y: 6, color: 'white' },
          { x: 7, y: 6, color: 'white' },
          { x: 8, y: 6, color: 'white' },
          { x: 6, y: 7, color: 'white' },
          { x: 7, y: 7, color: 'black' },
          { x: 8, y: 7, color: 'white' },
          { x: 8, y: 8, color: 'white' }
        ],
        marks: [
          { x: 0, y: 0, type: 'triangle' },
          { x: 3, y: 0, type: 'triangle' },
          { x: 4, y: 0, type: 'triangle' },
          { x: 3, y: 4, type: 'triangle' },
          { x: 7, y: 5, type: 'triangle' },
          { x: 8, y: 5, type: 'triangle' },


        ]
      }
    },
    {
      title: "Cách Tính Điểm",
      description: "Điểm số = Tổng số quân trên bàn cờ + Số lãnh thổ. Quan trọng: Quân Trắng được cộng thêm 7.5 điểm Komi để bù đắp lợi thế đi sau. Người có tổng điểm cao hơn sẽ thắng.",
      board: {
        stones: [
          // Khu vực đen góc trên trái
          { x: 1, y: 0, color: 'black' },
          { x: 0, y: 1, color: 'black' },
          { x: 1, y: 1, color: 'black' },
          
          // Khu vực đen góc trên giữa
          { x: 3, y: 0, color: 'black' },
          { x: 5, y: 0, color: 'black' },
          { x: 3, y: 1, color: 'black' },
          { x: 4, y: 1, color: 'black' },
          { x: 5, y: 1, color: 'black' },
          
          // Khu vực đen ở giữa
          { x: 3, y: 3, color: 'black' },
          { x: 4, y: 3, color: 'black' },
          { x: 2, y: 4, color: 'black' },
          { x: 4, y: 4, color: 'black' },
          { x: 2, y: 5, color: 'black' },
          { x: 3, y: 5, color: 'black' },
          { x: 4, y: 5, color: 'black' },
          
          // Khu vực trắng bên phải
          { x: 6, y: 4, color: 'white' },
          { x: 7, y: 4, color: 'white' },
          { x: 8, y: 4, color: 'white' },
          { x: 6, y: 5, color: 'white' },
          { x: 6, y: 6, color: 'white' },
          { x: 7, y: 6, color: 'white' },
          { x: 8, y: 6, color: 'white' },
          { x: 6, y: 7, color: 'white' },
          { x: 7, y: 7, color: 'black' },
          { x: 8, y: 7, color: 'white' },
          { x: 8, y: 8, color: 'white' }
        ],
        marks: [
          { x: 0, y: 0, type: 'triangle' },
          { x: 3, y: 0, type: 'triangle' },
          { x: 4, y: 0, type: 'triangle' },
          { x: 3, y: 4, type: 'triangle' },
          { x: 7, y: 5, type: 'triangle' },
          { x: 8, y: 5, type: 'triangle' },

        ]
      }
    }
  ];

  const nextStep = () => {
    if (currentStep < tutorialSteps.length - 1) {
      setCurrentStep(currentStep + 1);
    }
  };

  const prevStep = () => {
    if (currentStep > 0) {
      setCurrentStep(currentStep - 1);
    }
  };

  if (!isOpen) {
    return null;
  }

  return (
    <div className="go-tutorial-overlay" onClick={onClose}>
      <div className="go-tutorial-dialog" onClick={(e) => e.stopPropagation()}>
        {/* Header */}
        <div className="go-tutorial-header">
          <h2 className="go-tutorial-title">Hướng Dẫn Chơi Cờ Vây</h2>
          <button
            onClick={onClose}
            className="go-tutorial-close-btn"
          >
            <FaTimes size={24} />
          </button>
        </div>
        <p className="go-tutorial-step-indicator">Bước {currentStep + 1} / {tutorialSteps.length}</p>

        {/* Progress Bar */}
        <div className="go-tutorial-progress-bar">
          <div
            className="go-tutorial-progress-fill"
            style={{ width: `${((currentStep + 1) / tutorialSteps.length) * 100}%` }}
          />
        </div>

        {/* Content */}
        <div className="go-tutorial-content">
          <h3 className="go-tutorial-step-title">
            {tutorialSteps[currentStep].title}
          </h3>
          
          <div className="go-tutorial-main-content">
            {/* Bàn cờ minh họa - Bên trái */}
            <div className="go-tutorial-board-container">
              <GoBoard
                stones={tutorialSteps[currentStep].board.stones}
                marks={tutorialSteps[currentStep].board.marks}
              />
            </div>

            {/* Mô tả - Bên phải */}
            <p className="go-tutorial-description">
              {tutorialSteps[currentStep].description}
            </p>
          </div>
        </div>

        {/* Navigation */}
        <div className="go-tutorial-navigation">
          <button
            onClick={prevStep}
            disabled={currentStep === 0}
            className={`go-tutorial-nav-btn go-tutorial-nav-prev ${
              currentStep === 0 ? 'disabled' : ''
            }`}
          >
            <FaChevronLeft size={20} />
            Trước
          </button>

          <div className="go-tutorial-dots">
            {tutorialSteps.map((_, index) => (
              <div
                key={index}
                className={`go-tutorial-dot ${
                  index === currentStep ? 'active' : ''
                }`}
              />
            ))}
          </div>

          <button
            onClick={nextStep}
            disabled={currentStep === tutorialSteps.length - 1}
            className={`go-tutorial-nav-btn go-tutorial-nav-next ${
              currentStep === tutorialSteps.length - 1 ? 'disabled' : ''
            }`}
          >
            Tiếp
            <FaChevronRight size={20} />
          </button>
        </div>
      </div>
    </div>
  );
};

export default GoTutorial;

