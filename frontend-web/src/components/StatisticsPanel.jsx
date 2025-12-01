import { useState } from 'react'
import { FaChartBar, FaTrophy, FaTimes, FaPercent, FaCircle, FaChevronDown, FaChevronUp } from 'react-icons/fa'
import './StatisticsPanel.css'

const StatisticsPanel = ({ statistics, compact = false }) => {
  const [isExpanded, setIsExpanded] = useState(false)

  if (!statistics) {
    return (
      <div className={`statistics-panel ${compact ? 'compact' : ''}`}>
        <div className="loading">Đang tải...</div>
      </div>
    )
  }

  // Compact mode: hiển thị dạng horizontal bar với các số liệu chính
  if (compact) {
    return (
      <div className={`statistics-panel compact ${isExpanded ? 'expanded' : ''}`}>
        <div className="stats-compact-header" onClick={() => setIsExpanded(!isExpanded)}>
          <div className="stats-compact-main">
            <FaTrophy className="stats-compact-icon" />
            <div className="stats-compact-primary">
              <span className="stats-compact-label">Elo</span>
              <span className="stats-compact-value">{statistics.elo_rating || 1500}</span>
            </div>
            <div className="stats-compact-secondary">
              <span className="stats-compact-mini">
                <FaChartBar className="stats-mini-icon" />
                {statistics.total_matches || 0}
              </span>
              <span className="stats-compact-mini">
                <FaTrophy className="stats-mini-icon" />
                {statistics.wins || 0}
              </span>
              <span className="stats-compact-mini">
                <FaPercent className="stats-mini-icon" />
                {statistics.win_rate?.toFixed(0) || 0}%
              </span>
            </div>
          </div>
          <div className="stats-expand-toggle">
            {isExpanded ? <FaChevronUp /> : <FaChevronDown />}
          </div>
        </div>
        
        {isExpanded && (
          <div className="stats-compact-details">
            <div className="stats-detail-header">
              <h4>Chi tiết thống kê</h4>
              <button 
                className="stats-detail-close" 
                onClick={() => setIsExpanded(false)}
                aria-label="Đóng"
              >
                <FaTimes />
              </button>
            </div>
            
            <div className="stats-detail-row">
              <span className="stats-detail-label">Tổng trận:</span>
              <span className="stats-detail-value">{statistics.total_matches || 0}</span>
            </div>
            <div className="stats-detail-row">
              <span className="stats-detail-label">Thắng:</span>
              <span className="stats-detail-value">{statistics.wins || 0}</span>
            </div>
            <div className="stats-detail-row">
              <span className="stats-detail-label">Thua:</span>
              <span className="stats-detail-value">{statistics.losses || 0}</span>
            </div>
            <div className="stats-detail-row">
              <span className="stats-detail-label">Tỷ lệ thắng:</span>
              <span className="stats-detail-value">{statistics.win_rate?.toFixed(1) || 0}%</span>
            </div>
            
            {/* Statistics by AI level */}
            <div className="stats-detail-section">
              <div className="stats-detail-section-title">🤖 Chơi với Máy</div>
              
              <div className="stats-detail-subsection">
                <div className="stats-detail-subtitle">Dễ</div>
                <div className="stats-detail-row">
                  <span className="stats-detail-label">Trận:</span>
                  <span className="stats-detail-value">{statistics.matches_ai_easy || 0}</span>
                </div>
                <div className="stats-detail-row">
                  <span className="stats-detail-label">Thắng:</span>
                  <span className="stats-detail-value">{statistics.wins_ai_easy || 0}</span>
                </div>
                <div className="stats-detail-row">
                  <span className="stats-detail-label">Thua:</span>
                  <span className="stats-detail-value">{statistics.losses_ai_easy || 0}</span>
                </div>
                <div className="stats-detail-row">
                  <span className="stats-detail-label">Tỷ lệ:</span>
                  <span className="stats-detail-value">{statistics.win_rate_ai_easy?.toFixed(1) || 0}%</span>
                </div>
              </div>
              
              <div className="stats-detail-subsection">
                <div className="stats-detail-subtitle">Trung bình</div>
                <div className="stats-detail-row">
                  <span className="stats-detail-label">Trận:</span>
                  <span className="stats-detail-value">{statistics.matches_ai_medium || 0}</span>
                </div>
                <div className="stats-detail-row">
                  <span className="stats-detail-label">Thắng:</span>
                  <span className="stats-detail-value">{statistics.wins_ai_medium || 0}</span>
                </div>
                <div className="stats-detail-row">
                  <span className="stats-detail-label">Thua:</span>
                  <span className="stats-detail-value">{statistics.losses_ai_medium || 0}</span>
                </div>
                <div className="stats-detail-row">
                  <span className="stats-detail-label">Tỷ lệ:</span>
                  <span className="stats-detail-value">{statistics.win_rate_ai_medium?.toFixed(1) || 0}%</span>
                </div>
              </div>
              
              <div className="stats-detail-subsection">
                <div className="stats-detail-subtitle">Khó</div>
                <div className="stats-detail-row">
                  <span className="stats-detail-label">Trận:</span>
                  <span className="stats-detail-value">{statistics.matches_ai_hard || 0}</span>
                </div>
                <div className="stats-detail-row">
                  <span className="stats-detail-label">Thắng:</span>
                  <span className="stats-detail-value">{statistics.wins_ai_hard || 0}</span>
                </div>
                <div className="stats-detail-row">
                  <span className="stats-detail-label">Thua:</span>
                  <span className="stats-detail-value">{statistics.losses_ai_hard || 0}</span>
                </div>
                <div className="stats-detail-row">
                  <span className="stats-detail-label">Tỷ lệ:</span>
                  <span className="stats-detail-value">{statistics.win_rate_ai_hard?.toFixed(1) || 0}%</span>
                </div>
              </div>
              
              <div className="stats-detail-subsection">
                <div className="stats-detail-subtitle">Siêu khó</div>
                <div className="stats-detail-row">
                  <span className="stats-detail-label">Trận:</span>
                  <span className="stats-detail-value">{statistics.matches_ai_super_hard || 0}</span>
                </div>
                <div className="stats-detail-row">
                  <span className="stats-detail-label">Thắng:</span>
                  <span className="stats-detail-value">{statistics.wins_ai_super_hard || 0}</span>
                </div>
                <div className="stats-detail-row">
                  <span className="stats-detail-label">Thua:</span>
                  <span className="stats-detail-value">{statistics.losses_ai_super_hard || 0}</span>
                </div>
                <div className="stats-detail-row">
                  <span className="stats-detail-label">Tỷ lệ:</span>
                  <span className="stats-detail-value">{statistics.win_rate_ai_super_hard?.toFixed(1) || 0}%</span>
                </div>
              </div>
            </div>
            
            <div className="stats-detail-section">
              <div className="stats-detail-section-title">👤 Đấu Online (PvP)</div>
              <div className="stats-detail-row">
                <span className="stats-detail-label">Trận:</span>
                <span className="stats-detail-value">{statistics.matches_vs_player || 0}</span>
              </div>
              <div className="stats-detail-row">
                <span className="stats-detail-label">Thắng:</span>
                <span className="stats-detail-value">{statistics.wins_vs_player || 0}</span>
              </div>
              <div className="stats-detail-row">
                <span className="stats-detail-label">Thua:</span>
                <span className="stats-detail-value">{statistics.losses_vs_player || 0}</span>
              </div>
              <div className="stats-detail-row">
                <span className="stats-detail-label">Hòa:</span>
                <span className="stats-detail-value">{statistics.draws_vs_player || 0}</span>
              </div>
              <div className="stats-detail-row">
                <span className="stats-detail-label">Tỷ lệ thắng:</span>
                <span className="stats-detail-value">{statistics.win_rate_vs_player?.toFixed(1) || 0}%</span>
              </div>
            </div>
          </div>
        )}
      </div>
    )
  }

  // Full mode: giữ nguyên layout cũ
  return (
    <div className="statistics-panel">
      <div className="statistics-header">
        <FaChartBar className="panel-icon" />
        <h2>Thống Kê</h2>
      </div>
      <div className="stats-grid">
        <div className="stat-item">
          <div className="stat-icon">
            <FaCircle />
          </div>
          <div className="stat-content">
            <div className="stat-label">Điểm Elo</div>
            <div className="stat-value">{statistics.elo_rating || 1500}</div>
          </div>
        </div>
        <div className="stat-item">
          <div className="stat-icon">
            <FaChartBar />
          </div>
          <div className="stat-content">
            <div className="stat-label">Tổng trận</div>
            <div className="stat-value">{statistics.total_matches || 0}</div>
          </div>
        </div>
        <div className="stat-item">
          <div className="stat-icon">
            <FaTrophy />
          </div>
          <div className="stat-content">
            <div className="stat-label">Thắng</div>
            <div className="stat-value">{statistics.wins || 0}</div>
          </div>
        </div>
        <div className="stat-item">
          <div className="stat-icon">
            <FaTimes />
          </div>
          <div className="stat-content">
            <div className="stat-label">Thua</div>
            <div className="stat-value">{statistics.losses || 0}</div>
          </div>
        </div>
        <div className="stat-item">
          <div className="stat-icon">
            <FaPercent />
          </div>
          <div className="stat-content">
            <div className="stat-label">Tỷ lệ thắng</div>
            <div className="stat-value">{statistics.win_rate?.toFixed(1) || 0}%</div>
          </div>
        </div>
      </div>
    </div>
  )
}

export default StatisticsPanel

