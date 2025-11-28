import { FaTrophy, FaChevronRight, FaSpinner } from 'react-icons/fa'
import { getRankTier } from '../utils/rankUtils'
import './LeaderboardPreview.css'

const LeaderboardPreview = ({ topPlayers, onViewAll, loading }) => {
  if (loading) {
    return (
      <div className="leaderboard-preview">
        <div className="preview-header">
          <FaTrophy className="preview-icon" />
          <h3 className="preview-title">Xếp Hạng</h3>
        </div>
        <div className="preview-loading">
          <FaSpinner className="spinner" />
        </div>
      </div>
    )
  }

  if (!topPlayers || topPlayers.length === 0) {
    return (
      <div className="leaderboard-preview">
        <div className="preview-header">
          <FaTrophy className="preview-icon" />
          <h3 className="preview-title">Xếp Hạng</h3>
        </div>
        <div className="preview-empty">
          <p>Chưa có dữ liệu</p>
        </div>
      </div>
    )
  }

  return (
    <div className="leaderboard-preview">
      <div className="preview-header">
        <FaTrophy className="preview-icon" />
        <h3 className="preview-title">Xếp Hạng</h3>
      </div>
      
      <div className="preview-list">
        {topPlayers.slice(0, 5).map((player, index) => {
          const rank = index + 1
          const tier = getRankTier(player.elo_rating || 0)
          
          return (
            <div 
              key={player.user_id || index} 
              className="preview-entry"
            >
              <div className="preview-rank">
                {rank === 1 && '🥇'}
                {rank === 2 && '🥈'}
                {rank === 3 && '🥉'}
                {rank > 3 && `#${rank}`}
              </div>
              <div className="preview-info">
                <div 
                  className="preview-name" 
                  style={{ color: tier.color }}
                >
                  {player.username || player.display_name || 'N/A'}
                </div>
                <div className="preview-elo">
                  {player.elo_rating || 0}
                </div>
              </div>
            </div>
          )
        })}
      </div>

      <button className="preview-view-all" onClick={onViewAll}>
        <span>Xem tất cả</span>
        <FaChevronRight className="chevron-icon" />
      </button>
    </div>
  )
}

export default LeaderboardPreview

