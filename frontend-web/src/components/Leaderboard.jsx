import { useState, useEffect } from 'react'
import { FaTrophy, FaMedal, FaCrown, FaGem, FaStar, FaTimes, FaSpinner } from 'react-icons/fa'
import api from '../services/api'
import { getRankTier, formatRankInfo } from '../utils/rankUtils'
import './Leaderboard.css'

const Leaderboard = ({ isOpen, onClose }) => {
  const [leaderboard, setLeaderboard] = useState([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)
  const [currentUserRank, setCurrentUserRank] = useState(null)

  useEffect(() => {
    if (isOpen) {
      loadLeaderboard()
    }
  }, [isOpen])

  const loadLeaderboard = async () => {
    try {
      setLoading(true)
      setError(null)
      
      // Load leaderboard
      const leaderboardRes = await api.get('/statistics/leaderboard?limit=100')
      setLeaderboard(leaderboardRes.data || [])
      
      // Get current user's rank
      try {
        const statsRes = await api.get('/statistics/me')
        const userElo = statsRes.data?.elo_rating || 0
        
        // Find user's rank in leaderboard
        const userRankIndex = leaderboardRes.data?.findIndex(
          entry => entry.user_id === statsRes.data?.user_id
        )
        
        if (userRankIndex !== -1 && userRankIndex !== undefined) {
          setCurrentUserRank({
            ...leaderboardRes.data[userRankIndex],
            rank: userRankIndex + 1
          })
        } else {
          // User not in top 100, estimate rank
          const usersAbove = leaderboardRes.data?.filter(
            entry => entry.elo_rating > userElo
          ).length || 0
          setCurrentUserRank({
            rank: usersAbove + 1,
            elo_rating: userElo,
            username: statsRes.data?.username,
            total_matches: statsRes.data?.total_matches || 0,
            win_rate: statsRes.data?.win_rate || 0
          })
        }
      } catch (err) {
        console.warn('Could not load current user rank:', err)
      }
    } catch (err) {
      console.error('Failed to load leaderboard:', err)
      setError('Không thể tải bảng xếp hạng')
    } finally {
      setLoading(false)
    }
  }

  const getRankIcon = (rank) => {
    if (rank === 1) return <FaCrown className="rank-icon rank-gold" />
    if (rank === 2) return <FaMedal className="rank-icon rank-silver" />
    if (rank === 3) return <FaMedal className="rank-icon rank-bronze" />
    return null
  }

  const getRankBadge = (rank, isOver99 = false) => {
    const displayRank = isOver99 ? '99+' : rank
    if (rank <= 3 && !isOver99) {
      return (
        <div className="rank-badge top-rank">
          {getRankIcon(rank)}
          <span className="rank-number">{displayRank}</span>
        </div>
      )
    }
    return <div className="rank-badge"><span className="rank-number">{displayRank}</span></div>
  }

  if (!isOpen) return null

  return (
    <div className="leaderboard-overlay" onClick={onClose}>
      <div className="leaderboard-dialog" onClick={(e) => e.stopPropagation()}>
        <div className="leaderboard-header">
          <div className="leaderboard-title">
            <FaTrophy className="title-icon" />
            <h2>Bảng Xếp Hạng</h2>
          </div>
          <button className="leaderboard-close-btn" onClick={onClose}>
            <FaTimes />
          </button>
        </div>

        <div className="leaderboard-content">
          {loading ? (
            <div className="leaderboard-loading">
              <FaSpinner className="spinner" />
              <p>Đang tải...</p>
            </div>
          ) : error ? (
            <div className="leaderboard-error">
              <p>{error}</p>
              <button onClick={loadLeaderboard}>Thử lại</button>
            </div>
          ) : (
            <>
              {/* Top 3 Podium */}
              {leaderboard.length >= 3 && (
                <div className="leaderboard-podium">
                  <div className="podium-item second">
                    <div className="podium-rank">2</div>
                    <div className="podium-info">
                      <div className="podium-avatar">{leaderboard[1]?.username?.[0]?.toUpperCase() || '?'}</div>
                      <div className="podium-details">
                        <div className="podium-name">{leaderboard[1]?.username || 'N/A'}</div>
                        <div className="podium-elo">{leaderboard[1]?.elo_rating || 0} ELO</div>
                        <div className="podium-tier">{getRankTier(leaderboard[1]?.elo_rating || 0).name}</div>
                      </div>
                    </div>
                  </div>
                  
                  <div className="podium-item first">
                    <div className="podium-rank">1</div>
                    <div className="podium-info">
                      <div className="podium-avatar champion">{leaderboard[0]?.username?.[0]?.toUpperCase() || '?'}</div>
                      <div className="podium-details">
                        <div className="podium-name">{leaderboard[0]?.username || 'N/A'}</div>
                        <div className="podium-elo">{leaderboard[0]?.elo_rating || 0} ELO</div>
                        <div className="podium-tier">{getRankTier(leaderboard[0]?.elo_rating || 0).name}</div>
                      </div>
                    </div>
                  </div>
                  
                  <div className="podium-item third">
                    <div className="podium-rank">3</div>
                    <div className="podium-info">
                      <div className="podium-avatar">{leaderboard[2]?.username?.[0]?.toUpperCase() || '?'}</div>
                      <div className="podium-details">
                        <div className="podium-name">{leaderboard[2]?.username || 'N/A'}</div>
                        <div className="podium-elo">{leaderboard[2]?.elo_rating || 0} ELO</div>
                        <div className="podium-tier">{getRankTier(leaderboard[2]?.elo_rating || 0).name}</div>
                      </div>
                    </div>
                  </div>
                </div>
              )}

              {/* Current User Rank - Always show */}
              {currentUserRank && (
                <div className="current-user-rank">
                  <div className="user-rank-label">Vị trí của bạn</div>
                  <div className="leaderboard-entry user-entry">
                    {getRankBadge(currentUserRank.rank > 99 ? 99 : currentUserRank.rank, currentUserRank.rank > 99)}
                    <div className="entry-info">
                      <div className="entry-name">{currentUserRank.username}</div>
                      <div className="entry-stats">
                        <span className="entry-elo">{currentUserRank.elo_rating} ELO</span>
                        <span className="entry-matches">{currentUserRank.total_matches} trận</span>
                        <span className="entry-winrate">{currentUserRank.win_rate?.toFixed(1)}%</span>
                      </div>
                    </div>
                    <div className="entry-tier">
                      <span className="tier-badge" style={{ 
                        background: getRankTier(currentUserRank.elo_rating).bgColor,
                        borderColor: getRankTier(currentUserRank.elo_rating).borderColor,
                        color: getRankTier(currentUserRank.elo_rating).color
                      }}>
                        {getRankTier(currentUserRank.elo_rating).icon} {getRankTier(currentUserRank.elo_rating).name}
                      </span>
                    </div>
                  </div>
                </div>
              )}

              {/* Leaderboard List */}
              <div className="leaderboard-list">
                {leaderboard.map((entry, index) => {
                  const rank = index + 1
                  const rankInfo = formatRankInfo(entry.elo_rating)
                  
                  // Skip top 3 if podium is shown
                  if (rank <= 3 && leaderboard.length >= 3) return null
                  
                  // Check if this is current user (by user_id or username if user_id not available)
                  const isCurrentUser = currentUserRank && (
                    currentUserRank.user_id === entry.user_id || 
                    (currentUserRank.username === entry.username && rank === currentUserRank.rank)
                  )
                  
                  // Format rank: if > 99, show "99+"
                  const displayRank = rank > 99 ? '99+' : rank
                  
                  return (
                    <div 
                      key={entry.user_id || entry.username || index} 
                      className={`leaderboard-entry ${isCurrentUser ? 'user-entry' : ''}`}
                    >
                      {getRankBadge(rank, rank > 99)}
                      <div className="entry-info">
                        <div className="entry-name">{entry.username || entry.display_name || 'N/A'}</div>
                        <div className="entry-stats">
                          <span className="entry-elo">{entry.elo_rating || 0} ELO</span>
                          <span className="entry-matches">{entry.total_matches || 0} trận</span>
                          <span className="entry-winrate">{entry.win_rate?.toFixed(1) || 0}%</span>
                        </div>
                      </div>
                      <div className="entry-tier">
                        <span className="tier-badge" style={{ 
                          background: rankInfo.bgColor,
                          borderColor: rankInfo.borderColor,
                          color: rankInfo.color
                        }}>
                          {rankInfo.icon} {rankInfo.tier}
                        </span>
                      </div>
                    </div>
                  )
                })}
              </div>
            </>
          )}
        </div>
      </div>
    </div>
  )
}

export default Leaderboard

