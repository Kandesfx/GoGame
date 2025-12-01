import { FaHistory, FaClock, FaUser, FaRobot, FaTrophy, FaArrowUp, FaArrowDown } from 'react-icons/fa'
import './MatchList.css'

const MatchList = ({ matches = [], onMatchSelect, onMatchClick, compact = false }) => {
  const handleMatchClick = (match) => {
    if (onMatchClick) {
      onMatchClick(match)
    } else if (onMatchSelect) {
      onMatchSelect(match)
    }
  }
  
  const formatResult = (result) => {
    if (!result) return 'Đang chơi'
    if (result === 'DRAW') return 'Hòa'
    if (result.endsWith('+R')) {
      return result.startsWith('B') ? 'Đen thắng' : 'Trắng thắng'
    }
    if (result.includes('+')) {
      const winner = result.startsWith('B') ? 'Đen' : 'Trắng'
      return `${winner} thắng`
    }
    return result
  }
  
  const formatEloChange = (eloChange) => {
    if (eloChange === null || eloChange === undefined) return null
    if (eloChange > 0) {
      return { text: `+${eloChange}`, positive: true }
    } else if (eloChange < 0) {
      return { text: `${eloChange}`, positive: false }
    } else {
      return { text: '0', positive: null }
    }
  }
  
  const getPlayerInfo = (match) => {
    if (match.ai_level) {
      // AI match
      return {
        blackName: match.black_player_username || 'Bạn',
        whiteName: 'AI',
        userColor: 'Đen',
        userIsBlack: true
      }
    } else {
      // PvP match - hiển thị tên của cả 2 người chơi
      const userIsBlack = match.user_color === 'B'
      return {
        blackName: match.black_player_username || 'Người chơi',
        whiteName: match.white_player_username || 'Người chơi',
        userColor: userIsBlack ? 'Đen' : 'Trắng',
        userIsBlack
      }
    }
  }

  const formatDate = (dateString) => {
    if (!dateString) return 'Không rõ'
    try {
      const date = new Date(dateString)
      const now = new Date()
      const diffMs = now - date
      const diffMins = Math.floor(diffMs / 60000)
      const diffHours = Math.floor(diffMs / 3600000)
      const diffDays = Math.floor(diffMs / 86400000)

      // Format relative time
      if (diffMins < 1) {
        return 'Vừa xong'
      } else if (diffMins < 60) {
        return `${diffMins} phút trước`
      } else if (diffHours < 24) {
        return `${diffHours} giờ trước`
      } else if (diffDays < 7) {
        return `${diffDays} ngày trước`
      } else {
        // Nếu quá 7 ngày, hiển thị full date
        return date.toLocaleDateString('vi-VN', {
          day: '2-digit',
          month: '2-digit',
          year: 'numeric',
          hour: '2-digit',
          minute: '2-digit'
        })
      }
    } catch (e) {
      return dateString
    }
  }

  const formatFullDate = (dateString) => {
    if (!dateString) return 'Không rõ'
    try {
      const date = new Date(dateString)
      return date.toLocaleDateString('vi-VN', {
        weekday: 'long',
        day: '2-digit',
        month: '2-digit',
        year: 'numeric',
        hour: '2-digit',
        minute: '2-digit',
        second: '2-digit'
      })
    } catch (e) {
      return dateString
    }
  }

  const formatFullDateShort = (dateString) => {
    if (!dateString) return ''
    try {
      const date = new Date(dateString)
      return date.toLocaleDateString('vi-VN', {
        day: '2-digit',
        month: '2-digit',
        year: 'numeric',
        hour: '2-digit',
        minute: '2-digit'
      })
    } catch (e) {
      return ''
    }
  }

  const getMatchType = (match) => {
    if (match.ai_level) return `AI Level ${match.ai_level}`
    return 'PvP'
  }

  const getMatchTypeIcon = (match) => {
    if (match.ai_level) return <FaRobot className="match-type-icon" />
    return <FaUser className="match-type-icon" />
  }

  return (
    <div className={`match-list ${compact ? 'compact' : ''}`}>
      {!compact && (
        <div className="match-list-header">
          <FaHistory className="list-icon" />
          <h2>Lịch Sử Trận Đấu</h2>
        </div>
      )}
      <div className="match-items">
        {matches.length === 0 ? (
          <div className="empty-state">Chưa có trận đấu nào</div>
        ) : (
          matches.map((match) => {
            const playerInfo = getPlayerInfo(match)
            const eloChange = formatEloChange(match.user_elo_change)
            const isWin = match.result && (
              (match.user_color === 'B' && match.result.startsWith('B+')) ||
              (match.user_color === 'W' && match.result.startsWith('W+'))
            )
            const isDraw = match.result === 'DRAW'
            
            return (
              <div
                key={match.id}
                className="match-item"
                onClick={() => handleMatchClick(match)}
              >
                <div className="match-item-header">
                  <div className={`match-result-badge ${isWin ? 'result-win' : isDraw ? 'result-draw' : match.result ? 'result-loss' : ''}`}>
                    {formatResult(match.result)}
                  </div>
                  <div className="match-board-icon">
                    {match.board_size}x{match.board_size}
                  </div>
                </div>
                <div className={`match-item-details ${compact ? 'compact' : ''}`}>
                  <div className="match-type-info">
                    {getMatchTypeIcon(match)}
                    <span>{getMatchType(match)}</span>
                  </div>
                  
                  {/* Player Info - chỉ hiển thị khi không compact */}
                  {!compact && (
                    <div className="match-players-info">
                      <div className="player-info-row">
                        <span className="player-label">Đen:</span>
                        <span className={`player-name ${playerInfo.userIsBlack ? 'player-you' : ''}`}>
                          {playerInfo.blackName}
                        </span>
                        {playerInfo.userIsBlack && <span className="player-badge">(Bạn)</span>}
                      </div>
                      <div className="player-info-row">
                        <span className="player-label">Trắng:</span>
                        <span className={`player-name ${!playerInfo.userIsBlack ? 'player-you' : ''}`}>
                          {playerInfo.whiteName}
                        </span>
                        {!playerInfo.userIsBlack && <span className="player-badge">(Bạn)</span>}
                      </div>
                    </div>
                  )}
                  
                  {/* ELO Change - chỉ hiển thị cho PvP matches đã kết thúc */}
                  {!match.ai_level && match.result && (
                    <div className={`match-elo-change ${eloChange?.positive ? 'elo-positive' : eloChange?.positive === false ? 'elo-negative' : 'elo-neutral'}`}>
                      {eloChange?.positive ? <FaArrowUp className="elo-icon" /> : eloChange?.positive === false ? <FaArrowDown className="elo-icon" /> : null}
                      <span className="elo-text">
                        {match.user_elo_change !== null && match.user_elo_change !== undefined 
                          ? (match.user_elo_change > 0 ? `+${match.user_elo_change}` : `${match.user_elo_change}`)
                          : '0'}
                      </span>
                      <span className="elo-label">ELO</span>
                    </div>
                  )}
                  
                  <div className="match-time-wrapper">
                    <div className="match-time" title={formatFullDate(match.started_at || match.created_at || match.updated_at)}>
                      <FaClock className="time-icon" />
                      <span className="time-relative">{formatDate(match.started_at || match.created_at || match.updated_at)}</span>
                    </div>
                    {!compact && (
                      <div className="match-time-full">
                        {formatFullDateShort(match.started_at || match.created_at || match.updated_at)}
                      </div>
                    )}
                  </div>
                </div>
                {match.status && match.status !== 'finished' && (
                  <div className="match-status-badge">
                    {match.status === 'active' ? 'Đang chơi' : match.status}
                  </div>
                )}
              </div>
            )
          })
        )}
      </div>
    </div>
  )
}

export default MatchList

