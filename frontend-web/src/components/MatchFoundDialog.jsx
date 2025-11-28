import { useState, useEffect } from 'react'
import { FaCheck, FaTimes, FaUser, FaSpinner, FaClock } from 'react-icons/fa'
import { useAuth } from '../contexts/AuthContext'
import api from '../services/api'
import './MatchFoundDialog.css'

const MatchFoundDialog = ({ match, onStart, onCancel }) => {
  const { user } = useAuth()
  const [isReady, setIsReady] = useState(false)
  const [opponentReady, setOpponentReady] = useState(false)
  const [bothReady, setBothReady] = useState(false)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const [checkingInterval, setCheckingInterval] = useState(null)

  // Validate match object
  if (!match || !match.id) {
    console.error('❌ [MatchFoundDialog] Invalid match object:', match)
    return (
      <div className="match-found-dialog-overlay">
        <div className="match-found-dialog">
          <div className="error-message">
            Lỗi: Match không hợp lệ. Vui lòng thử lại.
          </div>
          <button onClick={onCancel} className="btn btn-secondary">
            Đóng
          </button>
        </div>
      </div>
    )
  }

  // Determine user's color and opponent info
  // QUAN TRỌNG: Nếu user_color không có từ backend, tính từ black_player_id và white_player_id
  let userIsBlack = false
  if (match.user_color === 'B' || match.user_color === 'W') {
    userIsBlack = match.user_color === 'B'
  } else if (user && match.black_player_id && match.white_player_id) {
    // Fallback: Tính từ player_id nếu user_color không có
    const userIdStr = String(user.id)
    const blackPlayerIdStr = String(match.black_player_id)
    userIsBlack = blackPlayerIdStr === userIdStr
    console.log('🔄 [MatchFoundDialog] user_color not provided, calculated from player_id:', {
      userId: userIdStr,
      blackPlayerId: blackPlayerIdStr,
      whitePlayerId: String(match.white_player_id),
      userIsBlack
    })
  } else {
    // Fallback cuối cùng: Mặc định là Black (người tạo match)
    userIsBlack = true
    console.warn('⚠️ [MatchFoundDialog] Cannot determine user color, defaulting to Black')
  }
  const opponentName = userIsBlack 
    ? (match.white_player_username || 'Người chơi')
    : (match.black_player_username || 'Người chơi')
  const userColor = userIsBlack ? 'Đen' : 'Trắng'
  const opponentColor = userIsBlack ? 'Trắng' : 'Đen'
  
  console.log('🎮 [MatchFoundDialog] Initialized with match:', {
    id: match.id,
    user_color: match.user_color,
    userIsBlack,
    black_ready: match.black_ready,
    white_ready: match.white_ready,
    room_code: match.room_code
  })
  
  // Initialize ready status từ match object khi component mount
  // QUAN TRỌNG: Chỉ initialize opponent ready status, KHÔNG set user's own ready status từ server
  // User's own ready status chỉ được set khi user bấm nút "Sẵn sàng"
  useEffect(() => {
    if (match && match.id) {
      // Fetch match data từ server để có ready status mới nhất
      const initializeReadyStatus = async () => {
        try {
          const response = await api.get(`/matches/${match.id}`, {
            timeout: 10000
          })
          const matchData = response.data
          
          if (matchData) {
            const blackReady = matchData.black_ready || false
            const whiteReady = matchData.white_ready || false
            const opponentReadyStatus = userIsBlack ? whiteReady : blackReady
            const bothReadyNow = blackReady && whiteReady
            
            // QUAN TRỌNG: Chỉ set opponent ready status từ server
            // KHÔNG set user's own ready status từ server khi initialize
            // User's own ready status chỉ được set khi user bấm nút "Sẵn sàng" (trong handleReady)
            // Điều này đảm bảo user luôn có thể bấm nút "Sẵn sàng" cho đến khi họ bấm
            setOpponentReady(opponentReadyStatus)
            setBothReady(bothReadyNow)
            
            // User's own ready status: KHÔNG set từ server khi initialize
            // Luôn bắt đầu với false (user chưa bấm) để user có thể bấm nút
            // Chỉ set true khi user bấm nút "Sẵn sàng"
            // (Nếu user đã bấm trước đó và refresh page, họ sẽ phải bấm lại - đây là behavior mong muốn)
            setIsReady(false)
            
            console.log('🔄 [MatchFoundDialog] Initialized ready status from server:', {
              userReady: isReady ? 'already set (user clicked)' : userReadyStatus,
              opponentReady: opponentReadyStatus,
              bothReady: bothReadyNow,
              blackReady,
              whiteReady
            })
          }
        } catch (error) {
          console.error('❌ [MatchFoundDialog] Error initializing ready status:', error)
          // Fallback to match object data
          const blackReady = match.black_ready || false
          const whiteReady = match.white_ready || false
          const opponentReadyStatus = userIsBlack ? whiteReady : blackReady
          
          // Chỉ set opponent ready status, không set user's own ready status
          setOpponentReady(opponentReadyStatus)
          setBothReady(blackReady && whiteReady)
          
          // User's own ready status: KHÔNG set từ server khi initialize
          // Luôn bắt đầu với false (user chưa bấm) để user có thể bấm nút
          setIsReady(false)
        }
      }
      
      initializeReadyStatus()
    }
  }, [match?.id, userIsBlack]) // Chỉ chạy khi match.id thay đổi

  // Polling để check opponent ready status - chạy ngay khi dialog mở, không cần đợi user ready
  useEffect(() => {
    if (!match || !match.id) {
      console.error('❌ [MatchFoundDialog] Cannot start polling - invalid match')
      return
    }
    
    if (!bothReady) {
      console.log('🔄 [MatchFoundDialog] Starting polling for match:', match.id)
      
      const interval = setInterval(async () => {
        try {
          const response = await api.get(`/matches/${match.id}`, {
            timeout: 10000
          })
          const matchData = response.data
          
          if (matchData) {
            const blackReady = matchData.black_ready || false
            const whiteReady = matchData.white_ready || false
            
            console.log('📊 [MatchFoundDialog] Polling update:', {
              blackReady,
              whiteReady,
              userIsBlack,
              isReady
            })
            
            // Update opponent ready status
            const opponentReadyStatus = userIsBlack ? whiteReady : blackReady
            setOpponentReady(opponentReadyStatus)
            
            // QUAN TRỌNG: Chỉ sync user's own ready status từ server nếu user CHƯA bấm sẵn sàng
            // Nếu user đã bấm sẵn sàng (isReady = true), KHÔNG sync từ server để tránh bị reset
            // Điều này đảm bảo user luôn có thể bấm nút "Sẵn sàng" cho đến khi họ bấm
            const userReadyStatus = userIsBlack ? blackReady : whiteReady
            
            // CHỈ sync user's own ready status khi:
            // 1. Local state là false (user chưa bấm sẵn sàng)
            // 2. Server có ready status khác với local state
            // 
            // KHÔNG sync khi:
            // - Local state là true (user đã bấm sẵn sàng) - giữ nguyên để user có thể bấm nút
            // - Local và server đều true - đã đúng, không cần sync
            if (!isReady) {
              // User chưa bấm sẵn sàng, sync từ server
              // (Thường sẽ là false, nhưng có thể là true nếu user đã bấm trước đó và refresh page)
              if (userReadyStatus !== isReady) {
                console.log('🔄 [MatchFoundDialog] Syncing user ready status from server:', userReadyStatus, '(user has not clicked ready yet)')
                setIsReady(userReadyStatus)
              }
            } else {
              // User đã bấm sẵn sàng (local = true)
              // KHÔNG sync từ server - giữ nguyên local state để user vẫn có thể bấm nút
              // Chỉ log nếu có mismatch để debug
              if (!userReadyStatus) {
                console.warn('⚠️ [MatchFoundDialog] User clicked ready (local=true) but server=false. Keeping local state to allow button to remain clickable.')
              }
            }
            
            // Check if both ready
            const bothReadyNow = blackReady && whiteReady
            setBothReady(bothReadyNow)
            
            // Nếu cả 2 đều ready, tự động start
            if (bothReadyNow) {
              console.log('🚀 [MatchFoundDialog] Both players ready, starting game')
              clearInterval(interval)
              setCheckingInterval(null)
              if (onStart) {
                onStart(match)
              }
            }
          }
        } catch (error) {
          console.error('❌ [MatchFoundDialog] Error checking opponent ready status:', error)
          if (error.response?.status === 404) {
            console.error('❌ [MatchFoundDialog] Match not found:', match.id)
            setError('Match không tồn tại. Vui lòng thử lại.')
            clearInterval(interval)
            setCheckingInterval(null)
          }
        }
      }, 1000) // Check every second
      
      setCheckingInterval(interval)
      
      return () => {
        console.log('🧹 [MatchFoundDialog] Cleaning up polling interval')
        clearInterval(interval)
        setCheckingInterval(null)
      }
    }
  }, [match?.id, bothReady, userIsBlack, onStart]) // Removed isReady to prevent re-render loop

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      if (checkingInterval) {
        clearInterval(checkingInterval)
        setCheckingInterval(null)
      }
    }
  }, [])

  const handleReady = async () => {
    try {
      setLoading(true)
      setError(null)
      
      console.log('🎮 [MatchFoundDialog] Setting ready for match:', match.id)
      
      const response = await api.post(`/matches/${match.id}/ready`, {
        ready: true
      })
      
      console.log('✅ [MatchFoundDialog] Ready response:', response.data)
      
      // Update ready status từ response
      const blackReady = response.data.black_ready || false
      const whiteReady = response.data.white_ready || false
      const userReadyStatus = userIsBlack ? blackReady : whiteReady
      const opponentReadyStatus = userIsBlack ? whiteReady : blackReady
      const bothReadyNow = response.data.both_ready || false
      
      // Update tất cả states
      setIsReady(userReadyStatus)
      setOpponentReady(opponentReadyStatus)
      setBothReady(bothReadyNow)
      
      console.log('🔄 [MatchFoundDialog] Updated ready status:', {
        userReady: userReadyStatus,
        opponentReady: opponentReadyStatus,
        bothReady: bothReadyNow
      })
      
      // Nếu cả 2 đều ready ngay lập tức, start luôn
      if (bothReadyNow) {
        console.log('🚀 [MatchFoundDialog] Both players ready, starting game')
        if (onStart) {
          onStart(match)
        }
      }
    } catch (error) {
      console.error('❌ [MatchFoundDialog] Error setting ready:', error)
      const errorMessage = error.response?.data?.detail || error.message || 'Không thể set ready status'
      setError(errorMessage)
      
      // Nếu match not found, có thể match đã bị xóa hoặc không tồn tại
      if (error.response?.status === 404) {
        console.error('❌ [MatchFoundDialog] Match not found:', match.id)
        setError(`Match không tồn tại. Vui lòng thử lại hoặc tạo match mới.`)
      }
    } finally {
      setLoading(false)
    }
  }

  const handleCancel = async () => {
    try {
      // Set ready = false
      await api.post(`/matches/${match.id}/ready`, {
        ready: false
      })
    } catch (error) {
      console.error('Error cancelling ready:', error)
    }
    
    if (onCancel) {
      onCancel()
    }
  }

  return (
    <div className="match-found-dialog-overlay">
      <div className="match-found-dialog">
        <div className="match-found-header">
          <div className="match-found-title">
            <FaCheck className="dialog-icon success" />
            <h2>Ghép Trận Thành Công!</h2>
          </div>
        </div>

        <div className="match-found-content">
          <div className="match-found-info">
            <div className="player-info-section">
              <div className="player-info you">
                <div className="player-label">Bạn</div>
                <div className="player-name">{match.user_color === 'B' ? 'Đen' : 'Trắng'}</div>
                <div className="player-status">
                  {isReady ? (
                    <span className="status-ready">
                      <FaCheck /> Đã sẵn sàng
                    </span>
                  ) : (
                    <span className="status-waiting">Chưa sẵn sàng</span>
                  )}
                </div>
              </div>

              <div className="vs-divider">VS</div>

              <div className="player-info opponent">
                <div className="player-label">Đối thủ</div>
                <div className="player-name">{opponentName}</div>
                <div className="player-color">{opponentColor}</div>
                <div className="player-status">
                  {opponentReady ? (
                    <span className="status-ready">
                      <FaCheck /> Đã sẵn sàng
                    </span>
                  ) : (
                    <span className="status-waiting">
                      <FaSpinner className="spinning" /> Đang chờ...
                    </span>
                  )}
                </div>
              </div>
            </div>

            <div className="match-details">
              <div className="detail-item">
                <span className="detail-label">Kích thước bàn cờ:</span>
                <span className="detail-value">{match.board_size}x{match.board_size}</span>
              </div>
              {match.room_code && (
                <div className="detail-item">
                  <span className="detail-label">Mã bàn:</span>
                  <span className="detail-value room-code">{match.room_code}</span>
                </div>
              )}
            </div>

            {error && (
              <div className="error-message">
                {error}
              </div>
            )}

            {bothReady && (
              <div className="both-ready-message">
                <FaCheck /> Cả hai người chơi đã sẵn sàng! Đang bắt đầu trận đấu...
              </div>
            )}
          </div>

          <div className="match-found-actions">
            {!isReady ? (
              <>
                <button
                  type="button"
                  onClick={handleCancel}
                  className="btn btn-secondary"
                  disabled={loading}
                >
                  <FaTimes /> Hủy
                </button>
                <button
                  type="button"
                  onClick={handleReady}
                  className="btn btn-primary"
                  disabled={loading}
                >
                  {loading ? (
                    <>
                      <FaSpinner className="spinning" /> Đang xử lý...
                    </>
                  ) : (
                    <>
                      <FaCheck /> Sẵn sàng
                    </>
                  )}
                </button>
              </>
            ) : (
              <div className="waiting-for-opponent">
                <FaClock /> Đang chờ đối thủ sẵn sàng...
                <button
                  type="button"
                  onClick={handleCancel}
                  className="btn btn-secondary btn-small"
                  disabled={loading}
                >
                  Hủy
                </button>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  )
}

export default MatchFoundDialog

