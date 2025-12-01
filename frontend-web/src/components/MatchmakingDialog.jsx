import { useState, useEffect, useRef } from 'react'
import { FaTimes, FaSearch, FaSpinner, FaClock, FaWindowMinimize } from 'react-icons/fa'
import api from '../services/api'
import './MatchmakingDialog.css'

const MatchmakingDialog = ({ onClose, onMatchFound }) => {
  const [boardSize, setBoardSize] = useState(9)
  const [inQueue, setInQueue] = useState(false)
  const [queueStatus, setQueueStatus] = useState(null)
  const [checkingInterval, setCheckingInterval] = useState(null)
  const [error, setError] = useState(null)
  const [isMinimized, setIsMinimized] = useState(false)
  const [waitTime, setWaitTime] = useState(0)
  const waitTimeRef = useRef(0)
  const waitTimeIntervalRef = useRef(null)

  // Start wait time counter
  useEffect(() => {
    if (inQueue) {
      waitTimeRef.current = 0
      setWaitTime(0)
      
      waitTimeIntervalRef.current = setInterval(() => {
        waitTimeRef.current += 1
        setWaitTime(waitTimeRef.current)
      }, 1000) // Update every second
      
      return () => {
        if (waitTimeIntervalRef.current) {
          clearInterval(waitTimeIntervalRef.current)
          waitTimeIntervalRef.current = null
        }
      }
    } else {
      if (waitTimeIntervalRef.current) {
        clearInterval(waitTimeIntervalRef.current)
        waitTimeIntervalRef.current = null
      }
      setWaitTime(0)
      waitTimeRef.current = 0
    }
  }, [inQueue])

  // Polling để check queue status và match
  useEffect(() => {
    if (inQueue) {
      // Immediate check khi vừa join queue
      const immediateCheck = async () => {
        try {
          const matchRes = await api.get('/matchmaking/queue/match', {
            timeout: 10000
          })
          console.log('🎮 [IMMEDIATE] Match check response:', matchRes.data)
          
          if (matchRes.data && matchRes.data.matched && matchRes.data.match) {
            const matchData = matchRes.data.match
            console.log('✅ [IMMEDIATE] Match found!', matchData)
            if (checkingInterval) {
              clearInterval(checkingInterval)
              setCheckingInterval(null)
            }
            setInQueue(false) // QUAN TRỌNG: Set inQueue = false khi match found
            setQueueStatus(null)
            if (waitTimeIntervalRef.current) {
              clearInterval(waitTimeIntervalRef.current)
              waitTimeIntervalRef.current = null
            }
            onMatchFound(matchData)
            return true // Match found, stop polling
          }
        } catch (error) {
          console.log('ℹ️ [IMMEDIATE] No match yet (this is normal)')
        }
        return false // No match yet, continue polling
      }
      
      // Run immediate check với delay nhỏ để đảm bảo match đã được tạo
      setTimeout(() => {
        immediateCheck()
      }, 500) // Delay 500ms để đảm bảo match đã được commit vào database
      
      const interval = setInterval(async () => {
        try {
          // Check queue status với timeout dài hơn
          const statusRes = await api.get('/matchmaking/queue/status', {
            timeout: 15000 // 15 seconds
          })
          console.log('📊 Queue status response:', statusRes.data)
          if (statusRes.data) {
            // Update queue status
            setQueueStatus(statusRes.data)
            // CHỈ set inQueue = false nếu backend explicitly nói không còn trong queue
            if (statusRes.data.in_queue === false) {
              console.log('⚠️ Backend says not in queue anymore')
              setInQueue(false)
              setQueueStatus(null)
              clearInterval(interval)
              setCheckingInterval(null)
              return
            }
            // Nếu in_queue = true hoặc undefined, giữ nguyên inQueue = true
          } else {
            // No data - có thể là lỗi, nhưng không tự động exit queue
            console.warn('⚠️ Queue status response has no data')
          }

          // Check if match found với timeout dài hơn
          const matchRes = await api.get('/matchmaking/queue/match', {
            timeout: 15000 // 15 seconds
          })
          console.log('🎮 Match check response:', matchRes.data)
          
          // Check response format: { matched: true, match: {...} } hoặc { matched: false }
          if (matchRes.data && matchRes.data.matched && matchRes.data.match) {
            // Match found! Hiển thị MatchFoundDialog thay vì tự động vào trận đấu
            const matchData = matchRes.data.match
            console.log('✅ Match found!', matchData)
            clearInterval(interval)
            setCheckingInterval(null)
            setInQueue(false) // QUAN TRỌNG: Set inQueue = false khi match found
            setQueueStatus(null)
            if (waitTimeIntervalRef.current) {
              clearInterval(waitTimeIntervalRef.current)
              waitTimeIntervalRef.current = null
            }
            // Gọi onMatchFound với match data - HomePage sẽ hiển thị MatchFoundDialog
            onMatchFound(matchData)
          } else if (matchRes.data && matchRes.data.id) {
            // Fallback: nếu response trực tiếp là match object (backward compatibility)
            console.log('✅ Match found! (fallback format)', matchRes.data)
            clearInterval(interval)
            setCheckingInterval(null)
            setInQueue(false) // QUAN TRỌNG: Set inQueue = false khi match found
            setQueueStatus(null)
            if (waitTimeIntervalRef.current) {
              clearInterval(waitTimeIntervalRef.current)
              waitTimeIntervalRef.current = null
            }
            onMatchFound(matchRes.data)
          }
        } catch (error) {
          console.error('❌ Error checking match:', error)
          // Không hiển thị error cho timeout vì đây là polling
          if (error.code === 'ECONNABORTED' || error.message?.includes('timeout')) {
            // Timeout là bình thường khi polling, chỉ log
            console.warn('⏱️ Polling timeout (this is normal)')
            // KHÔNG set inQueue = false khi timeout - có thể chỉ là network issue
          } else if (error.response?.status === 404) {
            // Not in queue anymore hoặc no match yet
            console.log('ℹ️ 404 response - checking if it\'s for status or match endpoint')
            // Chỉ exit queue nếu 404 từ status endpoint, không exit nếu 404 từ match endpoint
            if (error.config?.url?.includes('/queue/status')) {
              console.log('⚠️ Backend returned 404 for status - not in queue anymore')
              setInQueue(false)
              setQueueStatus(null)
              clearInterval(interval)
              setCheckingInterval(null)
            } else {
              // 404 từ match endpoint là bình thường (chưa có match)
              console.log('ℹ️ No match yet (404 is normal for match endpoint)')
            }
          } else if (error.response?.status === 401) {
            // Unauthorized - session expired
            console.error('🔒 Session expired')
            setInQueue(false)
            setQueueStatus(null)
            clearInterval(interval)
            setCheckingInterval(null)
          } else if (error.response?.status === 200 && error.response?.data) {
            // Response có data nhưng có thể format không đúng - log để debug
            console.warn('⚠️ Unexpected response format:', error.response.data)
          } else {
            // Other errors - chỉ log, không tự động exit queue
            console.warn('⚠️ Polling error (not exiting queue):', error.message)
          }
        }
      }, 1000) // Check every 1 second (giảm từ 2 giây để phát hiện match nhanh hơn)

      setCheckingInterval(interval)

      return () => {
        console.log('🧹 Polling cleanup - clearing interval')
        clearInterval(interval)
        setCheckingInterval(null)
      }
    }
  }, [inQueue, onMatchFound, onClose])

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      console.log('🧹 MatchmakingDialog cleanup - clearing intervals')
      if (checkingInterval) {
        clearInterval(checkingInterval)
        setCheckingInterval(null)
      }
      if (waitTimeIntervalRef.current) {
        clearInterval(waitTimeIntervalRef.current)
        waitTimeIntervalRef.current = null
      }
      // CHỈ leave queue khi component unmount (dialog đóng), không leave khi re-render
      // Sử dụng ref để tránh leave queue khi component re-render do state change
      if (inQueue) {
        console.log('🚪 Component unmounting - leaving queue')
        api.post('/matchmaking/queue/leave', {}, {
          timeout: 5000 // 5 seconds for cleanup
        }).catch((err) => {
          console.error('Error leaving queue on unmount:', err)
        })
      }
    }
  }, []) // Empty dependency array - chỉ chạy khi component unmount

  const handleJoinQueue = async () => {
    try {
      setError(null)
      console.log('🔄 Joining matchmaking queue...', { boardSize })
      // Tăng timeout cho join queue vì có thể cần auto-resign matches
      const response = await api.post('/matchmaking/queue/join', {
        board_size: boardSize
      }, {
        timeout: 30000 // 30 seconds - đủ thời gian cho auto-resign
      })
      console.log('✅ Joined queue successfully:', response.data)
      setInQueue(true)
      setQueueStatus(response.data)
    } catch (error) {
      console.error('❌ Error joining queue:', error)
      
      // Kiểm tra network error (backend không chạy)
      if (error.code === 'ECONNREFUSED' || error.code === 'ERR_NETWORK' || 
          (error.request && !error.response)) {
        setError('Không thể kết nối đến server. Vui lòng kiểm tra backend có đang chạy tại http://localhost:8000')
        console.error('💡 Backend có thể không chạy. Hãy chạy: cd backend && uvicorn app.main:app --reload')
        return
      }
      
      if (error.code === 'ECONNABORTED' || error.message?.includes('timeout')) {
        setError('Kết nối quá lâu. Vui lòng thử lại.')
      } else if (error.response) {
        // Server responded with error
        const errorMessage = error.response?.data?.detail || error.message || 'Không thể tham gia queue'
        setError(errorMessage)
      } else {
        // Other errors
        setError('Không thể tham gia queue. Vui lòng thử lại.')
      }
      
      console.error('Error details:', {
        code: error.code,
        status: error.response?.status,
        data: error.response?.data,
        message: error.message,
        request: error.request ? 'Request sent but no response' : 'No request sent'
      })
    }
  }

  const handleLeaveQueue = async () => {
    try {
      await api.post('/matchmaking/queue/leave', {}, {
        timeout: 15000 // 15 seconds
      })
      setInQueue(false)
      setQueueStatus(null)
      if (checkingInterval) {
        clearInterval(checkingInterval)
        setCheckingInterval(null)
      }
      setError(null)
    } catch (error) {
      console.error('Error leaving queue:', error)
      if (error.code === 'ECONNABORTED' || error.message?.includes('timeout')) {
        setError('Kết nối quá lâu. Vui lòng thử lại.')
      } else {
        setError(error.response?.data?.detail || error.message || 'Không thể rời khỏi queue')
      }
      // Vẫn set inQueue = false để UI có thể đóng dialog
      setInQueue(false)
      setQueueStatus(null)
      if (checkingInterval) {
        clearInterval(checkingInterval)
        setCheckingInterval(null)
      }
    }
  }

  const formatTime = (seconds) => {
    const mins = Math.floor(seconds / 60)
    const secs = seconds % 60
    return `${mins}:${secs.toString().padStart(2, '0')}`
  }

  const handleOverlayClick = async (e) => {
    if (e.target === e.currentTarget) {
      if (inQueue) {
        // Nếu đang trong queue, hủy queue trước khi đóng
        await handleLeaveQueue()
      }
      onClose()
    }
  }

  return (
    <div 
      className={`matchmaking-dialog-overlay ${isMinimized ? 'minimized' : ''}`} 
      onClick={handleOverlayClick}
    >
      <div 
        className={`matchmaking-dialog ${isMinimized ? 'minimized' : ''}`} 
        onClick={(e) => e.stopPropagation()}
      >
        <div className="matchmaking-dialog-header">
          <div className="matchmaking-dialog-title">
            <FaSearch className="dialog-icon" />
            <h2>Ghép Người Chơi Online</h2>
          </div>
          <div className="matchmaking-dialog-actions">
            {inQueue && (
              <button
                type="button"
                onClick={() => setIsMinimized(!isMinimized)}
                className="matchmaking-dialog-minimize"
                title={isMinimized ? "Mở rộng" : "Thu nhỏ"}
              >
                <FaWindowMinimize />
              </button>
            )}
            {!inQueue && (
              <button
                type="button"
                onClick={onClose}
                className="matchmaking-dialog-close"
                title="Đóng (Esc)"
              >
                <FaTimes />
              </button>
            )}
          </div>
        </div>

        {!inQueue ? (
          <div className="matchmaking-form">
            <div className="form-group">
              <label>
                <FaSearch className="label-icon" />
                Kích thước bàn cờ:
              </label>
              <select
                value={boardSize}
                onChange={(e) => setBoardSize(parseInt(e.target.value))}
                disabled={inQueue}
              >
                <option value="9">9x9 (Nhanh)</option>
                <option value="13">13x13 (Trung bình)</option>
                <option value="19">19x19 (Chuẩn)</option>
              </select>
            </div>

            {error && (
              <div className="error-message">
                {error}
              </div>
            )}

            <div className="dialog-actions">
              <button type="button" onClick={onClose} className="btn btn-secondary">
                Hủy
              </button>
              <button type="button" onClick={handleJoinQueue} className="btn btn-primary">
                <FaSearch /> Tìm đối thủ
              </button>
            </div>
          </div>
        ) : (
          <div className="matchmaking-queue">
            <div className="queue-status">
              <FaSpinner className="spinner-icon" />
              <h3>Đang tìm đối thủ...</h3>
              <div className="queue-info">
                <div className="queue-info-item">
                  <FaClock className="info-icon" />
                  <span>Thời gian chờ: {formatTime(waitTime)}</span>
                </div>
                <div className="queue-info-item">
                  <span>Người trong queue: {queueStatus?.queue_size || 1}</span>
                </div>
                <div className="queue-info-item">
                  <span>Khoảng ELO: ±{queueStatus?.elo_range || 200}</span>
                </div>
              </div>
            </div>

            <div className="queue-actions">
              <button 
                type="button" 
                onClick={handleLeaveQueue} 
                className="btn btn-secondary"
              >
                Hủy tìm kiếm
              </button>
            </div>
            
            {error && (
              <div className="error-message" style={{ marginTop: '1rem' }}>
                {error}
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  )
}

export default MatchmakingDialog

