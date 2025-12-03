import { useState } from 'react'
import { FaLightbulb, FaChartLine, FaEye, FaCoins, FaSpinner } from 'react-icons/fa'
import api from '../services/api'
import './PremiumFeatures.css'

/**
 * Component để hiển thị và xử lý các tính năng premium trong game
 */
const PremiumFeatures = ({ matchId, onHintReceived, onAnalysisReceived, onReviewReceived, disabled = false }) => {
  const [loading, setLoading] = useState({ hint: false, analysis: false, review: false })
  const [error, setError] = useState(null)
  const [hintResult, setHintResult] = useState(null)
  const [analysisRequestId, setAnalysisRequestId] = useState(null)
  const [reviewRequestId, setReviewRequestId] = useState(null)

  const handleHint = async () => {
    if (!matchId) {
      alert('Không tìm thấy ván cờ')
      return
    }

    try {
      setLoading(prev => ({ ...prev, hint: true }))
      setError(null)

      // Lấy board state hiện tại
      const matchResponse = await api.get(`/matches/${matchId}`)
      const boardState = matchResponse.data.state

      const response = await api.post('/premium/hint', {
        match_id: matchId,
        top_k: 3  // Top 3 nước đi tốt nhất
      })

      setHintResult(response.data)
      
      // Dispatch event để CoinDisplay tự động cập nhật
      window.dispatchEvent(new CustomEvent('coinBalanceUpdated'))
      
      if (onHintReceived) {
        onHintReceived(response.data)
      }

      // Hiển thị kết quả
      if (response.data && response.data.hints && response.data.hints.length > 0) {
        const hints = response.data.hints
        const hintMessages = hints.map((hint, idx) => {
          const position = hint.position
          const score = hint.score ? hint.score.toFixed(2) : 'N/A'
          return `${idx + 1}. Vị trí (${position[0]}, ${position[1]}): ${score} điểm`
        }).join('\n')
        
        alert(`💡 Gợi ý nước đi:\n\n${hintMessages}\n\nĐã sử dụng 10 coins`)
      }
    } catch (err) {
      console.error('Hint request failed:', err)
      const errorMsg = err.response?.data?.detail || 'Không thể lấy gợi ý'
      
      if (err.response?.status === 402) {
        alert(`❌ Không đủ coins! Cần 10 coins để sử dụng tính năng này.\n\n${errorMsg}`)
      } else {
        alert(`❌ ${errorMsg}`)
      }
      setError(errorMsg)
    } finally {
      setLoading(prev => ({ ...prev, hint: false }))
    }
  }

  const handleAnalysis = async () => {
    if (!matchId) {
      alert('Không tìm thấy ván cờ')
      return
    }

    try {
      setLoading(prev => ({ ...prev, analysis: true }))
      setError(null)

      const response = await api.post(`/premium/analysis?match_id=${matchId}`)

      // Dispatch event để CoinDisplay tự động cập nhật
      window.dispatchEvent(new CustomEvent('coinBalanceUpdated'))
      
      // Analysis là async, trả về request_id
      if (response.data && response.data.request_id) {
        setAnalysisRequestId(response.data.request_id)
        alert(`📊 Đang phân tích ván cờ...\n\nRequest ID: ${response.data.request_id}\n\nĐã sử dụng 20 coins. Kết quả sẽ được cập nhật khi hoàn thành.`)
        
        // Poll for results
        pollAnalysisResult(response.data.request_id)
      } else {
        alert('✅ Phân tích đã được gửi. Kết quả sẽ được cập nhật khi hoàn thành.')
      }
    } catch (err) {
      console.error('Analysis request failed:', err)
      const errorMsg = err.response?.data?.detail || 'Không thể phân tích ván cờ'
      
      if (err.response?.status === 402) {
        alert(`❌ Không đủ coins! Cần 20 coins để sử dụng tính năng này.\n\n${errorMsg}`)
      } else {
        alert(`❌ ${errorMsg}`)
      }
      setError(errorMsg)
    } finally {
      setLoading(prev => ({ ...prev, analysis: false }))
    }
  }

  const pollAnalysisResult = async (requestId) => {
    const maxAttempts = 30 // 30 attempts
    const interval = 2000 // 2 seconds
    let attempts = 0

    const poll = async () => {
      try {
        const response = await api.get(`/premium/requests/${requestId}`)
        const report = response.data

        if (report.status === 'completed') {
          setAnalysisRequestId(null)
          if (onAnalysisReceived) {
            onAnalysisReceived(report)
          }
          alert('✅ Phân tích hoàn thành! Kiểm tra kết quả trong bảng điều khiển.')
          return true
        } else if (report.status === 'failed') {
          setAnalysisRequestId(null)
          alert(`❌ Phân tích thất bại: ${report.error || 'Unknown error'}`)
          return true
        }

        attempts++
        if (attempts < maxAttempts) {
          setTimeout(poll, interval)
        } else {
          setAnalysisRequestId(null)
          alert('⏱️ Phân tích đang mất nhiều thời gian hơn dự kiến. Vui lòng kiểm tra lại sau.')
        }
      } catch (err) {
        console.error('Poll analysis result failed:', err)
        attempts++
        if (attempts < maxAttempts) {
          setTimeout(poll, interval)
        } else {
          setAnalysisRequestId(null)
          alert('❌ Không thể lấy kết quả phân tích')
        }
      }
    }

    setTimeout(poll, interval)
  }

  const handleReview = async () => {
    if (!matchId) {
      alert('Không tìm thấy ván cờ')
      return
    }

    try {
      setLoading(prev => ({ ...prev, review: true }))
      setError(null)

      const response = await api.post(`/premium/review?match_id=${matchId}`)

      // Dispatch event để CoinDisplay tự động cập nhật
      window.dispatchEvent(new CustomEvent('coinBalanceUpdated'))
      
      // Review là async, trả về request_id
      if (response.data && response.data.request_id) {
        setReviewRequestId(response.data.request_id)
        alert(`🔍 Đang review ván cờ...\n\nRequest ID: ${response.data.request_id}\n\nĐã sử dụng 30 coins. Kết quả sẽ được cập nhật khi hoàn thành.`)
        
        // Poll for results
        pollReviewResult(response.data.request_id)
      } else {
        alert('✅ Review đã được gửi. Kết quả sẽ được cập nhật khi hoàn thành.')
      }
    } catch (err) {
      console.error('Review request failed:', err)
      const errorMsg = err.response?.data?.detail || 'Không thể review ván cờ'
      
      if (err.response?.status === 402) {
        alert(`❌ Không đủ coins! Cần 30 coins để sử dụng tính năng này.\n\n${errorMsg}`)
      } else {
        alert(`❌ ${errorMsg}`)
      }
      setError(errorMsg)
    } finally {
      setLoading(prev => ({ ...prev, review: false }))
    }
  }

  const pollReviewResult = async (requestId) => {
    const maxAttempts = 60 // 60 attempts (2 minutes)
    const interval = 2000 // 2 seconds
    let attempts = 0

    const poll = async () => {
      try {
        const response = await api.get(`/premium/requests/${requestId}`)
        const report = response.data

        if (report.status === 'completed') {
          setReviewRequestId(null)
          if (onReviewReceived) {
            onReviewReceived(report)
          }
          alert('✅ Review hoàn thành! Kiểm tra kết quả trong bảng điều khiển.')
          return true
        } else if (report.status === 'failed') {
          setReviewRequestId(null)
          alert(`❌ Review thất bại: ${report.error || 'Unknown error'}`)
          return true
        }

        attempts++
        if (attempts < maxAttempts) {
          setTimeout(poll, interval)
        } else {
          setReviewRequestId(null)
          alert('⏱️ Review đang mất nhiều thời gian hơn dự kiến. Vui lòng kiểm tra lại sau.')
        }
      } catch (err) {
        console.error('Poll review result failed:', err)
        attempts++
        if (attempts < maxAttempts) {
          setTimeout(poll, interval)
        } else {
          setReviewRequestId(null)
          alert('❌ Không thể lấy kết quả review')
        }
      }
    }

    setTimeout(poll, interval)
  }

  return (
    <div className="premium-features">
      {error && (
        <div className="premium-features-error">
          {error}
        </div>
      )}
      
      <div className="premium-features-buttons">
        <button
          className="premium-feature-btn premium-hint-btn"
          onClick={handleHint}
          disabled={disabled || loading.hint || !matchId}
          title="💡 Gợi ý nước đi thông minh từ AI - Nhận top 3 nước đi tốt nhất với điểm số đánh giá (10 coins)"
        >
          {loading.hint ? (
            <FaSpinner className="spinner" />
          ) : (
            <FaLightbulb />
          )}
          <span className="premium-feature-name">Gợi ý</span>
          <span className="premium-feature-desc">Top 3 nước đi</span>
          <span className="premium-cost">10 <FaCoins /></span>
        </button>

        <button
          className="premium-feature-btn premium-analysis-btn"
          onClick={handleAnalysis}
          disabled={disabled || loading.analysis || !matchId || analysisRequestId !== null}
          title="📊 Phân tích vị trí hiện tại - Đánh giá chiến lược và điểm mạnh/yếu của vị trí (20 coins, async)"
        >
          {loading.analysis || analysisRequestId ? (
            <FaSpinner className="spinner" />
          ) : (
            <FaChartLine />
          )}
          <span className="premium-feature-name">Phân tích</span>
          <span className="premium-feature-desc">Vị trí hiện tại</span>
          <span className="premium-cost">20 <FaCoins /></span>
        </button>

        <button
          className="premium-feature-btn premium-review-btn"
          onClick={handleReview}
          disabled={disabled || loading.review || !matchId || reviewRequestId !== null}
          title="🔍 Review toàn diện - Đánh giá chi tiết toàn bộ ván cờ với phân tích từng nước đi (30 coins, async)"
        >
          {loading.review || reviewRequestId ? (
            <FaSpinner className="spinner" />
          ) : (
            <FaEye />
          )}
          <span className="premium-feature-name">Review</span>
          <span className="premium-feature-desc">Toàn bộ ván cờ</span>
          <span className="premium-cost">30 <FaCoins /></span>
        </button>
      </div>
    </div>
  )
}

export default PremiumFeatures

