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

      // Gọi ML analysis endpoint để lấy policy map và best moves
      const response = await api.post('/ml/analyze-position-from-match', {
        match_id: matchId
      })

      console.log('ML Analysis response for hints:', response.data)
      
      // Dispatch event để CoinDisplay tự động cập nhật
      window.dispatchEvent(new CustomEvent('coinBalanceUpdated'))
      
      // Extract hints từ ML analysis response
      // Lấy top moves từ intent.policy_map hoặc best_move
      const hints = []
      
      if (response.data) {
        // Lấy best_move nếu có
        if (response.data.best_move && response.data.best_move.position) {
          const [x, y] = response.data.best_move.position
          hints.push({
            move: [x, y],
            confidence: response.data.best_move.confidence || 0.9,
            is_pass: false
          })
        }
        
        // Lấy top moves từ intent.policy_map
        if (response.data.intent && response.data.intent.heatmap) {
          const policyMap = response.data.intent.heatmap
          const topMoves = []
          
          // Convert policy map thành array of [x, y, probability]
          for (let y = 0; y < policyMap.length; y++) {
            for (let x = 0; x < policyMap[y].length; x++) {
              const prob = policyMap[y][x]
              if (prob > 0.01) { // Chỉ lấy moves có probability > 1%
                topMoves.push({ x, y, prob })
              }
            }
          }
          
          // Sort by probability và lấy top 3
          topMoves.sort((a, b) => b.prob - a.prob)
          const top3 = topMoves.slice(0, 3)
          
          top3.forEach(move => {
            // Tránh trùng với best_move
            if (!hints.some(h => h.move[0] === move.x && h.move[1] === move.y)) {
              hints.push({
                move: [move.x, move.y],
                confidence: move.prob,
                is_pass: false
              })
            }
          })
        }
      }
      
      // Nếu không có hints từ ML, fallback về premium/hint
      if (hints.length === 0) {
        console.log('No hints from ML analysis, falling back to premium/hint')
        try {
          const fallbackResponse = await api.post('/premium/hint', {
            match_id: matchId,
            top_k: 3
          })
          const fallbackHints = fallbackResponse.data?.hints || []
          hints.push(...fallbackHints)
        } catch (fallbackErr) {
          console.warn('Fallback hint also failed:', fallbackErr)
        }
      }
      
      setHintResult({ hints })
      
      if (onHintReceived) {
        if (hints.length > 0) {
          onHintReceived(hints)
        } else {
          console.warn('No hints received')
          alert('⚠️ Không có gợi ý nào được tìm thấy cho vị trí hiện tại')
        }
      }
    } catch (err) {
      console.error('Hint request failed:', err)
      const errorMsg = err.response?.data?.detail || 'Không thể lấy gợi ý'
      
      // Nếu ML analysis không available (503), fallback về premium/hint
      if (err.response?.status === 503) {
        console.log('ML analysis not available, falling back to premium/hint')
        try {
          const fallbackResponse = await api.post('/premium/hint', {
            match_id: matchId,
            top_k: 3
          })
          const fallbackHints = fallbackResponse.data?.hints || []
          if (onHintReceived && fallbackHints.length > 0) {
            onHintReceived(fallbackHints)
          } else {
            alert('⚠️ Không có gợi ý nào được tìm thấy cho vị trí hiện tại')
          }
        } catch (fallbackErr) {
          console.error('Fallback hint failed:', fallbackErr)
          alert(`❌ ${fallbackErr.response?.data?.detail || 'Không thể lấy gợi ý'}`)
        }
      } else if (err.response?.status === 402) {
        alert(`❌ Không đủ coins! Cần 50 coins để sử dụng tính năng này.\n\n${errorMsg}`)
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

      // Gọi ML analysis endpoint để lấy evaluation (đánh giá chiến lược)
      const response = await api.post('/ml/analyze-position-from-match', {
        match_id: matchId
      })

      console.log('ML Analysis response for evaluation:', response.data)

      // Dispatch event để CoinDisplay tự động cập nhật
      window.dispatchEvent(new CustomEvent('coinBalanceUpdated'))
      
      // Extract chỉ phần evaluation từ ML analysis
      if (response.data && response.data.evaluation) {
        const evaluation = response.data.evaluation
        const bestMove = response.data.best_move
        
        console.log('Evaluation data from ML analysis:', evaluation)
        
        if (onAnalysisReceived) {
          // Chỉ gửi evaluation và bestMove, không gửi threats/attacks/intent
          onAnalysisReceived({
            id: 'ml-evaluation',
            feature: 'analysis',
            summary: `Win probability: ${(evaluation.win_probability || 0.5) * 100}%`,
            evaluation: evaluation,
            bestMove: bestMove,
            coins_spent: 20
          })
        }
      } else {
        // Fallback: nếu không có evaluation, thử premium/analysis
        console.log('No evaluation from ML analysis, falling back to premium/analysis')
        try {
          const fallbackResponse = await api.post(`/premium/analysis?match_id=${matchId}`)
          if (fallbackResponse.data && fallbackResponse.data.analysis) {
            const analysisData = fallbackResponse.data.analysis
            if (onAnalysisReceived) {
              onAnalysisReceived({
                id: fallbackResponse.data.request_id || 'unknown',
                feature: 'analysis',
                summary: `Win probability: ${(analysisData.win_probability || 0.5) * 100}%`,
                evaluation: {
                  win_probability: analysisData.win_probability || 0.5,
                  territory_estimate: analysisData.territory_estimate || { black: 0, white: 0 },
                  stone_count: analysisData.stone_count || { black: 0, white: 0 },
                  game_phase: analysisData.game_phase || 'middle'
                },
                bestMove: null,
                coins_spent: 20
              })
            }
          }
        } catch (fallbackErr) {
          console.error('Fallback analysis failed:', fallbackErr)
          alert('❌ Không thể lấy dữ liệu đánh giá từ server')
        }
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

        console.log('Poll analysis result:', report)
        
        // Backend trả về report với structure: {id, feature, summary, details, coins_spent}
        // details chứa analysis data
        if (report && report.details) {
          console.log('Analysis details from DB:', report.details)
          setAnalysisRequestId(null)
          if (onAnalysisReceived) {
            // Pass full report với details
            onAnalysisReceived({
              ...report,
              details: report.details || {}
            })
          }
          // Không cần alert, sẽ hiển thị trong MLAnalysisPanel
          return true
        } else if (report && report.error) {
          setAnalysisRequestId(null)
          alert(`❌ Phân tích thất bại: ${report.error || 'Unknown error'}`)
          return true
        } else if (report) {
          // Report tồn tại nhưng chưa có details - có thể chưa sẵn sàng
          console.log('Report exists but no details yet, waiting...', report)
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

      // Gọi ML analysis endpoint để lấy threats và attacks (mistakes và key moments)
      const response = await api.post('/ml/analyze-position-from-match', {
        match_id: matchId
      })

      console.log('ML Analysis response for review:', response.data)

      // Dispatch event để CoinDisplay tự động cập nhật
      window.dispatchEvent(new CustomEvent('coinBalanceUpdated'))
      
      // Extract review data từ ML analysis response
      const mistakes = []
      const key_moments = []
      
      if (response.data) {
        // Mistakes từ threats - các nhóm quân bị đe dọa
        if (response.data.threats && response.data.threats.regions) {
          response.data.threats.regions.forEach((region, index) => {
            if (region.positions && region.positions.length > 0) {
              // Lấy position đầu tiên của region làm mistake
              const position = Array.isArray(region.positions[0]) 
                ? region.positions[0] 
                : [region.positions[0].x, region.positions[0].y]
              
              mistakes.push({
                move_number: index + 1,
                color: 'B', // Cần xác định từ board state
                position: position,
                eval_delta: -10 - (region.severity || 0.5) * 10,
                severity: (region.severity || 0.5) > 0.7 ? 'major' : 'minor'
              })
            }
          })
        }
        
        // Key moments từ attacks - cơ hội tấn công
        if (response.data.attacks && response.data.attacks.opportunities) {
          response.data.attacks.opportunities.forEach((opp, index) => {
            if (opp.position) {
              const position = Array.isArray(opp.position)
                ? opp.position
                : [opp.position.x, opp.position.y]
              
              key_moments.push({
                move_number: index + 1,
                color: 'B', // Cần xác định từ board state
                position: position,
                eval_delta: (opp.confidence || 0.5) * 20,
                type: 'advantage_gain'
              })
            }
          })
        }
      }
      
      // Nếu không có data từ ML, fallback về premium/review
      if (mistakes.length === 0 && key_moments.length === 0) {
        console.log('No review data from ML analysis, falling back to premium/review')
        const fallbackResponse = await api.post(`/premium/review?match_id=${matchId}`)
        
        if (fallbackResponse.data && fallbackResponse.data.review) {
          const reviewData = fallbackResponse.data.review
          if (onReviewReceived) {
            onReviewReceived({
              id: fallbackResponse.data.request_id || 'unknown',
              feature: 'review',
              summary: `Review: ${reviewData?.statistics?.mistakes_count || 0} mistakes found`,
              details: reviewData || {},
              coins_spent: 30
            })
          }
          return
        }
      }
      
      // Tạo review data từ ML analysis
      const reviewData = {
        mistakes: mistakes.slice(0, 10), // Top 10
        key_moments: key_moments.slice(0, 5), // Top 5
        statistics: {
          total_moves: 0, // Cần lấy từ game history
          mistakes_count: mistakes.length,
          key_moments_count: key_moments.length,
          black_mistakes: mistakes.filter(m => m.color === 'B').length,
          white_mistakes: mistakes.filter(m => m.color === 'W').length
        }
      }
      
      console.log('Review data from ML analysis:', reviewData)
      
      if (onReviewReceived) {
        try {
          onReviewReceived({
            id: 'ml-analysis-review',
            feature: 'review',
            summary: `Review: ${reviewData.statistics.mistakes_count} mistakes found`,
            details: reviewData,
            coins_spent: 30
          })
        } catch (error) {
          console.error('Error calling onReviewReceived:', error)
          alert('❌ Có lỗi khi xử lý dữ liệu review')
        }
      }
    } catch (err) {
      console.error('Review request failed:', err)
      const errorMsg = err.response?.data?.detail || 'Không thể review ván cờ'
      
      // Nếu ML analysis không available (503), fallback về premium/review
      if (err.response?.status === 503) {
        console.log('ML analysis not available, falling back to premium/review')
        try {
          const fallbackResponse = await api.post(`/premium/review?match_id=${matchId}`)
          if (fallbackResponse.data && fallbackResponse.data.review) {
            const reviewData = fallbackResponse.data.review
            if (onReviewReceived) {
              onReviewReceived({
                id: fallbackResponse.data.request_id || 'unknown',
                feature: 'review',
                summary: `Review: ${reviewData?.statistics?.mistakes_count || 0} mistakes found`,
                details: reviewData || {},
                coins_spent: 30
              })
            }
          }
        } catch (fallbackErr) {
          console.error('Fallback review failed:', fallbackErr)
          alert(`❌ ${fallbackErr.response?.data?.detail || 'Không thể review ván cờ'}`)
        }
      } else if (err.response?.status === 402) {
        alert(`❌ Không đủ coins! Cần 50 coins để sử dụng tính năng này.\n\n${errorMsg}`)
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

        console.log('Poll review result:', report)
        
        // Backend trả về report với structure: {id, feature, summary, details, coins_spent}
        // details chứa: {mistakes: [], key_moments: [], statistics: {}}
        if (report && report.details) {
          console.log('Review details from DB:', report.details)
          setReviewRequestId(null)
          if (onReviewReceived) {
            // Pass full report với details
            onReviewReceived({
              ...report,
              details: report.details || {}
            })
          }
          // Không cần alert, sẽ hiển thị trong ReviewPanel
          return true
        } else if (report && report.error) {
          setReviewRequestId(null)
          alert(`❌ Review thất bại: ${report.error || 'Unknown error'}`)
          return true
        } else if (report) {
          // Report tồn tại nhưng không có details - có thể chưa sẵn sàng
          console.log('Report exists but no details yet, waiting...', report)
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

