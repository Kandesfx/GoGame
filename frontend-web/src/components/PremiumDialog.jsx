import { useState, useEffect } from 'react'
import { FaTimes, FaCrown, FaCheck, FaSpinner, FaCalendarAlt } from 'react-icons/fa'
import api from '../services/api'
import './PremiumDialog.css'

const PremiumDialog = ({ isOpen, onClose, onSubscribeSuccess }) => {
  const [plans, setPlans] = useState([])
  const [subscription, setSubscription] = useState(null)
  const [loading, setLoading] = useState(true)
  const [subscribing, setSubscribing] = useState(null)
  const [error, setError] = useState(null)
  const [balance, setBalance] = useState({ coins: 0 })

  useEffect(() => {
    if (isOpen) {
      loadData()
    }
  }, [isOpen])

  const loadData = async () => {
    try {
      setLoading(true)
      const [plansRes, subscriptionRes, balanceRes] = await Promise.all([
        api.get('/premium/subscription/plans'),
        api.get('/premium/subscription/status'),
        api.get('/coins/balance')
      ])
      setPlans(plansRes.data.plans)
      setSubscription(subscriptionRes.data)
      setBalance(balanceRes.data)
      setError(null)
    } catch (err) {
      console.error('Failed to load premium data:', err)
      setError('Không thể tải dữ liệu premium')
    } finally {
      setLoading(false)
    }
  }

  const handleSubscribe = async (planId) => {
    try {
      setSubscribing(planId)
      setError(null)
      
      // Mock payment token (trong production sẽ tích hợp payment gateway)
      const response = await api.post('/premium/subscription/subscribe', {
        plan: planId,
        payment_token: 'mock_token_' + Date.now()
      })
      
      // Refresh data
      const [subscriptionRes, balanceRes] = await Promise.all([
        api.get('/premium/subscription/status'),
        api.get('/coins/balance')
      ])
      setSubscription(subscriptionRes.data)
      setBalance(balanceRes.data)
      
      // Dispatch events để các components tự động cập nhật
      window.dispatchEvent(new CustomEvent('coinBalanceUpdated'))
      window.dispatchEvent(new CustomEvent('premiumStatusUpdated'))
      
      if (onSubscribeSuccess) {
        onSubscribeSuccess(response.data)
      }
      
      // Show success message
      alert(`✅ Đăng ký Premium thành công! Nhận được ${response.data.bonus_coins} bonus coins`)
    } catch (err) {
      console.error('Subscribe failed:', err)
      setError(err.response?.data?.detail || 'Đăng ký Premium thất bại')
    } finally {
      setSubscribing(null)
    }
  }

  const handleCancel = async () => {
    if (!window.confirm('Bạn có chắc muốn hủy Premium? Bạn vẫn có thể dùng đến hết hạn.')) {
      return
    }

    try {
      await api.post('/premium/subscription/cancel')
      await loadData()
      alert('✅ Đã hủy Premium. Bạn vẫn có thể dùng đến hết hạn.')
    } catch (err) {
      console.error('Cancel subscription failed:', err)
      setError(err.response?.data?.detail || 'Hủy Premium thất bại')
    }
  }

  if (!isOpen) return null

  const isPremiumActive = subscription && subscription.is_active
  const expiresAt = subscription ? new Date(subscription.expires_at) : null
  const daysLeft = expiresAt ? Math.ceil((expiresAt - new Date()) / (1000 * 60 * 60 * 24)) : 0

  return (
    <div className="premium-dialog-overlay" onClick={onClose}>
      <div className="premium-dialog" onClick={(e) => e.stopPropagation()}>
        <div className="premium-dialog-header">
          <div className="premium-dialog-title">
            <FaCrown className="premium-title-icon" />
            <h2>Premium Membership</h2>
          </div>
          <button className="premium-dialog-close" onClick={onClose}>
            <FaTimes />
          </button>
        </div>

        <div className="premium-dialog-content">
          {isPremiumActive && (
            <div className="premium-status-active">
              <FaCrown className="premium-status-icon" />
              <div className="premium-status-info">
                <h3>Bạn đang sử dụng Premium</h3>
                <p>
                  Hết hạn: {expiresAt.toLocaleDateString('vi-VN')} 
                  ({daysLeft} ngày còn lại)
                </p>
                <button 
                  className="premium-cancel-btn"
                  onClick={handleCancel}
                >
                  Hủy Premium
                </button>
              </div>
            </div>
          )}

          <div className="premium-benefits">
            <h3>✨ Đặc Quyền Thành Viên Premium</h3>
            <ul>
              <li>💎 <strong>Gợi ý nước đi thông minh:</strong> Nhận gợi ý từ AI với chi phí giảm 50%</li>
              <li>📈 <strong>Phân tích ván cờ chuyên sâu:</strong> Hiểu rõ từng nước đi và chiến lược</li>
              <li>🔬 <strong>Đánh giá toàn diện:</strong> Review chi tiết toàn bộ ván cờ với AI</li>
              <li>🎁 <strong>Quà tặng đăng ký:</strong> Nhận ngay bonus coins khi trở thành thành viên</li>
              <li>⭐ <strong>Hỗ trợ ưu tiên:</strong> Được ưu tiên xử lý mọi yêu cầu và phản hồi</li>
            </ul>
          </div>

          {error && (
            <div className="premium-error">
              {error}
            </div>
          )}

          {loading ? (
            <div className="premium-loading">
              <FaSpinner className="spinner" />
              <span>Đang tải...</span>
            </div>
          ) : (
            <div className="premium-plans">
              {plans.map((plan) => {
                const isSubscribing = subscribing === plan.id
                const isCurrentPlan = subscription && subscription.plan === plan.id && subscription.is_active
                const monthlyPrice = plan.duration_days === 30 
                  ? plan.price_usd 
                  : (plan.price_usd / (plan.duration_days / 30)).toFixed(2)
                
                return (
                  <div 
                    key={plan.id} 
                    className={`premium-plan ${isCurrentPlan ? 'current-plan' : ''} ${isSubscribing ? 'subscribing' : ''}`}
                  >
                    {isCurrentPlan && (
                      <div className="premium-plan-badge">Đang dùng</div>
                    )}
                    
                    <div className="premium-plan-header">
                      <h3>{plan.name}</h3>
                      {plan.bonus_coins > 0 && (
                        <span className="premium-plan-bonus">+{plan.bonus_coins} coins</span>
                      )}
                    </div>
                    
                    <div className="premium-plan-content">
                      <div className="premium-plan-price">
                        <span className="premium-plan-price-main">${plan.price_usd.toFixed(2)}</span>
                        <span className="premium-plan-price-period">
                          / {plan.duration_days === 30 ? 'tháng' : 'năm'}
                        </span>
                      </div>
                      
                      {plan.duration_days === 365 && (
                        <div className="premium-plan-savings">
                          Chỉ ${monthlyPrice}/tháng (tiết kiệm 17%)
                        </div>
                      )}
                      
                      <div className="premium-plan-duration">
                        <FaCalendarAlt />
                        <span>{plan.duration_days} ngày</span>
                      </div>
                      
                      {plan.bonus_coins > 0 && (
                        <div className="premium-plan-bonus-info">
                          🎁 Nhận {plan.bonus_coins.toLocaleString()} bonus coins khi đăng ký
                        </div>
                      )}
                    </div>
                    
                    <button
                      className="premium-plan-btn"
                      onClick={() => handleSubscribe(plan.id)}
                      disabled={isSubscribing || isCurrentPlan}
                    >
                      {isSubscribing ? (
                        <>
                          <FaSpinner className="spinner" />
                          <span>Đang xử lý...</span>
                        </>
                      ) : isCurrentPlan ? (
                        <>
                          <FaCheck />
                          <span>Đang sử dụng</span>
                        </>
                      ) : (
                        <>
                          <FaCrown />
                          <span>Đăng ký ngay</span>
                        </>
                      )}
                    </button>
                  </div>
                )
              })}
            </div>
          )}

          <div className="premium-footer">
            <p className="premium-note">
              🔒 Thanh toán được bảo mật 100%. Bạn có thể hủy đăng ký bất cứ lúc nào mà không mất quyền lợi đến hết hạn.
            </p>
          </div>
        </div>
      </div>
    </div>
  )
}

export default PremiumDialog

