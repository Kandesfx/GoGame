import { useState, useEffect } from 'react'
import { FaTimes, FaCoins, FaCheck, FaSpinner, FaGem, FaStar, FaGift } from 'react-icons/fa'
import api from '../services/api'
import './ShopDialog.css'

const ShopDialog = ({ isOpen, onClose, onPurchaseSuccess }) => {
  const [packages, setPackages] = useState([])
  const [loading, setLoading] = useState(true)
  const [purchasing, setPurchasing] = useState(null)
  const [error, setError] = useState(null)
  const [balance, setBalance] = useState({ coins: 0, has_daily_bonus: false })
  const [claimingBonus, setClaimingBonus] = useState(false)

  useEffect(() => {
    if (isOpen) {
      loadData()
    }
  }, [isOpen])

  const loadData = async () => {
    try {
      setLoading(true)
      const [packagesRes, balanceRes] = await Promise.all([
        api.get('/coins/packages'),
        api.get('/coins/balance')
      ])
      setPackages(packagesRes.data.packages)
      setBalance(balanceRes.data)
      setError(null)
    } catch (err) {
      console.error('Failed to load shop data:', err)
      setError('Không thể tải dữ liệu shop')
    } finally {
      setLoading(false)
    }
  }

  const handleClaimDailyBonus = async () => {
    try {
      setClaimingBonus(true)
      setError(null)
      const response = await api.post('/coins/daily-bonus')
      
      // Refresh balance after claiming
      const balanceRes = await api.get('/coins/balance')
      setBalance(balanceRes.data)
      
      // Dispatch event để CoinDisplay tự động cập nhật
      window.dispatchEvent(new CustomEvent('coinBalanceUpdated'))
      
      // Show success message
      if (response.data && response.data.coins_added) {
        alert(`🎉 Nhận được ${response.data.coins_added} coins từ daily bonus!`)
      }
    } catch (err) {
      console.error('Failed to claim daily bonus:', err)
      setError(err.response?.data?.detail || 'Không thể nhận daily bonus')
    } finally {
      setClaimingBonus(false)
    }
  }

  const handlePurchase = async (packageId) => {
    try {
      setPurchasing(packageId)
      setError(null)
      
      // Mock payment token (trong production sẽ tích hợp payment gateway)
      const response = await api.post('/coins/purchase', {
        package_id: packageId,
        payment_token: 'mock_token_' + Date.now()
      })
      
      // Refresh balance
      const balanceRes = await api.get('/coins/balance')
      setBalance(balanceRes.data)
      
      // Dispatch event để CoinDisplay tự động cập nhật
      window.dispatchEvent(new CustomEvent('coinBalanceUpdated'))
      
      if (onPurchaseSuccess) {
        onPurchaseSuccess(response.data)
      }
      
      // Show success message
      alert(`✅ Mua thành công! Nhận được ${response.data.coins_added} coins`)
    } catch (err) {
      console.error('Purchase failed:', err)
      setError(err.response?.data?.detail || 'Mua coins thất bại')
    } finally {
      setPurchasing(null)
    }
  }

  if (!isOpen) return null

  return (
    <div className="shop-dialog-overlay" onClick={onClose}>
      <div className="shop-dialog" onClick={(e) => e.stopPropagation()}>
        <div className="shop-dialog-header">
          <div className="shop-dialog-title">
            <FaCoins className="shop-title-icon" />
            <h2>Cửa Hàng Coins</h2>
          </div>
          <button className="shop-dialog-close" onClick={onClose}>
            <FaTimes />
          </button>
        </div>

        <div className="shop-dialog-content">
          <div className="shop-balance-section">
            <div className="shop-balance">
              <FaCoins className="shop-balance-icon" />
              <span>Số dư: <strong>{balance.coins.toLocaleString()}</strong> coins</span>
            </div>
            
            {/* Daily Bonus Section */}
            <div className="shop-daily-bonus-section">
              {balance.has_daily_bonus ? (
                <button 
                  className="shop-daily-bonus-btn" 
                  onClick={handleClaimDailyBonus}
                  disabled={claimingBonus}
                  title="Nhận daily bonus"
                  type="button"
                >
                  {claimingBonus ? (
                    <>
                      <FaSpinner className="spinner" />
                      <span>Đang nhận...</span>
                    </>
                  ) : (
                    <>
                      <FaGift />
                      <span>Nhận Daily Bonus</span>
                    </>
                  )}
                </button>
              ) : (
                <div className="shop-daily-bonus-message">
                  <FaGift className="shop-daily-bonus-message-icon" />
                  <span>Phần thưởng hôm nay đã nhận, hãy quay lại vào ngày mai</span>
                </div>
              )}
            </div>
          </div>

          {error && (
            <div className="shop-error">
              {error}
            </div>
          )}

          {loading ? (
            <div className="shop-loading">
              <FaSpinner className="spinner" />
              <span>Đang tải...</span>
            </div>
          ) : (
            <div className="shop-packages">
              {packages.map((pkg) => {
                const totalCoins = pkg.coins + pkg.bonus_coins
                const isPurchasing = purchasing === pkg.id
                
                return (
                  <div 
                    key={pkg.id} 
                    className={`shop-package ${isPurchasing ? 'purchasing' : ''}`}
                  >
                    <div className="shop-package-header">
                      <h3>{pkg.name}</h3>
                      {pkg.bonus_coins > 0 && (
                        <span className="shop-package-badge">+{pkg.bonus_coins} bonus</span>
                      )}
                    </div>
                    
                    <div className="shop-package-content">
                      <div className="shop-package-coins">
                        <FaCoins className="shop-package-icon" />
                        <span className="shop-package-amount">{totalCoins.toLocaleString()}</span>
                        <span className="shop-package-label">coins</span>
                      </div>
                      
                      <div className="shop-package-price">
                        ${pkg.price_usd.toFixed(2)}
                      </div>
                      
                      {pkg.bonus_coins > 0 && (
                        <div className="shop-package-breakdown">
                          <span>{pkg.coins.toLocaleString()} coins</span>
                          <span className="shop-package-plus">+</span>
                          <span className="shop-package-bonus">{pkg.bonus_coins.toLocaleString()} bonus</span>
                        </div>
                      )}
                    </div>
                    
                    <button
                      className="shop-package-btn"
                      onClick={() => handlePurchase(pkg.id)}
                      disabled={isPurchasing}
                    >
                      {isPurchasing ? (
                        <>
                          <FaSpinner className="spinner" />
                          <span>Đang xử lý...</span>
                        </>
                      ) : (
                        <>
                          <FaCheck />
                          <span>Mua ngay</span>
                        </>
                      )}
                    </button>
                  </div>
                )
              })}
            </div>
          )}

          <div className="shop-footer">
            <p className="shop-note">
              💎 Coins là đơn vị tiền tệ trong game, cho phép bạn sử dụng các tính năng premium như gợi ý nước đi thông minh, phân tích ván cờ chuyên sâu và đánh giá toàn diện. Mua ngay để nâng cao trải nghiệm chơi cờ của bạn!
            </p>
          </div>
        </div>
      </div>
    </div>
  )
}

export default ShopDialog

