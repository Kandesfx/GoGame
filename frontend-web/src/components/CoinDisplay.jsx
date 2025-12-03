import { useState, useEffect } from 'react'
import { FaCoins, FaPlus, FaGift, FaSpinner } from 'react-icons/fa'
import api from '../services/api'
import './CoinDisplay.css'

const CoinDisplay = ({ onShopClick, showShopButton = true, onBalanceUpdate }) => {
  const [balance, setBalance] = useState({ coins: 0, has_daily_bonus: false })
  const [loading, setLoading] = useState(true)
  const [claimingBonus, setClaimingBonus] = useState(false)
  const [error, setError] = useState(null)

  useEffect(() => {
    loadBalance()
    
    // Refresh balance mỗi 30 giây
    const interval = setInterval(loadBalance, 30000)
    
    // Listen for coin balance updates from other components
    const handleCoinUpdate = () => {
      loadBalance()
    }
    
    window.addEventListener('coinBalanceUpdated', handleCoinUpdate)
    
    return () => {
      clearInterval(interval)
      window.removeEventListener('coinBalanceUpdated', handleCoinUpdate)
    }
  }, [])

  const loadBalance = async () => {
    try {
      const response = await api.get('/coins/balance')
      setBalance(response.data)
      setError(null)
      if (onBalanceUpdate) {
        onBalanceUpdate(response.data)
      }
    } catch (err) {
      console.error('Failed to load coin balance:', err)
      setError(err)
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
      await loadBalance()
      
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

  if (loading) {
    return (
      <div className="coin-display">
        <FaCoins className="coin-icon" />
        <span className="coin-amount">...</span>
      </div>
    )
  }

  return (
    <div className="coin-display">
      <FaCoins className="coin-icon" />
      <span className="coin-amount">{balance.coins.toLocaleString()}</span>
      
      {/* Daily Bonus Button */}
      {balance.has_daily_bonus && (
        <button 
          className="coin-daily-bonus-btn" 
          onClick={handleClaimDailyBonus}
          disabled={claimingBonus}
          title="Nhận daily bonus"
          type="button"
        >
          {claimingBonus ? (
            <FaSpinner className="spinner" />
          ) : (
            <FaGift />
          )}
        </button>
      )}
      
      {/* Shop Button */}
      {showShopButton && onShopClick && (
        <button 
          className="coin-shop-btn" 
          onClick={onShopClick}
          title="Mua coins"
          type="button"
        >
          <FaPlus />
        </button>
      )}
    </div>
  )
}

export default CoinDisplay

