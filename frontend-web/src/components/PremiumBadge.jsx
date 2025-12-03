import { useState, useEffect } from 'react'
import { FaCrown, FaTimes } from 'react-icons/fa'
import api from '../services/api'
import './PremiumBadge.css'

const PremiumBadge = ({ onPremiumClick, showButton = true }) => {
  const [subscription, setSubscription] = useState(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    loadSubscriptionStatus()
    
    // Refresh mỗi 60 giây
    const interval = setInterval(loadSubscriptionStatus, 60000)
    
    // Listen for premium status updates from other components
    const handlePremiumUpdate = () => {
      loadSubscriptionStatus()
    }
    
    window.addEventListener('premiumStatusUpdated', handlePremiumUpdate)
    
    return () => {
      clearInterval(interval)
      window.removeEventListener('premiumStatusUpdated', handlePremiumUpdate)
    }
  }, [])

  const loadSubscriptionStatus = async () => {
    try {
      const response = await api.get('/premium/subscription/status')
      setSubscription(response.data)
    } catch (err) {
      console.error('Failed to load subscription status:', err)
      setSubscription(null)
    } finally {
      setLoading(false)
    }
  }

  if (loading) {
    return null
  }

  if (!subscription || !subscription.is_active) {
    if (!showButton) return null
    
    return (
      <button 
        className="premium-badge premium-badge-inactive"
        onClick={onPremiumClick}
        title="Nâng cấp Premium"
      >
        <FaCrown className="premium-icon" />
        <span>Premium</span>
      </button>
    )
  }

  // Premium active
  const expiresAt = new Date(subscription.expires_at)
  const daysLeft = Math.ceil((expiresAt - new Date()) / (1000 * 60 * 60 * 24))

  return (
    <div 
      className="premium-badge premium-badge-active"
      onClick={onPremiumClick}
      title={`Premium còn ${daysLeft} ngày`}
    >
      <FaCrown className="premium-icon premium-icon-active" />
      <span className="premium-text-active">Premium</span>
      {daysLeft <= 7 && (
        <span className="premium-days-left">{daysLeft}d</span>
      )}
      <span className="premium-checkmark">✓</span>
    </div>
  )
}

export default PremiumBadge

