import { useState, useEffect } from 'react'
import { FaTimes, FaCoins, FaArrowUp, FaArrowDown, FaSpinner } from 'react-icons/fa'
import api from '../services/api'
import './TransactionHistory.css'

const TransactionHistory = ({ isOpen, onClose }) => {
  const [transactions, setTransactions] = useState([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)

  useEffect(() => {
    if (isOpen) {
      loadTransactions()
    }
  }, [isOpen])

  const loadTransactions = async () => {
    try {
      setLoading(true)
      const response = await api.get('/coins/history')
      setTransactions(response.data)
      setError(null)
    } catch (err) {
      console.error('Failed to load transactions:', err)
      setError('Không thể tải lịch sử giao dịch')
    } finally {
      setLoading(false)
    }
  }

  const formatDate = (dateString) => {
    const date = new Date(dateString)
    return date.toLocaleString('vi-VN', {
      year: 'numeric',
      month: '2-digit',
      day: '2-digit',
      hour: '2-digit',
      minute: '2-digit'
    })
  }

  const getTransactionIcon = (type) => {
    switch (type) {
      case 'purchase':
      case 'earn':
        return <FaArrowUp className="transaction-icon transaction-icon-positive" />
      case 'spend':
        return <FaArrowDown className="transaction-icon transaction-icon-negative" />
      default:
        return <FaCoins className="transaction-icon" />
    }
  }

  const getTransactionLabel = (type, source) => {
    const sourceLabels = {
      'daily_login': 'Đăng nhập hàng ngày',
      'package_starter': 'Gói Starter',
      'package_basic': 'Gói Basic',
      'package_standard': 'Gói Standard',
      'package_premium': 'Gói Premium',
      'package_ultimate': 'Gói Ultimate',
      'premium_subscription_monthly': 'Premium Monthly',
      'premium_subscription_yearly': 'Premium Yearly',
      'premium_hint': 'Gợi ý nước đi',
      'premium_analysis': 'Phân tích ván cờ',
      'premium_review': 'Review ván cờ',
      'complete_game': 'Hoàn thành ván cờ',
      'win_game': 'Thắng ván cờ',
      'rank_up': 'Lên hạng',
      'achievement': 'Thành tích',
      'watch_ad': 'Xem quảng cáo'
    }
    
    const typeLabels = {
      'purchase': 'Mua',
      'earn': 'Nhận',
      'spend': 'Sử dụng'
    }
    
    const sourceLabel = sourceLabels[source] || source
    const typeLabel = typeLabels[type] || type
    
    return `${typeLabel}: ${sourceLabel}`
  }

  if (!isOpen) return null

  return (
    <div className="transaction-history-overlay" onClick={onClose}>
      <div className="transaction-history-dialog" onClick={(e) => e.stopPropagation()}>
        <div className="transaction-history-header">
          <h2>📜 Lịch sử giao dịch</h2>
          <button className="transaction-history-close" onClick={onClose}>
            <FaTimes />
          </button>
        </div>

        <div className="transaction-history-content">
          {error && (
            <div className="transaction-history-error">
              {error}
            </div>
          )}

          {loading ? (
            <div className="transaction-history-loading">
              <FaSpinner className="spinner" />
              <span>Đang tải...</span>
            </div>
          ) : transactions.length === 0 ? (
            <div className="transaction-history-empty">
              <FaCoins className="transaction-empty-icon" />
              <p>Chưa có giao dịch nào</p>
            </div>
          ) : (
            <div className="transaction-history-list">
              {transactions.map((tx) => (
                <div 
                  key={tx.id} 
                  className={`transaction-item ${tx.type}`}
                >
                  <div className="transaction-icon-wrapper">
                    {getTransactionIcon(tx.type)}
                  </div>
                  
                  <div className="transaction-details">
                    <div className="transaction-label">
                      {getTransactionLabel(tx.type, tx.source)}
                    </div>
                    <div className="transaction-date">
                      {formatDate(tx.created_at)}
                    </div>
                  </div>
                  
                  <div className={`transaction-amount ${tx.type}`}>
                    {tx.type === 'spend' ? '-' : '+'}
                    {Math.abs(tx.amount).toLocaleString()}
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      </div>
    </div>
  )
}

export default TransactionHistory

