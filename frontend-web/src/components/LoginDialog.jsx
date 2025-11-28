import { useState } from 'react'
import { useAuth } from '../contexts/AuthContext'
import { formatApiError, logApiError } from '../utils/debug'
import './LoginDialog.css'

const LoginDialog = () => {
  const [activeTab, setActiveTab] = useState('login')
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')
  const [successMessage, setSuccessMessage] = useState('')
  
  // Login form
  const [loginUsername, setLoginUsername] = useState('')
  const [loginPassword, setLoginPassword] = useState('')
  
  // Register form
  const [registerUsername, setRegisterUsername] = useState('')
  const [registerEmail, setRegisterEmail] = useState('')
  const [registerPassword, setRegisterPassword] = useState('')

  const { login, register } = useAuth()

  const handleLogin = async (e) => {
    e.preventDefault()
    
    // Prevent multiple submissions
    if (loading) {
      console.warn('⚠️ Login already in progress, ignoring duplicate request')
      return
    }
    
    setLoading(true)
    setError('')
    
    try {
      console.log('🔐 Attempting login...')
      await login(loginUsername, loginPassword)
      console.log('✅ Login successful - state will update automatically')
      // Show success message
      setSuccessMessage('Đăng nhập thành công! 🎉')
      setError('')
      // Clear form
      setLoginUsername('')
      setLoginPassword('')
      // Clear success message after 3 seconds
      setTimeout(() => {
        setSuccessMessage('')
      }, 3000)
      // No need to call onLogin() - AuthContext will update and App will re-render
    } catch (err) {
      logApiError(err, 'Login')
      setError(formatApiError(err))
      setSuccessMessage('')
    } finally {
      setLoading(false)
    }
  }

  const handleRegister = async (e) => {
    e.preventDefault()
    
    // Prevent multiple submissions
    if (loading) {
      console.warn('⚠️ Registration already in progress, ignoring duplicate request')
      return
    }
    
    setLoading(true)
    setError('')
    
    // Client-side validation
    if (registerUsername.length < 3) {
      setError('Tên đăng nhập phải có ít nhất 3 ký tự')
      setLoading(false)
      return
    }
    if (registerUsername.length > 32) {
      setError('Tên đăng nhập tối đa 32 ký tự')
      setLoading(false)
      return
    }
    if (registerPassword.length < 8) {
      setError('Mật khẩu phải có ít nhất 8 ký tự')
      setLoading(false)
      return
    }
    if (!registerEmail.includes('@') || !registerEmail.includes('.')) {
      setError('Vui lòng nhập địa chỉ email hợp lệ')
      setLoading(false)
      return
    }
    
    try {
      console.log('📝 Attempting registration...')
      await register(registerUsername, registerEmail, registerPassword)
      console.log('✅ Registration successful - state will update automatically')
      // Show success message
      setSuccessMessage('Đăng ký thành công! Chào mừng bạn! 🎊')
      setError('')
      // Clear form
      setRegisterUsername('')
      setRegisterEmail('')
      setRegisterPassword('')
      // Clear success message after 3 seconds
      setTimeout(() => {
        setSuccessMessage('')
      }, 3000)
      // No need to call onLogin() - AuthContext will update and App will re-render
    } catch (err) {
      logApiError(err, 'Register')
      const errorMsg = formatApiError(err)
      console.error('Registration failed:', errorMsg)
      setError(errorMsg)
      setSuccessMessage('')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="login-dialog">
      <div className="login-container">
        <h1>GoGame - 囲碁</h1>
        
        <div className="tabs">
          <button
            className={activeTab === 'login' ? 'active' : ''}
            onClick={() => {
              setActiveTab('login')
              setError('')
            }}
          >
            Đăng nhập
          </button>
          <button
            className={activeTab === 'register' ? 'active' : ''}
            onClick={() => {
              setActiveTab('register')
              setError('')
            }}
          >
            Đăng ký
          </button>
        </div>

        {error && <div className="error-message">{error}</div>}
        {successMessage && <div className="success-message">{successMessage}</div>}

        {activeTab === 'login' ? (
          <form onSubmit={handleLogin} className="login-form">
            <div className="form-group">
              <label>Tên đăng nhập/Email:</label>
              <input
                type="text"
                value={loginUsername}
                onChange={(e) => setLoginUsername(e.target.value)}
                required
                disabled={loading}
              />
            </div>
            <div className="form-group">
              <label>Mật khẩu:</label>
              <input
                type="password"
                value={loginPassword}
                onChange={(e) => setLoginPassword(e.target.value)}
                required
                disabled={loading}
              />
            </div>
            <button type="submit" disabled={loading} className="btn btn-primary">
              {loading ? 'Đang đăng nhập...' : '🔐 Đăng nhập'}
            </button>
            {loading && <div style={{ fontSize: '12px', color: '#666', marginTop: '8px' }}>Vui lòng đợi...</div>}
          </form>
        ) : (
          <form onSubmit={handleRegister} className="login-form">
            <div className="form-group">
              <label>Tên đăng nhập:</label>
              <input
                type="text"
                value={registerUsername}
                onChange={(e) => setRegisterUsername(e.target.value)}
                required
                disabled={loading}
                minLength={3}
                maxLength={32}
              />
            </div>
            <div className="form-group">
              <label>Email:</label>
              <input
                type="email"
                value={registerEmail}
                onChange={(e) => setRegisterEmail(e.target.value)}
                required
                disabled={loading}
              />
            </div>
            <div className="form-group">
              <label>Mật khẩu:</label>
              <input
                type="password"
                value={registerPassword}
                onChange={(e) => setRegisterPassword(e.target.value)}
                required
                disabled={loading}
                minLength={8}
              />
            </div>
            <button type="submit" disabled={loading} className="btn btn-primary">
              {loading ? 'Đang đăng ký...' : '✨ Đăng ký'}
            </button>
            {loading && <div style={{ fontSize: '12px', color: '#666', marginTop: '8px' }}>Vui lòng đợi...</div>}
          </form>
        )}
      </div>
    </div>
  )
}

export default LoginDialog

