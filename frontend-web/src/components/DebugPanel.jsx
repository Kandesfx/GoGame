import { useState, useEffect } from 'react'
import { useAuth } from '../contexts/AuthContext'
import api from '../services/api'
import './DebugPanel.css'

const DebugPanel = () => {
  const { user, token } = useAuth()
  const [isOpen, setIsOpen] = useState(false)
  const [debugInfo, setDebugInfo] = useState({})
  const [apiTest, setApiTest] = useState(null)

  useEffect(() => {
    // Load debug info
    const info = {
      token: token ? `${token.substring(0, 20)}...` : 'None',
      tokenLength: token?.length || 0,
      localStorageToken: localStorage.getItem('access_token')?.substring(0, 20) + '...' || 'None',
      user: user ? JSON.stringify(user, null, 2) : 'None',
      apiUrl: import.meta.env.VITE_API_URL || 'http://localhost:8000',
    }
    setDebugInfo(info)
  }, [token, user])

  const testApiConnection = async () => {
    setApiTest({ loading: true, error: null, success: false })
    try {
      const response = await api.get('/health')
      setApiTest({ loading: false, success: true, data: response.data })
    } catch (error) {
      setApiTest({ 
        loading: false, 
        success: false, 
        error: error.message,
        details: error.response?.data || error.request
      })
    }
  }

  const testMatchesHistory = async () => {
    setApiTest({ loading: true, error: null, success: false })
    try {
      const response = await api.get('/matches/history')
      setApiTest({ 
        loading: false, 
        success: true, 
        data: { count: response.data?.length || 0, matches: response.data }
      })
    } catch (error) {
      setApiTest({ 
        loading: false, 
        success: false, 
        error: error.message,
        details: error.response?.data || error.request
      })
    }
  }

  const clearStorage = () => {
    localStorage.clear()
    sessionStorage.clear()
    window.location.reload()
  }

  if (!isOpen) {
    return (
      <button 
        className="debug-toggle"
        onClick={() => setIsOpen(true)}
        title="Open Debug Panel"
      >
        🐛
      </button>
    )
  }

  return (
    <div className="debug-panel">
      <div className="debug-header">
        <h3>🐛 Debug Panel</h3>
        <button onClick={() => setIsOpen(false)}>✕</button>
      </div>

      <div className="debug-content">
        <section>
          <h4>Authentication</h4>
          <div className="debug-item">
            <strong>Token (state):</strong> {debugInfo.token}
          </div>
          <div className="debug-item">
            <strong>Token (localStorage):</strong> {debugInfo.localStorageToken}
          </div>
          <div className="debug-item">
            <strong>User:</strong>
            <pre>{debugInfo.user}</pre>
          </div>
        </section>

        <section>
          <h4>API Configuration</h4>
          <div className="debug-item">
            <strong>API URL:</strong> {debugInfo.apiUrl}
          </div>
        </section>

        <section>
          <h4>API Tests</h4>
          <div className="debug-actions">
            <button onClick={testApiConnection} className="btn btn-small">
              Test /health
            </button>
            <button onClick={testMatchesHistory} className="btn btn-small">
              Test /matches/history
            </button>
          </div>
          {apiTest && (
            <div className={`api-test-result ${apiTest.success ? 'success' : 'error'}`}>
              {apiTest.loading && <div>Loading...</div>}
              {apiTest.success && (
                <div>
                  ✅ Success: <pre>{JSON.stringify(apiTest.data, null, 2)}</pre>
                </div>
              )}
              {apiTest.error && (
                <div>
                  ❌ Error: {apiTest.error}
                  {apiTest.details && (
                    <pre>{JSON.stringify(apiTest.details, null, 2)}</pre>
                  )}
                </div>
              )}
            </div>
          )}
        </section>

        <section>
          <h4>Actions</h4>
          <div className="debug-actions">
            <button onClick={clearStorage} className="btn btn-small btn-danger">
              Clear Storage & Reload
            </button>
            <button 
              onClick={() => {
                console.log('🔍 Debug Info:', debugInfo)
                console.log('🔍 API Test:', apiTest)
                console.log('🔍 Auth Context:', { user, token })
              }} 
              className="btn btn-small"
            >
              Log to Console
            </button>
          </div>
        </section>

        <section>
          <h4>Quick Commands</h4>
          <div className="debug-commands">
            <code>window.debugGoGame()</code> - Run debug script
            <br />
            <code>window.testAPI()</code> - Test API connection
            <br />
            <code>window.clearAuth()</code> - Clear auth tokens
          </div>
        </section>
      </div>
    </div>
  )
}

export default DebugPanel

