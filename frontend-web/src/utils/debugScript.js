// Debug script for browser console
// Usage: Call window.debugGoGame() in browser console

// Get API URL from environment or default
function getApiUrl() {
  try {
    // Try to get from Vite env
    if (typeof import.meta !== 'undefined' && import.meta.env && import.meta.env.VITE_API_URL) {
      return import.meta.env.VITE_API_URL
    }
  } catch (e) {
    // Fallback to checking localStorage for API URL (set by app)
  }
  // Check localStorage for API URL (set by app)
  return localStorage.getItem('API_URL') || 'http://localhost:8000'
}

if (typeof window !== 'undefined') {
  window.debugGoGame = function() {
    const apiUrl = getApiUrl()
    
    console.group('🐛 GoGame Debug Information')
    
    // Authentication
    console.group('🔐 Authentication')
    const token = localStorage.getItem('access_token')
    const refreshToken = localStorage.getItem('refresh_token')
    console.log('Access Token:', token ? `${token.substring(0, 30)}...` : 'None')
    console.log('Refresh Token:', refreshToken ? `${refreshToken.substring(0, 30)}...` : 'None')
    console.log('Token Length:', token ? token.length : 0)
    console.groupEnd()
    
    // API Configuration
    console.group('🌐 API Configuration')
    console.log('API URL:', apiUrl)
    console.groupEnd()
    
    // Test API Connection
    console.group('🧪 API Tests')
    window.testAPI = async function() {
      const apiUrl = getApiUrl()
      try {
        const response = await fetch(`${apiUrl}/health`)
        if (!response.ok) {
          console.error(`❌ /health failed: ${response.status}`)
          return null
        }
        const data = await response.json()
        console.log('✅ /health:', data)
        return data
      } catch (error) {
        console.error('❌ /health failed:', error)
        console.error('💡 Make sure backend is running at', apiUrl)
        return null
      }
    }
    
    window.testMatchesHistory = async function() {
      const apiUrl = getApiUrl()
      try {
        const token = localStorage.getItem('access_token')
        if (!token) {
          console.error('❌ No token found')
          return null
        }
        const response = await fetch(`${apiUrl}/matches/history`, {
          headers: {
            'Authorization': `Bearer ${token}`,
            'Content-Type': 'application/json'
          }
        })
        if (!response.ok) {
          const errorText = await response.text()
          console.error(`❌ /matches/history failed: ${response.status}`, errorText)
          return null
        }
        const data = await response.json()
        console.log('✅ /matches/history:', data)
        return data
      } catch (error) {
        console.error('❌ /matches/history failed:', error)
        return null
      }
    }
    
    window.testStatistics = async function() {
      const apiUrl = getApiUrl()
      try {
        const token = localStorage.getItem('access_token')
        if (!token) {
          console.error('❌ No token found')
          return null
        }
        const response = await fetch(`${apiUrl}/statistics/me`, {
          headers: {
            'Authorization': `Bearer ${token}`,
            'Content-Type': 'application/json'
          }
        })
        if (!response.ok) {
          const errorText = await response.text()
          console.error(`❌ /statistics/me failed: ${response.status}`, errorText)
          return null
        }
        const data = await response.json()
        console.log('✅ /statistics/me:', data)
        return data
      } catch (error) {
        console.error('❌ /statistics/me failed:', error)
        return null
      }
    }
    
    console.log('Available functions:')
    console.log('  - window.testAPI() - Test /health endpoint')
    console.log('  - window.testMatchesHistory() - Test /matches/history')
    console.log('  - window.testStatistics() - Test /statistics/me')
    console.groupEnd()
    
    // Storage
    console.group('💾 Storage')
    console.log('LocalStorage keys:', Object.keys(localStorage))
    console.log('SessionStorage keys:', Object.keys(sessionStorage))
    console.groupEnd()
    
    // Actions
    console.group('⚡ Quick Actions')
    window.clearAuth = function() {
      localStorage.removeItem('access_token')
      localStorage.removeItem('refresh_token')
      console.log('✅ Auth tokens cleared')
      window.location.reload()
    }
    
    window.clearAllStorage = function() {
      localStorage.clear()
      sessionStorage.clear()
      console.log('✅ All storage cleared')
      window.location.reload()
    }
    
    console.log('Available functions:')
    console.log('  - window.clearAuth() - Clear auth tokens and reload')
    console.log('  - window.clearAllStorage() - Clear all storage and reload')
    console.groupEnd()
    
    console.groupEnd()
    console.log('✅ Debug script loaded! Use functions above to test.')
  }
  
  // Auto-run if in development (only if import.meta is available)
  try {
    if (typeof import.meta !== 'undefined' && import.meta.env && import.meta.env.DEV) {
      window.debugGoGame()
    }
  } catch (e) {
    // Ignore if import.meta is not available
  }
}
