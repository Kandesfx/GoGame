import { createContext, useContext, useState, useEffect } from 'react'
import api from '../services/api'

const AuthContext = createContext()

export const useAuth = () => {
  const context = useContext(AuthContext)
  if (!context) {
    throw new Error('useAuth must be used within AuthProvider')
  }
  return context
}

export const AuthProvider = ({ children }) => {
  // Initialize token from localStorage
  const [user, setUser] = useState(null)
  const [token, setToken] = useState(() => {
    // Initialize from localStorage on mount
    return localStorage.getItem('access_token')
  })

  // Sync token with localStorage and API headers
  useEffect(() => {
    if (token) {
      localStorage.setItem('access_token', token)
      api.defaults.headers.common['Authorization'] = `Bearer ${token}`
      
      // Fetch user info if not already loaded
      if (!user) {
        api.get('/users/me')
          .then(response => {
            if (response.data) {
              setUser({ username: response.data.username, ...response.data })
            }
          })
          .catch(error => {
            console.error('Failed to fetch user info:', error)
            // If token is invalid, clear it
            if (error.response?.status === 401) {
              setToken(null)
              setUser(null)
            }
          })
      }
    } else {
      localStorage.removeItem('access_token')
      delete api.defaults.headers.common['Authorization']
      setUser(null)
    }
  }, [token])
  
  // Listen for token updates from api interceptor (when auto-refresh happens)
  useEffect(() => {
    const handleTokenRefreshed = (e) => {
      const newToken = e.detail?.access_token
      if (newToken && newToken !== token) {
        console.log('🔄 Token refreshed, updating AuthContext state...')
        setToken(newToken)
      }
    }
    
    const handleStorageChange = (e) => {
      if (e.key === 'access_token') {
        const newToken = localStorage.getItem('access_token')
        if (newToken && newToken !== token) {
          console.log('🔄 Token updated from storage, syncing state...')
          setToken(newToken)
        } else if (!newToken && token) {
          // Token was removed
          setToken(null)
          setUser(null)
        }
      }
    }
    
    window.addEventListener('tokenRefreshed', handleTokenRefreshed)
    window.addEventListener('storage', handleStorageChange)
    
    return () => {
      window.removeEventListener('tokenRefreshed', handleTokenRefreshed)
      window.removeEventListener('storage', handleStorageChange)
    }
  }, [token])

  const login = async (username, password) => {
    try {
      console.log('📤 Login request to:', '/auth/login')
      const response = await api.post('/auth/login', {
        username_or_email: username,
        password: password,
      })
      
      console.log('📥 Login response:', response.status, response.data)
      
      // Check response structure
      if (!response.data || !response.data.token) {
        console.error('❌ Invalid response structure:', response.data)
        throw new Error('Invalid response from server: missing token')
      }
      
      const tokenData = response.data.token
      console.log('🔑 Token data:', tokenData)
      
      if (!tokenData.access_token) {
        console.error('❌ Missing access_token in response:', tokenData)
        throw new Error('Invalid response from server: missing access_token')
      }
      
      const accessToken = tokenData.access_token
      const refreshToken = tokenData.refresh_token
      
      console.log('💾 Saving tokens to localStorage...')
      localStorage.setItem('access_token', accessToken)
      if (refreshToken) {
        localStorage.setItem('refresh_token', refreshToken)
      }
      
      console.log('✅ Setting token state...')
      setToken(accessToken)
      setUser(response.data)
      
      // Verify token was saved
      const savedToken = localStorage.getItem('access_token')
      console.log('🔍 Verification - token in localStorage:', savedToken ? '✅ Present' : '❌ Missing')
      
      return response.data
    } catch (error) {
      console.error('❌ Login error:', error)
      throw error
    }
  }

  const register = async (username, email, password) => {
    const apiUrl = import.meta.env.VITE_API_URL || 'http://localhost:8000'
    
    try {
      console.log('📤 Registering to:', `${apiUrl}/auth/register`)
      const response = await api.post('/auth/register', {
        username,
        email,
        password,
      })
      
      console.log('✅ Registration response:', response.status, response.data)
      
      // Check response structure
      if (!response.data || !response.data.token) {
        console.error('Invalid response structure:', response.data)
        throw new Error('Invalid response from server')
      }
      
      const { token: accessToken, refresh_token } = response.data.token
      localStorage.setItem('access_token', accessToken)
      localStorage.setItem('refresh_token', refresh_token)
      setToken(accessToken)
      setUser(response.data)
      return response.data
    } catch (error) {
      console.error('❌ Register error in AuthContext:', error)
      
      // Re-throw with more context
      if (error.response) {
        // Server responded with error
        console.error('Server error:', error.response.status, error.response.data)
        throw error
      } else if (error.request) {
        // Network error - no response
        console.error('Network error - no response from server')
        console.error('Request config:', error.config)
        throw new Error(
          `Network error: Could not reach server at ${apiUrl}. ` +
          `Please ensure backend is running: cd backend && uvicorn app.main:app --reload`
        )
      } else {
        // Other error
        console.error('Other error:', error.message)
        throw error
      }
    }
  }

  const logout = () => {
    console.log('🔓 Logging out...')
    try {
      localStorage.removeItem('access_token')
      localStorage.removeItem('refresh_token')
      setToken(null)
      setUser(null)
      // Clear API authorization header
      delete api.defaults.headers.common['Authorization']
      console.log('✅ Logout successful')
    } catch (error) {
      console.error('❌ Logout error:', error)
    }
  }

  return (
    <AuthContext.Provider value={{ user, token, login, register, logout }}>
      {children}
    </AuthContext.Provider>
  )
}

