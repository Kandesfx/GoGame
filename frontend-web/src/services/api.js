import axios from 'axios'

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000'

console.log('🔗 API URL:', API_URL)

const api = axios.create({
  baseURL: API_URL,
  headers: {
    'Content-Type': 'application/json',
  },
  timeout: 10000, // 10 seconds timeout
})

// Add token to requests if available
const token = localStorage.getItem('access_token')
if (token) {
  api.defaults.headers.common['Authorization'] = `Bearer ${token}`
}

// Request interceptor
api.interceptors.request.use(
  async (config) => {
    // Kiểm tra và refresh token proactively trước mỗi request (chỉ cho protected endpoints)
    if (config.url && !config.url.includes('/auth/')) {
      await checkAndRefreshToken()
    }
    
    const token = localStorage.getItem('access_token')
    if (token) {
      config.headers.Authorization = `Bearer ${token}`
    } else {
      // Log warning if no token for protected endpoints
      if (config.url && !config.url.includes('/auth/')) {
        console.warn('⚠️ No token found for request:', config.method?.toUpperCase(), config.url)
      }
    }
    return config
  },
  (error) => {
    return Promise.reject(error)
  }
)

// Flag để tránh refresh token loop
let isRefreshing = false
let failedQueue = []

const processQueue = (error, token = null) => {
  failedQueue.forEach(prom => {
    if (error) {
      prom.reject(error)
    } else {
      prom.resolve(token)
    }
  })
  failedQueue = []
}

// Hàm decode JWT để lấy expiration time
const decodeJWT = (token) => {
  try {
    const base64Url = token.split('.')[1]
    const base64 = base64Url.replace(/-/g, '+').replace(/_/g, '/')
    const jsonPayload = decodeURIComponent(
      atob(base64)
        .split('')
        .map(c => '%' + ('00' + c.charCodeAt(0).toString(16)).slice(-2))
        .join('')
    )
    return JSON.parse(jsonPayload)
  } catch (e) {
    return null
  }
}

// Hàm kiểm tra và refresh token proactively
const checkAndRefreshToken = async () => {
  const accessToken = localStorage.getItem('access_token')
  const refreshToken = localStorage.getItem('refresh_token')
  
  if (!accessToken || !refreshToken || isRefreshing) {
    return
  }
  
  try {
    const decoded = decodeJWT(accessToken)
    if (!decoded || !decoded.exp) {
      return
    }
    
    // Kiểm tra nếu token còn ít hơn 60 phút (3600 giây) thì refresh
    // Refresh sớm để tránh gián đoạn game session
    const now = Math.floor(Date.now() / 1000)
    const timeUntilExpiry = decoded.exp - now
    
    // Refresh nếu còn ít hơn 60 phút (hoặc đã hết hạn)
    if (timeUntilExpiry < 3600) {
      console.log(`🔄 Token expires in ${Math.floor(timeUntilExpiry / 60)} minutes - refreshing proactively...`)
      isRefreshing = true
      
      try {
        const refreshResponse = await axios.post(`${API_URL}/auth/refresh`, {
          refresh_token: refreshToken
        }, {
          headers: {
            'Content-Type': 'application/json'
          }
        })
        
        const { access_token, refresh_token: newRefreshToken } = refreshResponse.data
        
        localStorage.setItem('access_token', access_token)
        if (newRefreshToken) {
          localStorage.setItem('refresh_token', newRefreshToken)
        }
        
        api.defaults.headers.common['Authorization'] = `Bearer ${access_token}`
        window.dispatchEvent(new CustomEvent('tokenRefreshed', { detail: { access_token } }))
        
        console.log('✅ Token refreshed proactively')
      } catch (error) {
        console.error('❌ Proactive token refresh failed:', error)
        // Không logout ở đây - để reactive refresh handle
      } finally {
        isRefreshing = false
      }
    }
  } catch (e) {
    // Ignore decode errors
  }
}

// Response interceptor for error handling
api.interceptors.response.use(
  (response) => {
    // Log successful requests in development
    if (import.meta.env.DEV) {
      console.log('✅ API Success:', response.config.method?.toUpperCase(), response.config.url, response.status)
    }
    return response
  },
  async (error) => {
    const originalRequest = error.config
    
    // Log error for debugging
    if (error.response) {
      // Server responded with error
      console.error('❌ API Error (Server Response):', {
        url: error.config?.url,
        method: error.config?.method,
        status: error.response.status,
        data: error.response.data,
      })
    } else if (error.request) {
      // Network error - no response
      console.error('❌ API Error (Network):', {
        url: error.config?.url,
        method: error.config?.method,
        message: 'No response from server',
        baseURL: error.config?.baseURL,
        code: error.code,
      })
      
      // Kiểm tra loại network error
      if (error.code === 'ECONNREFUSED' || error.code === 'ERR_NETWORK') {
        console.error('💡 Backend không chạy hoặc không thể kết nối!')
        console.error('   Hãy kiểm tra:')
        console.error('   1. Backend có đang chạy tại', error.config?.baseURL || API_URL, '?')
        console.error('   2. Chạy backend: cd backend && uvicorn app.main:app --reload')
        console.error('   3. Kiểm tra CORS settings nếu backend chạy ở port khác')
      } else {
        console.error('💡 Tip: Check if backend is running at', error.config?.baseURL || API_URL)
        console.error('   Start backend: cd backend && uvicorn app.main:app --reload')
      }
    } else {
      // Request setup error
      console.error('❌ API Error (Request Setup):', error.message)
    }
    
    // Xử lý 401 - Token expired
    if (error.response?.status === 401 && originalRequest && !originalRequest._retry) {
      // Nếu đang refresh, thêm request vào queue
      if (isRefreshing) {
        return new Promise((resolve, reject) => {
          failedQueue.push({ resolve, reject })
        }).then(token => {
            originalRequest.headers.Authorization = `Bearer ${token}`
            return api(originalRequest)
          }).catch(err => {
            return Promise.reject(err)
          })
      }
      
      originalRequest._retry = true
      isRefreshing = true
      
      const refreshToken = localStorage.getItem('refresh_token')
      
      // Nếu không có refresh token, logout ngay
      if (!refreshToken) {
        console.log('🔓 No refresh token - logging out')
        localStorage.removeItem('access_token')
        localStorage.removeItem('refresh_token')
        delete api.defaults.headers.common['Authorization']
        isRefreshing = false
        processQueue(error, null)
        // Không redirect - để backend kiểm soát session
        return Promise.reject(error)
      }
      
      try {
        console.log('🔄 Attempting to refresh access token...')
        // Gọi refresh token endpoint (không dùng api để tránh interceptor loop)
        const refreshResponse = await axios.post(`${API_URL}/auth/refresh`, {
          refresh_token: refreshToken
        }, {
          headers: {
            'Content-Type': 'application/json'
          }
        })
        
        const { access_token, refresh_token: newRefreshToken } = refreshResponse.data
        
        // Cập nhật tokens
        localStorage.setItem('access_token', access_token)
        if (newRefreshToken) {
          localStorage.setItem('refresh_token', newRefreshToken)
        }
        
        // Cập nhật API header
        api.defaults.headers.common['Authorization'] = `Bearer ${access_token}`
        originalRequest.headers.Authorization = `Bearer ${access_token}`
        
        // Dispatch event để AuthContext sync token
        window.dispatchEvent(new CustomEvent('tokenRefreshed', { detail: { access_token } }))
        
        console.log('✅ Token refreshed successfully')
        isRefreshing = false
        processQueue(null, access_token)
        
        // Retry original request với token mới
        return api(originalRequest)
      } catch (refreshError) {
        console.error('❌ Token refresh failed:', refreshError)
        // Refresh token cũng hết hạn hoặc invalid - logout
        localStorage.removeItem('access_token')
        localStorage.removeItem('refresh_token')
        delete api.defaults.headers.common['Authorization']
        isRefreshing = false
        processQueue(refreshError, null)
        // Không redirect - để backend kiểm soát session
        return Promise.reject(refreshError)
      }
    }
    
    return Promise.reject(error)
  }
)

export default api

