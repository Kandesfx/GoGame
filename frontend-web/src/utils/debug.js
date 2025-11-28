// Debug utilities

export const logApiError = (error, context = '') => {
  console.group(`❌ API Error ${context ? `(${context})` : ''}`)
  console.error('Error object:', error)
  if (error.response) {
    console.error('Response status:', error.response.status)
    console.error('Response data:', error.response.data)
    console.error('Response headers:', error.response.headers)
  } else if (error.request) {
    console.error('Request made but no response:', error.request)
  } else {
    console.error('Error message:', error.message)
  }
  console.groupEnd()
}

export const formatApiError = (error) => {
  if (error.response) {
    const data = error.response.data
    if (data.detail) {
      if (typeof data.detail === 'string') {
        return data.detail
      } else if (Array.isArray(data.detail)) {
        return data.detail.map(d => d.msg || d).join(', ')
      }
      return JSON.stringify(data.detail)
    }
    return data.message || `Lỗi máy chủ: ${error.response.status}`
  } else if (error.request) {
    return 'Lỗi kết nối: Không thể kết nối đến máy chủ. Máy chủ có đang chạy không?'
  }
  return error.message || 'Lỗi không xác định'
}

