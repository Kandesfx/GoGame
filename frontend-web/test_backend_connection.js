// Quick test script to check backend connection
// Run: node test_backend_connection.js

const API_URL = process.env.VITE_API_URL || 'http://localhost:8000'

async function testConnection() {
  console.log('🔍 Testing backend connection...')
  console.log('📍 API URL:', API_URL)
  console.log('')

  try {
    // Test health endpoint
    const response = await fetch(`${API_URL}/health`)
    const data = await response.json()
    
    if (response.ok) {
      console.log('✅ Backend is running!')
      console.log('   Response:', data)
      return true
    } else {
      console.log('❌ Backend responded with error:', response.status)
      return false
    }
  } catch (error) {
    console.log('❌ Cannot connect to backend')
    console.log('   Error:', error.message)
    console.log('')
    console.log('💡 Please start backend:')
    console.log('   cd backend')
    console.log('   uvicorn app.main:app --reload')
    return false
  }
}

testConnection()

