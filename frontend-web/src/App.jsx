import { useEffect, useState } from 'react'
import LoginDialog from './components/LoginDialog'
import HomePage from './components/HomePage'
import MainWindow from './components/MainWindow'
import DebugPanel from './components/DebugPanel'
import VideoBackground from './components/VideoBackground'
import { AuthProvider, useAuth } from './contexts/AuthContext'
import './App.css'
import './utils/debugScript'

function AppContent() {
  const { token, logout } = useAuth()
  const isAuthenticated = !!token
  const [currentView, setCurrentView] = useState('home') // 'home' or 'game'
  const [currentMatch, setCurrentMatch] = useState(null)

  // Debug: Log authentication state changes (only in development)
  useEffect(() => {
    if (import.meta.env.DEV) {
      console.log('🔍 AppContent - Authentication state:', {
        hasToken: !!token,
        tokenLength: token?.length,
      })
    }
  }, [token])

  const handleLogout = () => {
    if (import.meta.env.DEV) {
      console.log('🔓 handleLogout called')
    }
    try {
      logout()
      setCurrentView('home')
      setCurrentMatch(null)
    } catch (error) {
      console.error('❌ Logout error:', error)
    }
  }

  const handleStartMatch = (match) => {
    setCurrentMatch(match)
    setCurrentView('game')
  }

  const handleBackToHome = () => {
    setCurrentView('home')
    setCurrentMatch(null)
  }

  return (
    <div className="App">
      <VideoBackground />
      <div className="app-content" style={{ position: 'relative', zIndex: 1 }}>
        {!isAuthenticated ? (
          <LoginDialog />
        ) : currentView === 'home' ? (
          <HomePage onStartMatch={handleStartMatch} />
        ) : (
          <MainWindow 
            onLogout={handleLogout}
            onBackToHome={handleBackToHome}
            initialMatch={currentMatch}
          />
        )}
        {/* Debug Panel - only in development */}
        {import.meta.env.DEV && <DebugPanel />}
      </div>
    </div>
  )
}

function App() {
  return (
    <AuthProvider>
      <AppContent />
    </AuthProvider>
  )
}

export default App

