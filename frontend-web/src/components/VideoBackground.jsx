import { memo, useRef, useEffect } from 'react'
import './VideoBackground.css'

const VideoBackground = memo(() => {
  const videoRef = useRef(null)

  useEffect(() => {
    const video = videoRef.current
    if (!video) return

    // Đảm bảo video settings
    video.loop = true
    video.muted = true
    video.playsInline = true
    
    // Xử lý lỗi
    const handleError = (e) => {
      console.error('❌ Video error:', e, video.error)
    }
    
    // Thử play video
    const tryPlay = async () => {
      try {
        if (video.readyState >= 2) {
          await video.play()
          console.log('▶️ Video playing')
        }
      } catch (err) {
        console.warn('⚠️ Autoplay prevented:', err)
        // Thử play khi user tương tác
        const playOnInteraction = async () => {
          try {
            await video.play()
            console.log('▶️ Video playing (after interaction)')
          } catch (e) {
            console.error('❌ Failed to play:', e)
          }
        }
        document.addEventListener('click', playOnInteraction, { once: true })
        document.addEventListener('touchstart', playOnInteraction, { once: true })
      }
    }
    
    // Event listeners
    video.addEventListener('error', handleError)
    video.addEventListener('loadedmetadata', tryPlay)
    video.addEventListener('canplay', tryPlay)
    video.addEventListener('canplaythrough', tryPlay)
    
    // Cleanup
    return () => {
      if (video) {
        video.removeEventListener('error', handleError)
        video.removeEventListener('loadedmetadata', tryPlay)
        video.removeEventListener('canplay', tryPlay)
        video.removeEventListener('canplaythrough', tryPlay)
      }
    }
  }, [])

  return (
    <div className="video-background-container">
      <video
        ref={videoRef}
        className="video-background"
        autoPlay
        loop
        muted
        playsInline
        preload="auto"
        disablePictureInPicture
        disableRemotePlayback
      >
        <source src="/assets/videoloop.webm" type="video/webm" />
        Your browser does not support the video tag.
      </video>
      {/* Fallback gradient nếu video không load được */}
      <div className="video-fallback">
        <div className="fallback-gradient" />
      </div>
      {/* Dark overlay để text dễ đọc hơn */}
      <div className="video-overlay" />
      {/* Scanline effect */}
      <div className="scanlines" />
    </div>
  )
})

VideoBackground.displayName = 'VideoBackground'

export default VideoBackground

