/**
 * Utility để phát âm thanh đánh cờ
 * File âm thanh dài 16 giây với 10 âm thanh ở các vị trí: 1, 2, 4, 5, 7, 8, 10, 11, 13, 15 giây
 * Mỗi nước đánh sẽ phát một âm thanh khác nhau theo thứ tự, lặp lại khi hết 10 âm thanh
 */

// Các vị trí thời gian của âm thanh trong file (tính bằng giây)
const SOUND_TIMES = [0.5, 2.2, 3.9, 5.6, 7.3, 8.5, 10.2, 11.9, 13.6, 15.3]
const SOUND_DURATION = 1 // Thời lượng phát mỗi âm thanh (giây)

// Biến đếm số nước đánh để chọn âm thanh theo thứ tự
let moveCount = 0

/**
 * Phát âm thanh đánh cờ theo thứ tự
 * @param {string} soundFile - Đường dẫn đến file âm thanh
 * @param {boolean} enabled - Có bật âm thanh không
 */
export const playStoneSound = (soundFile = '/assets/zz-un-floor-goban-rich.v7.webm', enabled = true) => {
  if (!enabled) {
    return
  }
  
  try {
    // Chọn âm thanh theo thứ tự (lặp lại khi hết 10 âm thanh)
    const soundIndex = moveCount % SOUND_TIMES.length
    const startTime = SOUND_TIMES[soundIndex]
    
    // Tăng biến đếm cho lần phát tiếp theo
    moveCount++
    
    // Tạo audio element mới mỗi lần để có thể phát nhiều âm thanh cùng lúc
    const audio = new Audio(soundFile)
    audio.currentTime = startTime
    
    // Biến để track timeout và cleanup
    let timeoutId = null
    let timeUpdateHandler = null
    let isStopped = false
    
    // Hàm cleanup để dừng và xóa audio
    const stopAndCleanup = () => {
      if (isStopped) return
      isStopped = true
      
      try {
        audio.pause()
        audio.currentTime = 0
        if (timeoutId) {
          clearTimeout(timeoutId)
          timeoutId = null
        }
        if (timeUpdateHandler) {
          audio.removeEventListener('timeupdate', timeUpdateHandler)
          timeUpdateHandler = null
        }
        audio.remove()
      } catch (e) {
        // Ignore cleanup errors
      }
    }
    
    // Tính thời gian kết thúc
    const endTime = startTime + SOUND_DURATION
    
    // Lắng nghe timeupdate để dừng đúng thời điểm
    timeUpdateHandler = () => {
      if (audio.currentTime >= endTime || audio.currentTime >= audio.duration) {
        stopAndCleanup()
      }
    }
    audio.addEventListener('timeupdate', timeUpdateHandler)
    
    // Backup: dừng sau SOUND_DURATION giây (fallback nếu timeupdate không hoạt động)
    timeoutId = setTimeout(() => {
      stopAndCleanup()
    }, SOUND_DURATION * 1000)
    
    // Phát âm thanh
    const playPromise = audio.play()
    
    if (playPromise !== undefined) {
      playPromise
        .then(() => {
          // Audio đã bắt đầu phát, timeout và timeupdate sẽ tự động dừng
        })
        .catch(error => {
          // Nếu autoplay bị chặn, cleanup ngay
          console.warn('⚠️ Autoplay prevented, sound will play on next interaction:', error)
          stopAndCleanup()
        })
    }
    
    // Cleanup nếu có lỗi
    audio.addEventListener('error', (e) => {
      console.warn('⚠️ Error playing stone sound:', e)
      stopAndCleanup()
    })
    
    // Cleanup khi audio kết thúc (fallback)
    audio.addEventListener('ended', () => {
      stopAndCleanup()
    })
    
  } catch (error) {
    console.warn('⚠️ Error in playStoneSound:', error)
  }
}

/**
 * Reset biến đếm (khi bắt đầu trận mới)
 */
export const resetStoneSoundCounter = () => {
  moveCount = 0
}

/**
 * Kiểm tra xem file âm thanh có tồn tại không
 */
export const checkSoundFile = async (soundFile) => {
  try {
    const response = await fetch(soundFile, { method: 'HEAD' })
    return response.ok
  } catch {
    return false
  }
}

