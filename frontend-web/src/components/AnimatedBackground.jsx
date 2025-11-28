import { motion } from 'framer-motion'
import { memo } from 'react'
import './AnimatedBackground.css'

const AnimatedBackground = memo(() => {
  // Màu gradient theo theme cờ vây - nâu/gỗ tối
  const gradientColors = [
    '#2c1810', // Dark brown
    '#3e2723', // Dark brown 2
    '#4a3428', // Medium dark brown
    '#5c4033', // Primary brown
    '#6b5438', // Medium brown
    '#5c4033', // Primary brown (repeat)
    '#4a3428', // Medium dark brown (repeat)
    '#3e2723', // Dark brown 2 (repeat)
  ]

  return (
    <div className="animated-background-container">
      {/* Main animated gradient */}
      <motion.div
        className="animated-gradient"
        style={{
          background: `linear-gradient(-45deg, ${gradientColors.join(', ')})`,
          backgroundSize: '400% 400%',
        }}
        animate={{
          backgroundPosition: ['0% 50%', '100% 50%', '0% 50%'],
        }}
        transition={{
          duration: 20,
          repeat: Infinity,
          ease: 'easeInOut',
        }}
      />
    </div>
  )
})

AnimatedBackground.displayName = 'AnimatedBackground'

export default AnimatedBackground

