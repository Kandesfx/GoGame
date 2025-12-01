/**
 * Utility functions for rank tiers and ELO calculations
 */

// Rank tier definitions based on ELO
export const RANK_TIERS = {
  UNRANKED: {
    name: 'Chưa xếp hạng',
    nameEn: 'Unranked',
    minElo: 0,
    maxElo: 999,
    color: '#6b7280',
    bgColor: 'rgba(107, 114, 128, 0.1)',
    borderColor: 'rgba(107, 114, 128, 0.3)',
    icon: '⚪',
    gradient: 'linear-gradient(135deg, #6b7280 0%, #4b5563 100%)'
  },
  BRONZE: {
    name: 'Đồng',
    nameEn: 'Bronze',
    minElo: 1000,
    maxElo: 1299,
    color: '#cd7f32',
    bgColor: 'rgba(205, 127, 50, 0.1)',
    borderColor: 'rgba(205, 127, 50, 0.3)',
    icon: '🥉',
    gradient: 'linear-gradient(135deg, #cd7f32 0%, #a0522d 100%)'
  },
  SILVER: {
    name: 'Bạc',
    nameEn: 'Silver',
    minElo: 1300,
    maxElo: 1599,
    color: '#c0c0c0',
    bgColor: 'rgba(192, 192, 192, 0.1)',
    borderColor: 'rgba(192, 192, 192, 0.3)',
    icon: '🥈',
    gradient: 'linear-gradient(135deg, #c0c0c0 0%, #a8a8a8 100%)'
  },
  GOLD: {
    name: 'Vàng',
    nameEn: 'Gold',
    minElo: 1600,
    maxElo: 1899,
    color: '#ffd700',
    bgColor: 'rgba(255, 215, 0, 0.1)',
    borderColor: 'rgba(255, 215, 0, 0.3)',
    icon: '🥇',
    gradient: 'linear-gradient(135deg, #ffd700 0%, #ffb347 100%)'
  },
  PLATINUM: {
    name: 'Bạch Kim',
    nameEn: 'Platinum',
    minElo: 1900,
    maxElo: 2199,
    color: '#e5e4e2',
    bgColor: 'rgba(229, 228, 226, 0.1)',
    borderColor: 'rgba(229, 228, 226, 0.3)',
    icon: '💎',
    gradient: 'linear-gradient(135deg, #e5e4e2 0%, #b8b6b4 100%)'
  },
  DIAMOND: {
    name: 'Kim Cương',
    nameEn: 'Diamond',
    minElo: 2200,
    maxElo: 2499,
    color: '#b9f2ff',
    bgColor: 'rgba(185, 242, 255, 0.1)',
    borderColor: 'rgba(185, 242, 255, 0.3)',
    icon: '💠',
    gradient: 'linear-gradient(135deg, #b9f2ff 0%, #87ceeb 100%)'
  },
  MASTER: {
    name: 'Bậc Thầy',
    nameEn: 'Master',
    minElo: 2500,
    maxElo: 2799,
    color: '#9b59b6',
    bgColor: 'rgba(155, 89, 182, 0.1)',
    borderColor: 'rgba(155, 89, 182, 0.3)',
    icon: '👑',
    gradient: 'linear-gradient(135deg, #9b59b6 0%, #8e44ad 100%)'
  },
  GRANDMASTER: {
    name: 'Đại Sư',
    nameEn: 'Grandmaster',
    minElo: 2800,
    maxElo: Infinity,
    color: '#e74c3c',
    bgColor: 'rgba(231, 76, 60, 0.1)',
    borderColor: 'rgba(231, 76, 60, 0.3)',
    icon: '🌟',
    gradient: 'linear-gradient(135deg, #e74c3c 0%, #c0392b 100%)'
  }
}

/**
 * Get rank tier based on ELO rating
 * @param {number} elo - ELO rating
 * @returns {object} Rank tier object
 */
export function getRankTier(elo) {
  const eloValue = elo || 0
  
  // Find the tier that matches the ELO range
  for (const [key, tier] of Object.entries(RANK_TIERS)) {
    if (eloValue >= tier.minElo && eloValue <= tier.maxElo) {
      return { ...tier, key }
    }
  }
  
  // Default to UNRANKED if no match
  return { ...RANK_TIERS.UNRANKED, key: 'UNRANKED' }
}

/**
 * Get rank tier name in Vietnamese
 * @param {number} elo - ELO rating
 * @returns {string} Rank tier name
 */
export function getRankName(elo) {
  return getRankTier(elo).name
}

/**
 * Get rank tier icon
 * @param {number} elo - ELO rating
 * @returns {string} Rank tier icon
 */
export function getRankIcon(elo) {
  return getRankTier(elo).icon
}

/**
 * Get rank tier color
 * @param {number} elo - ELO rating
 * @returns {string} Rank tier color
 */
export function getRankColor(elo) {
  return getRankTier(elo).color
}

/**
 * Format ELO with rank tier badge
 * @param {number} elo - ELO rating
 * @returns {object} Formatted rank info
 */
export function formatRankInfo(elo) {
  const tier = getRankTier(elo)
  return {
    elo,
    tier: tier.name,
    icon: tier.icon,
    color: tier.color,
    bgColor: tier.bgColor,
    borderColor: tier.borderColor,
    gradient: tier.gradient
  }
}

