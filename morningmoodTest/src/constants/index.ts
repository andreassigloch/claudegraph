export const API_CONFIG = {
  QUOTES_API_URL: 'https://api.quotable.io',
  REQUEST_TIMEOUT: 10000,
};

export const STORAGE_KEYS = {
  USER_PREFS: '@morningmood:userPrefs',
  FAVORITES: '@morningmood:favorites',
  HISTORY: '@morningmood:history',
  CURRENT_QUOTE: '@morningmood:currentQuote',
  NOTIFICATION_ID: '@morningmood:notificationId',
};

export const NOTIFICATION_CONFIG = {
  DEFAULT_TIME: '08:00',
  CHANNEL_ID: 'daily-quotes',
  CHANNEL_NAME: 'Daily Quotes',
};

export const THEME = {
  primary: '#007AFF',
  secondary: '#5856D6',
  danger: '#FF3B30',
  success: '#34C759',
  warning: '#FF9500',
  background: '#f5f5f5',
  surface: '#ffffff',
  text: {
    primary: '#333333',
    secondary: '#666666',
    tertiary: '#999999',
  },
};

export const CATEGORIES = [
  { id: 'inspirational', name: 'Inspirational', icon: 'star' },
  { id: 'motivational', name: 'Motivational', icon: 'rocket' },
  { id: 'wisdom', name: 'Wisdom', icon: 'bulb' },
  { id: 'happiness', name: 'Happiness', icon: 'happy' },
  { id: 'success', name: 'Success', icon: 'trophy' },
  { id: 'life', name: 'Life', icon: 'heart' },
];

export const MUSIC_TRACKS = {
  calm: [
    {
      id: 'calm-1',
      name: 'Morning Meditation',
      uri: 'https://example.com/tracks/morning-meditation.mp3',
      duration: 300,
    },
  ],
  energetic: [
    {
      id: 'energetic-1',
      name: 'Rise and Shine',
      uri: 'https://example.com/tracks/rise-and-shine.mp3',
      duration: 240,
    },
  ],
  focus: [
    {
      id: 'focus-1',
      name: 'Deep Focus',
      uri: 'https://example.com/tracks/deep-focus.mp3',
      duration: 600,
    },
  ],
};