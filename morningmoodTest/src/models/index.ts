// Data models based on architecture schemas

export interface Quote {
  id: string;
  text: string;
  author: string;
  category: string;
  date: Date;
  isFavorite?: boolean;
}

export interface UserPrefs {
  notificationTime: string; // HH:MM format
  notificationsEnabled: boolean;
  categories: string[];
  musicEnabled: boolean;
  theme: 'light' | 'dark' | 'auto';
}

export interface NotificationData {
  id: string;
  title: string;
  body: string;
  scheduledTime: Date;
  isActive: boolean;
}

export interface MusicTrack {
  id: string;
  name: string;
  uri: string;
  duration: number;
  category: 'calm' | 'energetic' | 'focus';
}

export interface HistoryEntry {
  quoteId: string;
  viewedAt: Date;
  mood?: 'happy' | 'neutral' | 'sad';
  notes?: string;
}