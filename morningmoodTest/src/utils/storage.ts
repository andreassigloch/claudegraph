import AsyncStorage from '@react-native-async-storage/async-storage';
import { Quote, UserPrefs, HistoryEntry } from '@models/index';

const STORAGE_KEYS = {
  USER_PREFS: '@morningmood:userPrefs',
  FAVORITES: '@morningmood:favorites',
  HISTORY: '@morningmood:history',
  CURRENT_QUOTE: '@morningmood:currentQuote',
  NOTIFICATION_ID: '@morningmood:notificationId'
};

export class StorageService {
  /**
   * Save user preferences
   */
  static async saveUserPrefs(prefs: UserPrefs): Promise<void> {
    try {
      await AsyncStorage.setItem(STORAGE_KEYS.USER_PREFS, JSON.stringify(prefs));
    } catch (error) {
      console.error('Error saving user preferences:', error);
    }
  }

  /**
   * Load user preferences
   */
  static async loadUserPrefs(): Promise<UserPrefs | null> {
    try {
      const data = await AsyncStorage.getItem(STORAGE_KEYS.USER_PREFS);
      return data ? JSON.parse(data) : null;
    } catch (error) {
      console.error('Error loading user preferences:', error);
      return null;
    }
  }

  /**
   * Save favorite quote
   */
  static async saveFavorite(quote: Quote): Promise<void> {
    try {
      const favorites = await this.loadFavorites();
      const updatedFavorites = [...favorites, { ...quote, isFavorite: true }];
      await AsyncStorage.setItem(STORAGE_KEYS.FAVORITES, JSON.stringify(updatedFavorites));
    } catch (error) {
      console.error('Error saving favorite:', error);
    }
  }

  /**
   * Remove favorite quote
   */
  static async removeFavorite(quoteId: string): Promise<void> {
    try {
      const favorites = await this.loadFavorites();
      const updatedFavorites = favorites.filter(q => q.id !== quoteId);
      await AsyncStorage.setItem(STORAGE_KEYS.FAVORITES, JSON.stringify(updatedFavorites));
    } catch (error) {
      console.error('Error removing favorite:', error);
    }
  }

  /**
   * Load favorite quotes
   */
  static async loadFavorites(): Promise<Quote[]> {
    try {
      const data = await AsyncStorage.getItem(STORAGE_KEYS.FAVORITES);
      return data ? JSON.parse(data) : [];
    } catch (error) {
      console.error('Error loading favorites:', error);
      return [];
    }
  }

  /**
   * Save history entry
   */
  static async saveHistoryEntry(entry: HistoryEntry): Promise<void> {
    try {
      const history = await this.loadHistory();
      const updatedHistory = [entry, ...history].slice(0, 100); // Keep last 100 entries
      await AsyncStorage.setItem(STORAGE_KEYS.HISTORY, JSON.stringify(updatedHistory));
    } catch (error) {
      console.error('Error saving history entry:', error);
    }
  }

  /**
   * Load history
   */
  static async loadHistory(): Promise<HistoryEntry[]> {
    try {
      const data = await AsyncStorage.getItem(STORAGE_KEYS.HISTORY);
      return data ? JSON.parse(data) : [];
    } catch (error) {
      console.error('Error loading history:', error);
      return [];
    }
  }

  /**
   * Save current quote
   */
  static async saveCurrentQuote(quote: Quote): Promise<void> {
    try {
      await AsyncStorage.setItem(STORAGE_KEYS.CURRENT_QUOTE, JSON.stringify(quote));
    } catch (error) {
      console.error('Error saving current quote:', error);
    }
  }

  /**
   * Load current quote
   */
  static async loadCurrentQuote(): Promise<Quote | null> {
    try {
      const data = await AsyncStorage.getItem(STORAGE_KEYS.CURRENT_QUOTE);
      return data ? JSON.parse(data) : null;
    } catch (error) {
      console.error('Error loading current quote:', error);
      return null;
    }
  }

  /**
   * Save notification ID
   */
  static async saveNotificationId(id: string): Promise<void> {
    try {
      await AsyncStorage.setItem(STORAGE_KEYS.NOTIFICATION_ID, id);
    } catch (error) {
      console.error('Error saving notification ID:', error);
    }
  }

  /**
   * Load notification ID
   */
  static async loadNotificationId(): Promise<string | null> {
    try {
      return await AsyncStorage.getItem(STORAGE_KEYS.NOTIFICATION_ID);
    } catch (error) {
      console.error('Error loading notification ID:', error);
      return null;
    }
  }

  /**
   * Clear all storage
   */
  static async clearAll(): Promise<void> {
    try {
      await AsyncStorage.multiRemove(Object.values(STORAGE_KEYS));
    } catch (error) {
      console.error('Error clearing storage:', error);
    }
  }
}