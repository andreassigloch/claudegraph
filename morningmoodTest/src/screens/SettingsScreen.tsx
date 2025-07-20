import React, { useState, useEffect } from 'react';
import {
  View,
  Text,
  StyleSheet,
  ScrollView,
  Switch,
  TouchableOpacity,
  Alert,
  Platform
} from 'react-native';
import DateTimePicker from '@react-native-community/datetimepicker';
import { Ionicons } from '@expo/vector-icons';
import { UserPrefs } from '@models/index';
import { StorageService } from '@utils/storage';
import { NotificationService } from '@services/notificationService';
import { MusicService } from '@services/musicService';

const CATEGORIES = [
  { id: 'inspirational', name: 'Inspirational' },
  { id: 'motivational', name: 'Motivational' },
  { id: 'wisdom', name: 'Wisdom' },
  { id: 'happiness', name: 'Happiness' },
  { id: 'success', name: 'Success' },
  { id: 'life', name: 'Life' },
];

export default function SettingsScreen() {
  const [prefs, setPrefs] = useState<UserPrefs>({
    notificationTime: '08:00',
    notificationsEnabled: true,
    categories: ['inspirational'],
    musicEnabled: false,
    theme: 'auto'
  });
  const [showTimePicker, setShowTimePicker] = useState(false);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    loadPreferences();
  }, []);

  const loadPreferences = async () => {
    try {
      const savedPrefs = await StorageService.loadUserPrefs();
      if (savedPrefs) {
        setPrefs(savedPrefs);
      }
    } catch (error) {
      console.error('Error loading preferences:', error);
    } finally {
      setLoading(false);
    }
  };

  const savePreferences = async (newPrefs: UserPrefs) => {
    try {
      await StorageService.saveUserPrefs(newPrefs);
      setPrefs(newPrefs);
      
      // Update notifications if needed
      if (newPrefs.notificationsEnabled) {
        await setupNotifications(newPrefs.notificationTime);
      } else {
        await NotificationService.cancelAllNotifications();
      }
      
      // Update music settings
      if (!newPrefs.musicEnabled) {
        await MusicService.stopMusic();
      }
    } catch (error) {
      console.error('Error saving preferences:', error);
      Alert.alert('Error', 'Failed to save preferences');
    }
  };

  const setupNotifications = async (time: string) => {
    const hasPermission = await NotificationService.requestPermissions();
    if (!hasPermission) {
      Alert.alert(
        'Permission Required',
        'Please enable notifications in your device settings to receive daily quotes.'
      );
      return;
    }
    
    // Cancel existing notifications
    await NotificationService.cancelAllNotifications();
    
    // Schedule new notification
    // Note: In a real app, you'd fetch a quote here
    const mockQuote = {
      id: 'daily',
      text: 'Your daily wisdom awaits',
      author: 'Morning Mood',
      category: 'inspirational',
      date: new Date()
    };
    
    await NotificationService.scheduleDailyNotification(time, mockQuote);
  };

  const toggleNotifications = async (enabled: boolean) => {
    const newPrefs = { ...prefs, notificationsEnabled: enabled };
    await savePreferences(newPrefs);
  };

  const toggleMusic = async (enabled: boolean) => {
    const newPrefs = { ...prefs, musicEnabled: enabled };
    await savePreferences(newPrefs);
  };

  const toggleCategory = (categoryId: string) => {
    const newCategories = prefs.categories.includes(categoryId)
      ? prefs.categories.filter(c => c !== categoryId)
      : [...prefs.categories, categoryId];
    
    if (newCategories.length === 0) {
      Alert.alert('Category Required', 'Please select at least one category');
      return;
    }
    
    const newPrefs = { ...prefs, categories: newCategories };
    savePreferences(newPrefs);
  };

  const handleTimeChange = (event: any, selectedDate?: Date) => {
    setShowTimePicker(Platform.OS === 'ios');
    
    if (selectedDate) {
      const hours = selectedDate.getHours().toString().padStart(2, '0');
      const minutes = selectedDate.getMinutes().toString().padStart(2, '0');
      const newTime = `${hours}:${minutes}`;
      
      const newPrefs = { ...prefs, notificationTime: newTime };
      savePreferences(newPrefs);
    }
  };

  const clearData = () => {
    Alert.alert(
      'Clear All Data',
      'This will remove all your favorites, history, and preferences. Are you sure?',
      [
        { text: 'Cancel', style: 'cancel' },
        {
          text: 'Clear',
          style: 'destructive',
          onPress: async () => {
            await StorageService.clearAll();
            await loadPreferences(); // Reload default preferences
            Alert.alert('Success', 'All data has been cleared');
          }
        }
      ]
    );
  };

  return (
    <ScrollView style={styles.container}>
      {/* Notifications Section */}
      <View style={styles.section}>
        <Text style={styles.sectionTitle}>Notifications</Text>
        
        <View style={styles.settingRow}>
          <View style={styles.settingInfo}>
            <Text style={styles.settingLabel}>Daily Quotes</Text>
            <Text style={styles.settingDescription}>
              Receive a new quote every day
            </Text>
          </View>
          <Switch
            value={prefs.notificationsEnabled}
            onValueChange={toggleNotifications}
            trackColor={{ false: '#767577', true: '#81b0ff' }}
            thumbColor={prefs.notificationsEnabled ? '#007AFF' : '#f4f3f4'}
          />
        </View>
        
        {prefs.notificationsEnabled && (
          <TouchableOpacity
            style={styles.settingRow}
            onPress={() => setShowTimePicker(true)}
          >
            <View style={styles.settingInfo}>
              <Text style={styles.settingLabel}>Notification Time</Text>
              <Text style={styles.settingValue}>{prefs.notificationTime}</Text>
            </View>
            <Ionicons name="chevron-forward" size={20} color="#999" />
          </TouchableOpacity>
        )}
      </View>

      {/* Categories Section */}
      <View style={styles.section}>
        <Text style={styles.sectionTitle}>Quote Categories</Text>
        
        {CATEGORIES.map(category => (
          <TouchableOpacity
            key={category.id}
            style={styles.categoryRow}
            onPress={() => toggleCategory(category.id)}
          >
            <Text style={styles.categoryLabel}>{category.name}</Text>
            <Ionicons
              name={prefs.categories.includes(category.id) ? 'checkmark-circle' : 'ellipse-outline'}
              size={24}
              color={prefs.categories.includes(category.id) ? '#007AFF' : '#999'}
            />
          </TouchableOpacity>
        ))}
      </View>

      {/* Music Section */}
      <View style={styles.section}>
        <Text style={styles.sectionTitle}>Background Music</Text>
        
        <View style={styles.settingRow}>
          <View style={styles.settingInfo}>
            <Text style={styles.settingLabel}>Ambient Music</Text>
            <Text style={styles.settingDescription}>
              Play calming music while browsing
            </Text>
          </View>
          <Switch
            value={prefs.musicEnabled}
            onValueChange={toggleMusic}
            trackColor={{ false: '#767577', true: '#81b0ff' }}
            thumbColor={prefs.musicEnabled ? '#007AFF' : '#f4f3f4'}
          />
        </View>
      </View>

      {/* Data Section */}
      <View style={styles.section}>
        <Text style={styles.sectionTitle}>Data</Text>
        
        <TouchableOpacity
          style={styles.dangerButton}
          onPress={clearData}
        >
          <Text style={styles.dangerButtonText}>Clear All Data</Text>
        </TouchableOpacity>
      </View>

      {/* About Section */}
      <View style={styles.section}>
        <Text style={styles.sectionTitle}>About</Text>
        
        <View style={styles.aboutRow}>
          <Text style={styles.aboutLabel}>Version</Text>
          <Text style={styles.aboutValue}>1.0.0</Text>
        </View>
        
        <View style={styles.aboutRow}>
          <Text style={styles.aboutLabel}>Created with</Text>
          <Text style={styles.aboutValue}>❤️ and React Native</Text>
        </View>
      </View>

      {showTimePicker && (
        <DateTimePicker
          value={new Date(`2000-01-01T${prefs.notificationTime}:00`)}
          mode="time"
          is24Hour={true}
          display="default"
          onChange={handleTimeChange}
        />
      )}
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#f5f5f5',
  },
  section: {
    backgroundColor: 'white',
    marginTop: 20,
    paddingHorizontal: 20,
    paddingVertical: 15,
  },
  sectionTitle: {
    fontSize: 16,
    fontWeight: '600',
    color: '#333',
    marginBottom: 15,
  },
  settingRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    paddingVertical: 15,
    borderBottomWidth: 1,
    borderBottomColor: '#f0f0f0',
  },
  settingInfo: {
    flex: 1,
  },
  settingLabel: {
    fontSize: 16,
    color: '#333',
    marginBottom: 5,
  },
  settingDescription: {
    fontSize: 14,
    color: '#999',
  },
  settingValue: {
    fontSize: 14,
    color: '#007AFF',
  },
  categoryRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    paddingVertical: 12,
    borderBottomWidth: 1,
    borderBottomColor: '#f0f0f0',
  },
  categoryLabel: {
    fontSize: 16,
    color: '#333',
  },
  dangerButton: {
    backgroundColor: '#FF3B30',
    padding: 15,
    borderRadius: 8,
    alignItems: 'center',
  },
  dangerButtonText: {
    color: 'white',
    fontSize: 16,
    fontWeight: '600',
  },
  aboutRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    paddingVertical: 10,
  },
  aboutLabel: {
    fontSize: 14,
    color: '#666',
  },
  aboutValue: {
    fontSize: 14,
    color: '#333',
  },
});