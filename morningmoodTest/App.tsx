import React, { useEffect } from 'react';
import { StatusBar } from 'expo-status-bar';
import { SafeAreaProvider } from 'react-native-safe-area-context';
import * as Notifications from 'expo-notifications';
import AppNavigator from '@navigation/AppNavigator';
import { NotificationService } from '@services/notificationService';
import { MusicService } from '@services/musicService';

export default function App() {
  useEffect(() => {
    // Initialize services
    initializeApp();
  }, []);

  const initializeApp = async () => {
    try {
      // Request notification permissions
      await NotificationService.requestPermissions();
      
      // Initialize music service
      await MusicService.initialize();
      
      // Set up notification listeners
      const subscription = Notifications.addNotificationReceivedListener(notification => {
        console.log('Notification received:', notification);
      });

      return () => subscription.remove();
    } catch (error) {
      console.error('Error initializing app:', error);
    }
  };

  return (
    <SafeAreaProvider>
      <StatusBar style="light" />
      <AppNavigator />
    </SafeAreaProvider>
  );
}