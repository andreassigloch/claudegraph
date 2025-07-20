import * as Notifications from 'expo-notifications';
import { Platform } from 'react-native';
import { NotificationData, Quote } from '@models/index';

// Configure notification behavior
Notifications.setNotificationHandler({
  handleNotification: async () => ({
    shouldShowAlert: true,
    shouldPlaySound: true,
    shouldSetBadge: false,
  }),
});

export class NotificationService {
  /**
   * Request notification permissions
   */
  static async requestPermissions(): Promise<boolean> {
    const { status: existingStatus } = await Notifications.getPermissionsAsync();
    let finalStatus = existingStatus;
    
    if (existingStatus !== 'granted') {
      const { status } = await Notifications.requestPermissionsAsync();
      finalStatus = status;
    }
    
    if (finalStatus !== 'granted') {
      console.log('Failed to get push token for push notification!');
      return false;
    }
    
    return true;
  }

  /**
   * Schedule daily notification
   */
  static async scheduleDailyNotification(time: string, quote: Quote): Promise<string> {
    const [hours, minutes] = time.split(':').map(Number);
    
    const trigger = {
      hour: hours,
      minute: minutes,
      repeats: true,
    };

    const notificationId = await Notifications.scheduleNotificationAsync({
      content: {
        title: 'Daily Wisdom',
        body: `"${quote.text}" - ${quote.author}`,
        data: { quoteId: quote.id },
      },
      trigger,
    });

    return notificationId;
  }

  /**
   * Cancel scheduled notification
   */
  static async cancelNotification(notificationId: string): Promise<void> {
    await Notifications.cancelScheduledNotificationAsync(notificationId);
  }

  /**
   * Cancel all notifications
   */
  static async cancelAllNotifications(): Promise<void> {
    await Notifications.cancelAllScheduledNotificationsAsync();
  }

  /**
   * Get all scheduled notifications
   */
  static async getScheduledNotifications(): Promise<NotificationData[]> {
    const scheduled = await Notifications.getAllScheduledNotificationsAsync();
    
    return scheduled.map(notif => ({
      id: notif.identifier,
      title: notif.content.title || '',
      body: notif.content.body || '',
      scheduledTime: new Date(notif.trigger),
      isActive: true
    }));
  }

  /**
   * Send immediate notification
   */
  static async sendImmediateNotification(title: string, body: string, data?: any): Promise<void> {
    await Notifications.presentNotificationAsync({
      content: {
        title,
        body,
        data,
      },
    });
  }
}