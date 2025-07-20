import { Audio } from 'expo-av';
import { MusicTrack } from '@models/index';

export class MusicService {
  private static sound: Audio.Sound | null = null;
  private static isPlaying: boolean = false;

  /**
   * Initialize audio settings
   */
  static async initialize(): Promise<void> {
    try {
      await Audio.setAudioModeAsync({
        allowsRecordingIOS: false,
        staysActiveInBackground: true,
        playsInSilentModeIOS: true,
        shouldDuckAndroid: true,
        playThroughEarpieceAndroid: false,
      });
    } catch (error) {
      console.error('Error initializing audio:', error);
    }
  }

  /**
   * Play background music
   */
  static async playBackgroundMusic(track: MusicTrack): Promise<void> {
    try {
      // Stop current sound if playing
      if (this.sound) {
        await this.stopMusic();
      }

      // Create and load new sound
      const { sound } = await Audio.Sound.createAsync(
        { uri: track.uri },
        { shouldPlay: true, isLooping: true, volume: 0.5 }
      );

      this.sound = sound;
      this.isPlaying = true;

      // Set up playback status update
      sound.setOnPlaybackStatusUpdate((status) => {
        if (status.isLoaded) {
          this.isPlaying = status.isPlaying;
        }
      });
    } catch (error) {
      console.error('Error playing music:', error);
    }
  }

  /**
   * Pause music
   */
  static async pauseMusic(): Promise<void> {
    if (this.sound && this.isPlaying) {
      await this.sound.pauseAsync();
      this.isPlaying = false;
    }
  }

  /**
   * Resume music
   */
  static async resumeMusic(): Promise<void> {
    if (this.sound && !this.isPlaying) {
      await this.sound.playAsync();
      this.isPlaying = true;
    }
  }

  /**
   * Stop music
   */
  static async stopMusic(): Promise<void> {
    if (this.sound) {
      await this.sound.stopAsync();
      await this.sound.unloadAsync();
      this.sound = null;
      this.isPlaying = false;
    }
  }

  /**
   * Set volume
   */
  static async setVolume(volume: number): Promise<void> {
    if (this.sound) {
      await this.sound.setVolumeAsync(Math.max(0, Math.min(1, volume)));
    }
  }

  /**
   * Get available music tracks
   */
  static getAvailableTracks(): MusicTrack[] {
    return [
      {
        id: 'calm-1',
        name: 'Morning Meditation',
        uri: 'https://example.com/tracks/morning-meditation.mp3',
        duration: 300,
        category: 'calm'
      },
      {
        id: 'energetic-1',
        name: 'Rise and Shine',
        uri: 'https://example.com/tracks/rise-and-shine.mp3',
        duration: 240,
        category: 'energetic'
      },
      {
        id: 'focus-1',
        name: 'Deep Focus',
        uri: 'https://example.com/tracks/deep-focus.mp3',
        duration: 600,
        category: 'focus'
      }
    ];
  }

  /**
   * Check if music is playing
   */
  static getIsPlaying(): boolean {
    return this.isPlaying;
  }
}