import React, { useState, useEffect } from 'react';
import {
  View,
  Text,
  StyleSheet,
  TouchableOpacity,
  RefreshControl,
  ScrollView,
  ActivityIndicator,
  Share
} from 'react-native';
import { useNavigation } from '@react-navigation/native';
import { Ionicons } from '@expo/vector-icons';
import { Quote } from '@models/index';
import { QuotesService } from '@services/quotesService';
import { StorageService } from '@utils/storage';
import { MusicService } from '@services/musicService';
import QuoteCard from '@components/QuoteCard';
import { format } from 'date-fns';

export default function HomeScreen() {
  const navigation = useNavigation();
  const [quote, setQuote] = useState<Quote | null>(null);
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const [isFavorite, setIsFavorite] = useState(false);

  useEffect(() => {
    loadDailyQuote();
    checkIfFavorite();
  }, []);

  const loadDailyQuote = async () => {
    try {
      setLoading(true);
      
      // Check if we already have today's quote
      const savedQuote = await StorageService.loadCurrentQuote();
      if (savedQuote && isSameDay(new Date(savedQuote.date), new Date())) {
        setQuote(savedQuote);
      } else {
        // Fetch new quote
        const userPrefs = await StorageService.loadUserPrefs();
        const category = userPrefs?.categories[0];
        const newQuote = await QuotesService.fetchDailyQuote(category);
        
        setQuote(newQuote);
        await StorageService.saveCurrentQuote(newQuote);
        
        // Save to history
        await StorageService.saveHistoryEntry({
          quoteId: newQuote.id,
          viewedAt: new Date()
        });
      }
    } catch (error) {
      console.error('Error loading daily quote:', error);
    } finally {
      setLoading(false);
    }
  };

  const checkIfFavorite = async () => {
    if (!quote) return;
    
    const favorites = await StorageService.loadFavorites();
    setIsFavorite(favorites.some(fav => fav.id === quote.id));
  };

  const onRefresh = async () => {
    setRefreshing(true);
    await loadDailyQuote();
    setRefreshing(false);
  };

  const toggleFavorite = async () => {
    if (!quote) return;
    
    if (isFavorite) {
      await StorageService.removeFavorite(quote.id);
      setIsFavorite(false);
    } else {
      await StorageService.saveFavorite(quote);
      setIsFavorite(true);
    }
  };

  const shareQuote = async () => {
    if (!quote) return;
    
    try {
      await Share.share({
        message: `"${quote.text}" - ${quote.author}\n\nShared from Morning Mood`,
      });
    } catch (error) {
      console.error('Error sharing quote:', error);
    }
  };

  const isSameDay = (date1: Date, date2: Date) => {
    return format(date1, 'yyyy-MM-dd') === format(date2, 'yyyy-MM-dd');
  };

  if (loading) {
    return (
      <View style={styles.centerContainer}>
        <ActivityIndicator size="large" color="#007AFF" />
      </View>
    );
  }

  return (
    <ScrollView 
      style={styles.container}
      refreshControl={
        <RefreshControl refreshing={refreshing} onRefresh={onRefresh} />
      }
    >
      <View style={styles.dateContainer}>
        <Text style={styles.dateText}>{format(new Date(), 'EEEE, MMMM d, yyyy')}</Text>
      </View>

      {quote && (
        <>
          <QuoteCard
            quote={quote}
            onPress={() => navigation.navigate('QuoteDetail', { quoteId: quote.id })}
          />

          <View style={styles.actionContainer}>
            <TouchableOpacity
              style={styles.actionButton}
              onPress={toggleFavorite}
            >
              <Ionicons 
                name={isFavorite ? 'heart' : 'heart-outline'} 
                size={24} 
                color={isFavorite ? '#FF3B30' : '#007AFF'} 
              />
              <Text style={styles.actionText}>
                {isFavorite ? 'Favorited' : 'Favorite'}
              </Text>
            </TouchableOpacity>

            <TouchableOpacity
              style={styles.actionButton}
              onPress={shareQuote}
            >
              <Ionicons name="share-outline" size={24} color="#007AFF" />
              <Text style={styles.actionText}>Share</Text>
            </TouchableOpacity>

            <TouchableOpacity
              style={styles.actionButton}
              onPress={() => navigation.navigate('Category', { category: quote.category })}
            >
              <Ionicons name="grid-outline" size={24} color="#007AFF" />
              <Text style={styles.actionText}>More</Text>
            </TouchableOpacity>
          </View>
        </>
      )}
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#f5f5f5',
  },
  centerContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
  },
  dateContainer: {
    padding: 20,
    alignItems: 'center',
  },
  dateText: {
    fontSize: 18,
    color: '#666',
    fontWeight: '500',
  },
  actionContainer: {
    flexDirection: 'row',
    justifyContent: 'space-around',
    paddingVertical: 20,
    paddingHorizontal: 40,
  },
  actionButton: {
    alignItems: 'center',
  },
  actionText: {
    marginTop: 5,
    fontSize: 12,
    color: '#007AFF',
  },
});