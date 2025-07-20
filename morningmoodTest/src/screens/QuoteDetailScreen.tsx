import React, { useState, useEffect } from 'react';
import {
  View,
  Text,
  StyleSheet,
  ScrollView,
  TouchableOpacity,
  Share,
  ActivityIndicator
} from 'react-native';
import { RouteProp } from '@react-navigation/native';
import { StackNavigationProp } from '@react-navigation/stack';
import { Ionicons } from '@expo/vector-icons';
import { Quote } from '@models/index';
import { StorageService } from '@utils/storage';
import { RootStackParamList } from '@navigation/AppNavigator';

type QuoteDetailScreenRouteProp = RouteProp<RootStackParamList, 'QuoteDetail'>;
type QuoteDetailScreenNavigationProp = StackNavigationProp<RootStackParamList, 'QuoteDetail'>;

interface Props {
  route: QuoteDetailScreenRouteProp;
  navigation: QuoteDetailScreenNavigationProp;
}

export default function QuoteDetailScreen({ route, navigation }: Props) {
  const { quoteId } = route.params;
  const [quote, setQuote] = useState<Quote | null>(null);
  const [isFavorite, setIsFavorite] = useState(false);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    loadQuoteDetails();
  }, [quoteId]);

  const loadQuoteDetails = async () => {
    try {
      setLoading(true);
      
      // Try to find quote in favorites or current quote
      const favorites = await StorageService.loadFavorites();
      const currentQuote = await StorageService.loadCurrentQuote();
      
      const foundQuote = favorites.find(fav => fav.id === quoteId) || 
                        (currentQuote?.id === quoteId ? currentQuote : null);
      
      if (foundQuote) {
        setQuote(foundQuote);
        setIsFavorite(favorites.some(fav => fav.id === quoteId));
      }
    } catch (error) {
      console.error('Error loading quote details:', error);
    } finally {
      setLoading(false);
    }
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
        message: `"${quote.text}"\n\n— ${quote.author}\n\nShared from Morning Mood`,
      });
    } catch (error) {
      console.error('Error sharing quote:', error);
    }
  };

  if (loading) {
    return (
      <View style={styles.centerContainer}>
        <ActivityIndicator size="large" color="#007AFF" />
      </View>
    );
  }

  if (!quote) {
    return (
      <View style={styles.centerContainer}>
        <Text style={styles.errorText}>Quote not found</Text>
      </View>
    );
  }

  return (
    <ScrollView style={styles.container}>
      <View style={styles.quoteContainer}>
        <Text style={styles.quoteText}>"{quote.text}"</Text>
        
        <View style={styles.authorContainer}>
          <View style={styles.divider} />
          <Text style={styles.authorText}>{quote.author}</Text>
        </View>
        
        <View style={styles.categoryBadge}>
          <Text style={styles.categoryText}>{quote.category}</Text>
        </View>
      </View>

      <View style={styles.actionContainer}>
        <TouchableOpacity
          style={[styles.actionButton, isFavorite ? styles.removeButton : styles.primaryButton]}
          onPress={toggleFavorite}
        >
          <Ionicons 
            name={isFavorite ? 'heart' : 'heart-outline'} 
            size={24} 
            color="white" 
          />
          <Text style={styles.primaryButtonText}>
            {isFavorite ? 'Remove from Favorites' : 'Add to Favorites'}
          </Text>
        </TouchableOpacity>

        <TouchableOpacity
          style={[styles.actionButton, styles.secondaryButton]}
          onPress={shareQuote}
        >
          <Ionicons name="share-outline" size={24} color="#007AFF" />
          <Text style={styles.secondaryButtonText}>Share Quote</Text>
        </TouchableOpacity>

        <TouchableOpacity
          style={[styles.actionButton, styles.secondaryButton]}
          onPress={() => navigation.navigate('Category', { category: quote.category })}
        >
          <Ionicons name="grid-outline" size={24} color="#007AFF" />
          <Text style={styles.secondaryButtonText}>More {quote.category} Quotes</Text>
        </TouchableOpacity>
      </View>

      <View style={styles.infoSection}>
        <Text style={styles.infoTitle}>About this quote</Text>
        <Text style={styles.infoText}>
          This {quote.category} quote by {quote.author} can help inspire and motivate you 
          throughout your day. Save it to your favorites to revisit whenever you need 
          a boost of wisdom.
        </Text>
      </View>
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
  errorText: {
    fontSize: 16,
    color: '#999',
  },
  quoteContainer: {
    backgroundColor: 'white',
    margin: 20,
    padding: 30,
    borderRadius: 15,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 8,
    elevation: 5,
    alignItems: 'center',
  },
  quoteText: {
    fontSize: 20,
    fontStyle: 'italic',
    color: '#333',
    textAlign: 'center',
    lineHeight: 30,
    marginBottom: 20,
  },
  authorContainer: {
    alignItems: 'center',
    marginTop: 10,
  },
  divider: {
    width: 40,
    height: 2,
    backgroundColor: '#007AFF',
    marginBottom: 10,
  },
  authorText: {
    fontSize: 16,
    color: '#666',
    fontWeight: '500',
  },
  categoryBadge: {
    backgroundColor: '#007AFF',
    paddingHorizontal: 15,
    paddingVertical: 5,
    borderRadius: 15,
    marginTop: 20,
  },
  categoryText: {
    color: 'white',
    fontSize: 12,
    fontWeight: '600',
    textTransform: 'uppercase',
  },
  actionContainer: {
    paddingHorizontal: 20,
    marginBottom: 30,
  },
  actionButton: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    padding: 15,
    borderRadius: 10,
    marginBottom: 10,
  },
  primaryButton: {
    backgroundColor: '#007AFF',
  },
  removeButton: {
    backgroundColor: '#FF3B30',
  },
  secondaryButton: {
    backgroundColor: 'white',
    borderWidth: 1,
    borderColor: '#007AFF',
  },
  primaryButtonText: {
    color: 'white',
    fontSize: 16,
    fontWeight: '600',
    marginLeft: 10,
  },
  secondaryButtonText: {
    color: '#007AFF',
    fontSize: 16,
    fontWeight: '600',
    marginLeft: 10,
  },
  infoSection: {
    backgroundColor: 'white',
    marginHorizontal: 20,
    marginBottom: 20,
    padding: 20,
    borderRadius: 10,
  },
  infoTitle: {
    fontSize: 16,
    fontWeight: '600',
    color: '#333',
    marginBottom: 10,
  },
  infoText: {
    fontSize: 14,
    color: '#666',
    lineHeight: 20,
  },
});