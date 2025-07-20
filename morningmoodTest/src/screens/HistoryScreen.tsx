import React, { useState, useCallback } from 'react';
import {
  View,
  Text,
  StyleSheet,
  FlatList,
  TouchableOpacity,
  SectionList
} from 'react-native';
import { useNavigation, useFocusEffect } from '@react-navigation/native';
import { format, isToday, isYesterday, parseISO } from 'date-fns';
import { HistoryEntry, Quote } from '@models/index';
import { StorageService } from '@utils/storage';
import { QuotesService } from '@services/quotesService';

interface HistorySection {
  title: string;
  data: (HistoryEntry & { quote?: Quote })[];
}

export default function HistoryScreen() {
  const navigation = useNavigation();
  const [sections, setSections] = useState<HistorySection[]>([]);
  const [loading, setLoading] = useState(true);

  useFocusEffect(
    useCallback(() => {
      loadHistory();
    }, [])
  );

  const loadHistory = async () => {
    try {
      setLoading(true);
      const history = await StorageService.loadHistory();
      const favorites = await StorageService.loadFavorites();
      
      // Group history by date
      const grouped = history.reduce((acc, entry) => {
        const dateKey = format(new Date(entry.viewedAt), 'yyyy-MM-dd');
        if (!acc[dateKey]) {
          acc[dateKey] = [];
        }
        
        // Try to find quote in favorites
        const quote = favorites.find(fav => fav.id === entry.quoteId);
        acc[dateKey].push({ ...entry, quote });
        
        return acc;
      }, {} as Record<string, (HistoryEntry & { quote?: Quote })[]>);

      // Convert to sections
      const sectionData = Object.entries(grouped)
        .sort(([a], [b]) => b.localeCompare(a))
        .map(([date, entries]) => ({
          title: formatSectionTitle(date),
          data: entries
        }));

      setSections(sectionData);
    } catch (error) {
      console.error('Error loading history:', error);
    } finally {
      setLoading(false);
    }
  };

  const formatSectionTitle = (dateString: string) => {
    const date = parseISO(dateString);
    if (isToday(date)) return 'Today';
    if (isYesterday(date)) return 'Yesterday';
    return format(date, 'EEEE, MMMM d');
  };

  const renderSectionHeader = ({ section }: { section: HistorySection }) => (
    <View style={styles.sectionHeader}>
      <Text style={styles.sectionTitle}>{section.title}</Text>
    </View>
  );

  const renderItem = ({ item }: { item: HistoryEntry & { quote?: Quote } }) => (
    <TouchableOpacity
      style={styles.historyItem}
      onPress={() => {
        if (item.quote) {
          navigation.navigate('QuoteDetail', { quoteId: item.quoteId });
        }
      }}
    >
      <View style={styles.timeContainer}>
        <Text style={styles.timeText}>
          {format(new Date(item.viewedAt), 'h:mm a')}
        </Text>
      </View>
      
      <View style={styles.quotePreview}>
        {item.quote ? (
          <>
            <Text style={styles.quoteText} numberOfLines={2}>
              "{item.quote.text}"
            </Text>
            <Text style={styles.authorText}>— {item.quote.author}</Text>
          </>
        ) : (
          <Text style={styles.unavailableText}>Quote no longer available</Text>
        )}
        
        {item.mood && (
          <View style={styles.moodContainer}>
            <Text style={styles.moodLabel}>Mood:</Text>
            <Text style={styles.moodText}>{item.mood}</Text>
          </View>
        )}
      </View>
    </TouchableOpacity>
  );

  const renderEmpty = () => (
    <View style={styles.emptyContainer}>
      <Text style={styles.emptyText}>No history yet</Text>
      <Text style={styles.emptySubtext}>
        Your daily quotes will appear here
      </Text>
    </View>
  );

  return (
    <View style={styles.container}>
      <SectionList
        sections={sections}
        keyExtractor={(item, index) => `${item.quoteId}-${index}`}
        renderItem={renderItem}
        renderSectionHeader={renderSectionHeader}
        contentContainerStyle={sections.length === 0 ? styles.emptyList : undefined}
        ListEmptyComponent={renderEmpty}
      />
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#f5f5f5',
  },
  emptyList: {
    flex: 1,
  },
  sectionHeader: {
    backgroundColor: '#f5f5f5',
    paddingHorizontal: 20,
    paddingTop: 20,
    paddingBottom: 10,
  },
  sectionTitle: {
    fontSize: 16,
    fontWeight: '600',
    color: '#666',
  },
  historyItem: {
    flexDirection: 'row',
    backgroundColor: 'white',
    marginHorizontal: 10,
    marginBottom: 10,
    padding: 15,
    borderRadius: 10,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.05,
    shadowRadius: 4,
    elevation: 2,
  },
  timeContainer: {
    marginRight: 15,
    alignItems: 'center',
  },
  timeText: {
    fontSize: 12,
    color: '#999',
  },
  quotePreview: {
    flex: 1,
  },
  quoteText: {
    fontSize: 14,
    color: '#333',
    fontStyle: 'italic',
    marginBottom: 5,
  },
  authorText: {
    fontSize: 12,
    color: '#666',
  },
  unavailableText: {
    fontSize: 14,
    color: '#999',
    fontStyle: 'italic',
  },
  moodContainer: {
    flexDirection: 'row',
    marginTop: 8,
  },
  moodLabel: {
    fontSize: 12,
    color: '#999',
    marginRight: 5,
  },
  moodText: {
    fontSize: 12,
    color: '#007AFF',
    textTransform: 'capitalize',
  },
  emptyContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    padding: 40,
  },
  emptyText: {
    fontSize: 18,
    fontWeight: '600',
    color: '#666',
  },
  emptySubtext: {
    fontSize: 14,
    color: '#999',
    marginTop: 10,
    textAlign: 'center',
  },
});