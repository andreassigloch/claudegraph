import { useState, useEffect } from 'react';
import { Quote } from '@models/index';
import { QuotesService } from '@services/quotesService';
import { StorageService } from '@utils/storage';

export function useQuotes() {
  const [quotes, setQuotes] = useState<Quote[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const fetchDailyQuote = async (category?: string) => {
    setLoading(true);
    setError(null);
    try {
      const quote = await QuotesService.fetchDailyQuote(category);
      setQuotes([quote]);
      return quote;
    } catch (err) {
      setError('Failed to fetch daily quote');
      return null;
    } finally {
      setLoading(false);
    }
  };

  const fetchQuotesByCategory = async (category: string, limit?: number) => {
    setLoading(true);
    setError(null);
    try {
      const fetchedQuotes = await QuotesService.fetchQuotesByCategory(category, limit);
      setQuotes(fetchedQuotes);
      return fetchedQuotes;
    } catch (err) {
      setError('Failed to fetch quotes');
      return [];
    } finally {
      setLoading(false);
    }
  };

  const searchQuotes = async (query: string) => {
    setLoading(true);
    setError(null);
    try {
      const results = await QuotesService.searchQuotes(query);
      setQuotes(results);
      return results;
    } catch (err) {
      setError('Failed to search quotes');
      return [];
    } finally {
      setLoading(false);
    }
  };

  return {
    quotes,
    loading,
    error,
    fetchDailyQuote,
    fetchQuotesByCategory,
    searchQuotes,
  };
}

export function useFavorites() {
  const [favorites, setFavorites] = useState<Quote[]>([]);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    loadFavorites();
  }, []);

  const loadFavorites = async () => {
    setLoading(true);
    try {
      const favs = await StorageService.loadFavorites();
      setFavorites(favs);
    } catch (error) {
      console.error('Error loading favorites:', error);
    } finally {
      setLoading(false);
    }
  };

  const addFavorite = async (quote: Quote) => {
    await StorageService.saveFavorite(quote);
    setFavorites([...favorites, { ...quote, isFavorite: true }]);
  };

  const removeFavorite = async (quoteId: string) => {
    await StorageService.removeFavorite(quoteId);
    setFavorites(favorites.filter(fav => fav.id !== quoteId));
  };

  const isFavorite = (quoteId: string) => {
    return favorites.some(fav => fav.id === quoteId);
  };

  return {
    favorites,
    loading,
    addFavorite,
    removeFavorite,
    isFavorite,
    refresh: loadFavorites,
  };
}