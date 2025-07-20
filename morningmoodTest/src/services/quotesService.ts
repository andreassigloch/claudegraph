import axios from 'axios';
import { Quote } from '@models/index';

const API_BASE_URL = 'https://api.quotable.io'; // Example API

export class QuotesService {
  /**
   * Fetch daily quote from external API
   */
  static async fetchDailyQuote(category?: string): Promise<Quote> {
    try {
      const params = category ? { tags: category } : {};
      const response = await axios.get(`${API_BASE_URL}/random`, { params });
      
      return {
        id: response.data._id,
        text: response.data.content,
        author: response.data.author,
        category: response.data.tags[0] || 'general',
        date: new Date()
      };
    } catch (error) {
      console.error('Error fetching quote:', error);
      // Return fallback quote
      return {
        id: 'fallback-1',
        text: 'Every day is a new beginning.',
        author: 'Unknown',
        category: 'inspirational',
        date: new Date()
      };
    }
  }

  /**
   * Fetch quotes by category
   */
  static async fetchQuotesByCategory(category: string, limit: number = 10): Promise<Quote[]> {
    try {
      const response = await axios.get(`${API_BASE_URL}/quotes`, {
        params: { tags: category, limit }
      });
      
      return response.data.results.map((quote: any) => ({
        id: quote._id,
        text: quote.content,
        author: quote.author,
        category: quote.tags[0] || category,
        date: new Date()
      }));
    } catch (error) {
      console.error('Error fetching quotes by category:', error);
      return [];
    }
  }

  /**
   * Search quotes by keyword
   */
  static async searchQuotes(query: string): Promise<Quote[]> {
    try {
      const response = await axios.get(`${API_BASE_URL}/search/quotes`, {
        params: { query, limit: 20 }
      });
      
      return response.data.results.map((quote: any) => ({
        id: quote._id,
        text: quote.content,
        author: quote.author,
        category: quote.tags[0] || 'general',
        date: new Date()
      }));
    } catch (error) {
      console.error('Error searching quotes:', error);
      return [];
    }
  }
}