# Morning Mood - Daily Wisdom App

A React Native/Expo application that provides daily inspirational quotes with features for favorites, history tracking, and personalized notifications.

## Project Structure

```
morningmoodTest/
├── src/
│   ├── components/       # Reusable UI components
│   │   └── QuoteCard.tsx
│   ├── screens/         # Screen components for each route
│   │   ├── HomeScreen.tsx
│   │   ├── FavoritesScreen.tsx
│   │   ├── HistoryScreen.tsx
│   │   ├── SettingsScreen.tsx
│   │   ├── QuoteDetailScreen.tsx
│   │   └── CategoryScreen.tsx
│   ├── services/        # External service integrations
│   │   ├── quotesService.ts      # Quotes API integration
│   │   ├── notificationService.ts # Push notifications
│   │   └── musicService.ts       # Background music player
│   ├── utils/           # Utility functions
│   │   └── storage.ts   # AsyncStorage operations
│   ├── models/          # TypeScript data models
│   │   └── index.ts     # Quote, UserPrefs, etc.
│   ├── navigation/      # Navigation configuration
│   │   └── AppNavigator.tsx
│   ├── hooks/           # Custom React hooks
│   │   └── useQuotes.ts
│   └── constants/       # App constants
│       └── index.ts
├── assets/              # Images, fonts, etc.
├── App.tsx             # Main app component
├── package.json        # Dependencies
├── app.json           # Expo configuration
├── tsconfig.json      # TypeScript configuration
└── babel.config.js    # Babel configuration
```

## Features

- **Daily Quotes**: Automatically fetch and display a new inspirational quote each day
- **Categories**: Browse quotes by different categories (Inspirational, Motivational, Wisdom, etc.)
- **Favorites**: Save your favorite quotes for quick access
- **History**: Track all quotes you've viewed with timestamps
- **Notifications**: Receive daily quote notifications at your preferred time
- **Background Music**: Optional ambient music while browsing
- **Offline Support**: Cached quotes available when offline

## Getting Started

1. Install dependencies:
   ```bash
   npm install
   ```

2. Start the development server:
   ```bash
   npm start
   ```

3. Run on iOS:
   ```bash
   npm run ios
   ```

4. Run on Android:
   ```bash
   npm run android
   ```

## Architecture Alignment

This implementation follows the ClaudeGraph architecture design with:

- **Use Cases**: Each screen represents a primary use case (View Daily Quote, Manage Favorites, etc.)
- **Actors**: External services (QuotesAPI, NotificationService, MusicService, LocalStorage)
- **Functions**: All defined functions implemented as methods in services and components
- **Data Flow**: Proper separation between UI components and business logic
- **Schemas**: TypeScript interfaces matching the architecture data models

## Key Components

### Services
- **QuotesService**: Handles all quote-related API calls
- **NotificationService**: Manages push notifications for daily quotes
- **MusicService**: Controls background music playback
- **StorageService**: Manages local data persistence

### Screens
- **HomeScreen**: Displays the daily quote with actions
- **FavoritesScreen**: Shows all saved favorite quotes
- **HistoryScreen**: Timeline of previously viewed quotes
- **SettingsScreen**: User preferences and app configuration
- **QuoteDetailScreen**: Full quote view with sharing options
- **CategoryScreen**: Browse quotes by specific category

## Configuration

The app uses several configuration files:
- `app.json`: Expo-specific configuration
- `src/constants/index.ts`: App constants and configuration values
- Environment variables can be added via `.env` file (not included in repo)

## Development Notes

- The app uses TypeScript for type safety
- Navigation is handled by React Navigation
- State management is primarily through React hooks and local component state
- AsyncStorage is used for data persistence
- The quotes API endpoint should be updated with a real service URL

## Future Enhancements

- User authentication and cloud sync
- Social sharing features
- Quote creation and submission
- Themed quote collections
- Widget support for iOS/Android