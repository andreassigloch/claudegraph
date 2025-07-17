#!/usr/bin/env python3
"""
Event-Driven Architecture for Code Architecture Analyzer

This module provides event handling infrastructure to decouple analysis
components and enable reactive processing patterns.

Components included:
- EventBus: Central event coordination
- Event: Base event types and data structures
- EventHandlers: Specific event processing logic
- EventSubscriptions: Registration and lifecycle management
"""

from .event_bus import EventBus, EventSubscription
from .events import (
    Event, 
    AnalysisEvent,
    ProjectDiscoveredEvent,
    FileParseEvent,
    ActorDetectedEvent,
    GraphGeneratedEvent,
    ErrorEvent
)
from .handlers import (
    EventHandler,
    LoggingEventHandler,
    MetricsEventHandler,
    CacheEventHandler
)

__all__ = [
    'EventBus',
    'EventSubscription',
    'Event',
    'AnalysisEvent',
    'ProjectDiscoveredEvent',
    'FileParseEvent',
    'ActorDetectedEvent',
    'GraphGeneratedEvent',
    'ErrorEvent',
    'EventHandler',
    'LoggingEventHandler',
    'MetricsEventHandler',
    'CacheEventHandler'
]