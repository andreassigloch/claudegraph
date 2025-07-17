#!/usr/bin/env python3
"""
Event Bus for Code Architecture Analyzer

Provides centralized event coordination with subscription management,
asynchronous processing, and error handling. Enables reactive patterns
throughout the analysis system.
"""

import logging
import asyncio
import threading
import time
from typing import Dict, List, Optional, Any, Callable, Set, Type, Union
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, Future
from collections import defaultdict, deque
from abc import ABC, abstractmethod

from .events import Event

logger = logging.getLogger(__name__)


@dataclass
class EventSubscription:
    """Represents a subscription to events."""
    
    subscription_id: str
    event_types: Set[Type[Event]]
    handler: Callable[[Event], None]
    priority: int = 0
    is_async: bool = False
    filter_func: Optional[Callable[[Event], bool]] = None
    max_retries: int = 3
    timeout_seconds: float = 30.0
    created_at: float = field(default_factory=time.time)
    
    def matches_event(self, event: Event) -> bool:
        """Check if this subscription matches the given event."""
        # Check event type
        if not any(isinstance(event, event_type) for event_type in self.event_types):
            return False
        
        # Apply filter if present
        if self.filter_func and not self.filter_func(event):
            return False
        
        return True


@dataclass
class EventMetrics:
    """Metrics for event processing."""
    
    events_published: int = 0
    events_processed: int = 0
    events_failed: int = 0
    subscriptions_active: int = 0
    processing_errors: int = 0
    average_processing_time: float = 0.0
    queue_size: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary."""
        return {
            'events_published': self.events_published,
            'events_processed': self.events_processed,
            'events_failed': self.events_failed,
            'subscriptions_active': self.subscriptions_active,
            'processing_errors': self.processing_errors,
            'average_processing_time': self.average_processing_time,
            'queue_size': self.queue_size
        }


class EventBus:
    """
    Central event bus for coordinating events throughout the analysis system.
    
    Features:
    - Synchronous and asynchronous event handling
    - Priority-based subscription ordering
    - Event filtering and routing
    - Error handling and retry logic
    - Performance metrics and monitoring
    - Thread-safe operation
    """
    
    def __init__(self, 
                 max_workers: int = 4,
                 queue_size: int = 1000,
                 enable_async: bool = True):
        """Initialize event bus."""
        self.max_workers = max_workers
        self.queue_size = queue_size
        self.enable_async = enable_async
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Subscriptions organized by event type for efficient lookup
        self._subscriptions: Dict[Type[Event], List[EventSubscription]] = defaultdict(list)
        self._all_subscriptions: Dict[str, EventSubscription] = {}
        
        # Event processing
        self._event_queue: deque = deque(maxlen=queue_size)
        self._processing_futures: Dict[str, Future] = {}
        
        # Thread pool for async processing
        if enable_async:
            self._thread_pool = ThreadPoolExecutor(max_workers=max_workers)
        else:
            self._thread_pool = None
        
        # Metrics and monitoring
        self._metrics = EventMetrics()
        self._processing_times: deque = deque(maxlen=100)  # Last 100 processing times
        
        # Shutdown handling
        self._shutdown = False
        
        logger.info(f"Event bus initialized with {max_workers} workers, async={enable_async}")
    
    def subscribe(self, 
                  event_types: Union[Type[Event], List[Type[Event]]], 
                  handler: Callable[[Event], None],
                  priority: int = 0,
                  is_async: bool = False,
                  filter_func: Optional[Callable[[Event], bool]] = None,
                  max_retries: int = 3,
                  timeout_seconds: float = 30.0) -> str:
        """
        Subscribe to events.
        
        Args:
            event_types: Event type(s) to subscribe to
            handler: Function to handle events
            priority: Subscription priority (higher = processed first)
            is_async: Whether to process events asynchronously
            filter_func: Optional filter function
            max_retries: Maximum retry attempts on failure
            timeout_seconds: Timeout for async processing
            
        Returns:
            Subscription ID
        """
        with self._lock:
            # Normalize event types to set
            if not isinstance(event_types, list):
                event_types = [event_types]
            event_type_set = set(event_types)
            
            # Create subscription
            subscription_id = f"sub_{int(time.time() * 1000000)}"
            subscription = EventSubscription(
                subscription_id=subscription_id,
                event_types=event_type_set,
                handler=handler,
                priority=priority,
                is_async=is_async and self.enable_async,
                filter_func=filter_func,
                max_retries=max_retries,
                timeout_seconds=timeout_seconds
            )
            
            # Store subscription
            self._all_subscriptions[subscription_id] = subscription
            
            # Add to event type mappings
            for event_type in event_type_set:
                self._subscriptions[event_type].append(subscription)
                # Sort by priority (descending)
                self._subscriptions[event_type].sort(key=lambda s: s.priority, reverse=True)
            
            self._metrics.subscriptions_active = len(self._all_subscriptions)
            
            logger.debug(f"Subscribed {subscription_id} to {len(event_type_set)} event types")
            return subscription_id
    
    def unsubscribe(self, subscription_id: str) -> bool:
        """
        Unsubscribe from events.
        
        Args:
            subscription_id: ID of subscription to remove
            
        Returns:
            True if subscription was found and removed
        """
        with self._lock:
            subscription = self._all_subscriptions.get(subscription_id)
            if not subscription:
                return False
            
            # Remove from event type mappings
            for event_type in subscription.event_types:
                if event_type in self._subscriptions:
                    self._subscriptions[event_type] = [
                        s for s in self._subscriptions[event_type] 
                        if s.subscription_id != subscription_id
                    ]
            
            # Remove from all subscriptions
            del self._all_subscriptions[subscription_id]
            
            self._metrics.subscriptions_active = len(self._all_subscriptions)
            
            logger.debug(f"Unsubscribed {subscription_id}")
            return True
    
    def publish(self, event: Event) -> bool:
        """
        Publish an event to all matching subscribers.
        
        Args:
            event: Event to publish
            
        Returns:
            True if event was published successfully
        """
        if self._shutdown:
            logger.warning("Event bus is shutdown, ignoring event")
            return False
        
        try:
            with self._lock:
                # Find matching subscriptions
                matching_subscriptions = []
                
                # Check exact type matches
                event_type = type(event)
                if event_type in self._subscriptions:
                    for subscription in self._subscriptions[event_type]:
                        if subscription.matches_event(event):
                            matching_subscriptions.append(subscription)
                
                # Check parent class matches
                for parent_type in event_type.__mro__[1:]:  # Skip the event itself
                    if parent_type in self._subscriptions:
                        for subscription in self._subscriptions[parent_type]:
                            if (subscription not in matching_subscriptions and 
                                subscription.matches_event(event)):
                                matching_subscriptions.append(subscription)
                
                if not matching_subscriptions:
                    logger.debug(f"No subscribers for event {event.event_type}")
                    return True
                
                # Sort by priority
                matching_subscriptions.sort(key=lambda s: s.priority, reverse=True)
                
                # Process subscriptions
                processed = 0
                for subscription in matching_subscriptions:
                    try:
                        if subscription.is_async:
                            self._process_async(event, subscription)
                        else:
                            self._process_sync(event, subscription)
                        processed += 1
                    except Exception as e:
                        logger.error(f"Failed to process event {event.event_id} "
                                   f"for subscription {subscription.subscription_id}: {e}")
                        self._metrics.processing_errors += 1
                
                self._metrics.events_published += 1
                self._metrics.events_processed += processed
                
                logger.debug(f"Published event {event.event_type} to {processed} subscribers")
                return True
                
        except Exception as e:
            logger.error(f"Failed to publish event {event.event_id}: {e}")
            self._metrics.events_failed += 1
            return False
    
    def _process_sync(self, event: Event, subscription: EventSubscription):
        """Process event synchronously."""
        start_time = time.time()
        
        try:
            subscription.handler(event)
            processing_time = time.time() - start_time
            self._update_processing_time(processing_time)
            
        except Exception as e:
            logger.error(f"Sync handler failed for event {event.event_id}: {e}")
            raise
    
    def _process_async(self, event: Event, subscription: EventSubscription):
        """Process event asynchronously."""
        if not self._thread_pool:
            logger.warning("Async processing requested but thread pool not available")
            self._process_sync(event, subscription)
            return
        
        def async_handler():
            start_time = time.time()
            retries = 0
            
            while retries <= subscription.max_retries:
                try:
                    subscription.handler(event)
                    processing_time = time.time() - start_time
                    self._update_processing_time(processing_time)
                    return
                    
                except Exception as e:
                    retries += 1
                    if retries <= subscription.max_retries:
                        logger.warning(f"Async handler failed (attempt {retries}): {e}")
                        time.sleep(0.1 * retries)  # Exponential backoff
                    else:
                        logger.error(f"Async handler failed after {retries} attempts: {e}")
                        raise
        
        # Submit to thread pool
        future = self._thread_pool.submit(async_handler)
        self._processing_futures[f"{event.event_id}_{subscription.subscription_id}"] = future
        
        # Clean up completed futures
        self._cleanup_futures()
    
    def _update_processing_time(self, processing_time: float):
        """Update processing time metrics."""
        self._processing_times.append(processing_time)
        if self._processing_times:
            self._metrics.average_processing_time = sum(self._processing_times) / len(self._processing_times)
    
    def _cleanup_futures(self):
        """Clean up completed async processing futures."""
        completed_keys = []
        for key, future in self._processing_futures.items():
            if future.done():
                completed_keys.append(key)
        
        for key in completed_keys:
            del self._processing_futures[key]
    
    def wait_for_completion(self, timeout: Optional[float] = None) -> bool:
        """
        Wait for all async events to complete processing.
        
        Args:
            timeout: Maximum time to wait in seconds
            
        Returns:
            True if all events completed within timeout
        """
        if not self._thread_pool:
            return True
        
        start_time = time.time()
        
        while self._processing_futures:
            if timeout and (time.time() - start_time) > timeout:
                return False
            
            self._cleanup_futures()
            time.sleep(0.01)  # Small delay
        
        return True
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get event bus metrics."""
        with self._lock:
            self._metrics.queue_size = len(self._event_queue)
            return self._metrics.to_dict()
    
    def get_subscription_info(self) -> List[Dict[str, Any]]:
        """Get information about all active subscriptions."""
        with self._lock:
            subscriptions = []
            for subscription in self._all_subscriptions.values():
                subscriptions.append({
                    'subscription_id': subscription.subscription_id,
                    'event_types': [et.__name__ for et in subscription.event_types],
                    'priority': subscription.priority,
                    'is_async': subscription.is_async,
                    'has_filter': subscription.filter_func is not None,
                    'max_retries': subscription.max_retries,
                    'timeout_seconds': subscription.timeout_seconds,
                    'created_at': subscription.created_at
                })
            return subscriptions
    
    def clear_subscriptions(self):
        """Clear all subscriptions."""
        with self._lock:
            self._subscriptions.clear()
            self._all_subscriptions.clear()
            self._metrics.subscriptions_active = 0
            logger.info("All subscriptions cleared")
    
    def shutdown(self, timeout: float = 30.0):
        """
        Shutdown the event bus gracefully.
        
        Args:
            timeout: Maximum time to wait for completion
        """
        logger.info("Shutting down event bus")
        self._shutdown = True
        
        # Wait for async operations to complete
        if not self.wait_for_completion(timeout):
            logger.warning(f"Event bus shutdown timeout after {timeout}s")
        
        # Shutdown thread pool
        if self._thread_pool:
            self._thread_pool.shutdown(wait=True)
        
        logger.info("Event bus shutdown complete")


# Convenience functions for common event bus operations
_default_event_bus: Optional[EventBus] = None


def get_default_event_bus() -> EventBus:
    """Get the default global event bus instance."""
    global _default_event_bus
    if _default_event_bus is None:
        _default_event_bus = EventBus()
    return _default_event_bus


def publish_event(event: Event) -> bool:
    """Publish an event to the default event bus."""
    return get_default_event_bus().publish(event)


def subscribe_to_events(event_types: Union[Type[Event], List[Type[Event]]], 
                       handler: Callable[[Event], None],
                       **kwargs) -> str:
    """Subscribe to events on the default event bus."""
    return get_default_event_bus().subscribe(event_types, handler, **kwargs)


def unsubscribe_from_events(subscription_id: str) -> bool:
    """Unsubscribe from events on the default event bus."""
    return get_default_event_bus().unsubscribe(subscription_id)