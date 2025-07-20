// Load PersonalActivityTracker Architecture into Neo4j
// Architecture: Mac activity tracking app with calendar integration and wellness monitoring

// 1. Create System
CREATE (sys:SYS {
    Name: "PersonalActivityTracker",
    ID: "SYS_PersonalActivityTracker",
    Descr: "Mac app for personal activity tracking with calendar integration and wellness monitoring"
})

// 2. Create Use Cases
CREATE (uc1:UC {
    Name: "DetectActivityBlocks",
    ID: "UC_DetectActivityBlocks", 
    Descr: "Monitor app focus continuously, log activities, and trigger classification prompts after 15min stabilization"
})

CREATE (uc2:UC {
    Name: "ClassifyActivity",
    ID: "UC_ClassifyActivity",
    Descr: "Auto-classify activities using smart defaults, scan git for project names, prompt user only when uncertain"
})

CREATE (uc3:UC {
    Name: "CreateCalendarEntry", 
    ID: "UC_CreateCalendarEntry",
    Descr: "Generate calendar events with project names and store permanently in Calendar.app"
})

CREATE (uc4:UC {
    Name: "ProvideWellnessNudges",
    ID: "UC_ProvideWellnessNudges", 
    Descr: "Monitor session duration, detect long blocks >1h, trigger break notifications and Apple Timer integration"
})

CREATE (uc5:UC {
    Name: "ManageDataRetention",
    ID: "UC_ManageDataRetention",
    Descr: "Separate daily cleanup cycle to delete local logs after 7 days, ensuring only calendar entries remain"
})

// 3. Create Actors
CREATE (a1:ACTOR {
    Name: "User",
    ID: "ACTOR_User",
    Descr: "Person using the app who receives classification prompts and provides activity categorization"
})

CREATE (a2:ACTOR {
    Name: "CalendarApp", 
    ID: "ACTOR_CalendarApp",
    Descr: "macOS Calendar.app for permanent storage of classified activity blocks"
})

CREATE (a3:ACTOR {
    Name: "GitSystem",
    ID: "ACTOR_GitSystem", 
    Descr: "Local git repositories providing project name detection through commit analysis"
})

CREATE (a4:ACTOR {
    Name: "SystemMonitor",
    ID: "ACTOR_SystemMonitor",
    Descr: "macOS Accessibility APIs for app focus monitoring and window tracking"
})

CREATE (a5:ACTOR {
    Name: "NotificationCenter",
    ID: "ACTOR_NotificationCenter",
    Descr: "macOS notification system for break reminders and classification prompts"
})

CREATE (a6:ACTOR {
    Name: "AppleTimer",
    ID: "ACTOR_AppleTimer",
    Descr: "macOS built-in timer for break countdown functionality"
})

// 4. Create Schemas
CREATE (s1:SCHEMA {
    Name: "ActivityBlock",
    ID: "SCHEMA_ActivityBlock",
    Descr: "Data structure for tracked activity sessions with classification and timing",
    Struct: '{"startTime": "datetime", "endTime": "datetime", "application": "string", "classification": "enum[Business, Private, Coding, Meeting]", "projectName": "string?", "autoClassified": "boolean", "gitCommits": ["string"]}'
})

CREATE (s2:SCHEMA {
    Name: "ClassificationRule",
    ID: "SCHEMA_ClassificationRule", 
    Descr: "Rules for automatic activity classification based on application patterns",
    Struct: '{"applicationPattern": "string", "autoClassification": "string?", "confidence": "number", "requiresUserInput": "boolean", "dropdownOptions": ["string"], "checkGit": "boolean?"}'
})

CREATE (s3:SCHEMA {
    Name: "CalendarEvent",
    ID: "SCHEMA_CalendarEvent",
    Descr: "Calendar entry format for permanent activity storage", 
    Struct: '{"title": "string", "startDate": "datetime", "endDate": "datetime", "notes": "string", "calendarId": "string"}'
})

// 5. Create Function Chains
CREATE (fc1:FCHAIN {
    Name: "ActivityDetectionFlow",
    ID: "FCHAIN_ActivityDetectionFlow",
    Descr: "Continuous monitoring and stabilized classification trigger workflow"
})

CREATE (fc2:FCHAIN {
    Name: "ClassificationFlow", 
    ID: "FCHAIN_ClassificationFlow",
    Descr: "Smart classification with git integration and selective user prompting"
})

CREATE (fc3:FCHAIN {
    Name: "CalendarIntegrationFlow",
    ID: "FCHAIN_CalendarIntegrationFlow",
    Descr: "Calendar event creation and permanent storage workflow"
})

CREATE (fc4:FCHAIN {
    Name: "WellnessMonitoringFlow",
    ID: "FCHAIN_WellnessMonitoringFlow", 
    Descr: "Session tracking, break detection, and Apple Timer integration"
})

CREATE (fc5:FCHAIN {
    Name: "DataRetentionFlow",
    ID: "FCHAIN_DataRetentionFlow",
    Descr: "Separate daily cleanup cycle for 7-day log retention and privacy compliance"
})

// 6. Create Functions for ActivityDetectionFlow
CREATE (f1:FUNC {
    Name: "MonitorAppFocus",
    ID: "FUNC_MonitorAppFocus",
    Descr: "Continuously track active application and window focus changes"
})

CREATE (f2:FUNC {
    Name: "LogActivityContinuously", 
    ID: "FUNC_LogActivityContinuously",
    Descr: "Record all activity data immediately without waiting for classification"
})

CREATE (f3:FUNC {
    Name: "Wait15MinStabilization",
    ID: "FUNC_Wait15MinStabilization",
    Descr: "Delay classification prompts until 15 minutes of stable activity detected"
})

CREATE (f4:FUNC {
    Name: "TriggerClassificationPrompt",
    ID: "FUNC_TriggerClassificationPrompt", 
    Descr: "Initiate user classification request after stabilization period"
})

// 7. Create Functions for ClassificationFlow
CREATE (f5:FUNC {
    Name: "AnalyzeApplication",
    ID: "FUNC_AnalyzeApplication",
    Descr: "Identify application type and determine classification approach"
})

CREATE (f6:FUNC {
    Name: "ApplySmartDefaults",
    ID: "FUNC_ApplySmartDefaults",
    Descr: "Apply confidence-based auto-classification (Office→Business 90%, Terminal→Coding 100%)"
})

CREATE (f7:FUNC {
    Name: "ScanGitForProjectName",
    ID: "FUNC_ScanGitForProjectName", 
    Descr: "Analyze git commits in active directory to extract project name for coding activities"
})

CREATE (f8:FUNC {
    Name: "PromptUserOnlyIfUncertain",
    ID: "FUNC_PromptUserOnlyIfUncertain",
    Descr: "Show classification dropdown only when auto-classification confidence is low"
})

CREATE (f9:FUNC {
    Name: "FinalizeClassification",
    ID: "FUNC_FinalizeClassification",
    Descr: "Complete activity classification and prepare for calendar integration"
})

// 8. Create Functions for CalendarIntegrationFlow
CREATE (f10:FUNC {
    Name: "FormatActivityWithProjectName",
    ID: "FUNC_FormatActivityWithProjectName",
    Descr: "Create descriptive activity title including project name when available"
})

CREATE (f11:FUNC {
    Name: "CreateCalendarEvent",
    ID: "FUNC_CreateCalendarEvent", 
    Descr: "Generate calendar entry with proper timing and description"
})

CREATE (f12:FUNC {
    Name: "VerifyCalendarStorage",
    ID: "FUNC_VerifyCalendarStorage",
    Descr: "Confirm successful calendar entry creation and accessibility"
})

CREATE (f13:FUNC {
    Name: "MarkForRetentionTracking",
    ID: "FUNC_MarkForRetentionTracking",
    Descr: "Tag local logs for 7-day retention countdown after calendar storage"
})

// 9. Create Functions for WellnessMonitoringFlow
CREATE (f14:FUNC {
    Name: "TrackSessionDuration",
    ID: "FUNC_TrackSessionDuration",
    Descr: "Monitor continuous activity duration for wellness notifications"
})

CREATE (f15:FUNC {
    Name: "DetectLongSessions", 
    ID: "FUNC_DetectLongSessions",
    Descr: "Identify activity blocks exceeding 1 hour duration"
})

CREATE (f16:FUNC {
    Name: "TriggerBreakNotification",
    ID: "FUNC_TriggerBreakNotification",
    Descr: "Send break reminder notification for long activity sessions"
})

CREATE (f17:FUNC {
    Name: "LaunchAppleTimer",
    ID: "FUNC_LaunchAppleTimer",
    Descr: "Start macOS timer for break countdown or next work session"
})

// 10. Create Functions for DataRetentionFlow
CREATE (f18:FUNC {
    Name: "ScheduleDailyCleanupCheck",
    ID: "FUNC_ScheduleDailyCleanupCheck",
    Descr: "Run daily background task to identify expired local logs"
})

CREATE (f19:FUNC {
    Name: "IdentifyExpiredLogs",
    ID: "FUNC_IdentifyExpiredLogs", 
    Descr: "Find local activity logs older than 7 days with confirmed calendar storage"
})

CREATE (f20:FUNC {
    Name: "DeleteLocalData",
    ID: "FUNC_DeleteLocalData",
    Descr: "Permanently remove expired local logs while preserving calendar entries"
})

CREATE (f21:FUNC {
    Name: "VerifyOnlyCalendarRemains",
    ID: "FUNC_VerifyOnlyCalendarRemains",
    Descr: "Confirm successful data deletion and calendar-only retention"
})

// 11. Connect System to Use Cases
MATCH (sys:SYS {ID: "SYS_PersonalActivityTracker"})
MATCH (uc1:UC {ID: "UC_DetectActivityBlocks"})
MATCH (uc2:UC {ID: "UC_ClassifyActivity"})
MATCH (uc3:UC {ID: "UC_CreateCalendarEntry"})
MATCH (uc4:UC {ID: "UC_ProvideWellnessNudges"})
MATCH (uc5:UC {ID: "UC_ManageDataRetention"})
CREATE (sys)-[:compose]->(uc1)
CREATE (sys)-[:compose]->(uc2)
CREATE (sys)-[:compose]->(uc3)
CREATE (sys)-[:compose]->(uc4)
CREATE (sys)-[:compose]->(uc5)

// 12. CRITICAL: Connect System to Actors
MATCH (sys:SYS {ID: "SYS_PersonalActivityTracker"})
MATCH (a1:ACTOR {ID: "ACTOR_User"})
MATCH (a2:ACTOR {ID: "ACTOR_CalendarApp"})
MATCH (a3:ACTOR {ID: "ACTOR_GitSystem"})
MATCH (a4:ACTOR {ID: "ACTOR_SystemMonitor"})
MATCH (a5:ACTOR {ID: "ACTOR_NotificationCenter"})
MATCH (a6:ACTOR {ID: "ACTOR_AppleTimer"})
CREATE (sys)-[:compose]->(a1)
CREATE (sys)-[:compose]->(a2)
CREATE (sys)-[:compose]->(a3)
CREATE (sys)-[:compose]->(a4)
CREATE (sys)-[:compose]->(a5)
CREATE (sys)-[:compose]->(a6)

// 13. CRITICAL: Connect System to Schemas
MATCH (sys:SYS {ID: "SYS_PersonalActivityTracker"})
MATCH (s1:SCHEMA {ID: "SCHEMA_ActivityBlock"})
MATCH (s2:SCHEMA {ID: "SCHEMA_ClassificationRule"})
MATCH (s3:SCHEMA {ID: "SCHEMA_CalendarEvent"})
CREATE (sys)-[:compose]->(s1)
CREATE (sys)-[:compose]->(s2)
CREATE (sys)-[:compose]->(s3)

// 14. Connect Use Cases to Function Chains
MATCH (uc1:UC {ID: "UC_DetectActivityBlocks"})
MATCH (uc2:UC {ID: "UC_ClassifyActivity"})
MATCH (uc3:UC {ID: "UC_CreateCalendarEntry"})
MATCH (uc4:UC {ID: "UC_ProvideWellnessNudges"})
MATCH (uc5:UC {ID: "UC_ManageDataRetention"})
MATCH (fc1:FCHAIN {ID: "FCHAIN_ActivityDetectionFlow"})
MATCH (fc2:FCHAIN {ID: "FCHAIN_ClassificationFlow"})
MATCH (fc3:FCHAIN {ID: "FCHAIN_CalendarIntegrationFlow"})
MATCH (fc4:FCHAIN {ID: "FCHAIN_WellnessMonitoringFlow"})
MATCH (fc5:FCHAIN {ID: "FCHAIN_DataRetentionFlow"})
CREATE (uc1)-[:compose]->(fc1)
CREATE (uc2)-[:compose]->(fc2)
CREATE (uc3)-[:compose]->(fc3)
CREATE (uc4)-[:compose]->(fc4)
CREATE (uc5)-[:compose]->(fc5)

// 15. Connect Function Chains to Functions
MATCH (fc1:FCHAIN {ID: "FCHAIN_ActivityDetectionFlow"})
MATCH (f1:FUNC {ID: "FUNC_MonitorAppFocus"})
MATCH (f2:FUNC {ID: "FUNC_LogActivityContinuously"})
MATCH (f3:FUNC {ID: "FUNC_Wait15MinStabilization"})
MATCH (f4:FUNC {ID: "FUNC_TriggerClassificationPrompt"})
CREATE (fc1)-[:compose]->(f1)
CREATE (fc1)-[:compose]->(f2)
CREATE (fc1)-[:compose]->(f3)
CREATE (fc1)-[:compose]->(f4)

MATCH (fc2:FCHAIN {ID: "FCHAIN_ClassificationFlow"})
MATCH (f5:FUNC {ID: "FUNC_AnalyzeApplication"})
MATCH (f6:FUNC {ID: "FUNC_ApplySmartDefaults"})
MATCH (f7:FUNC {ID: "FUNC_ScanGitForProjectName"})
MATCH (f8:FUNC {ID: "FUNC_PromptUserOnlyIfUncertain"})
MATCH (f9:FUNC {ID: "FUNC_FinalizeClassification"})
CREATE (fc2)-[:compose]->(f5)
CREATE (fc2)-[:compose]->(f6)
CREATE (fc2)-[:compose]->(f7)
CREATE (fc2)-[:compose]->(f8)
CREATE (fc2)-[:compose]->(f9)

MATCH (fc3:FCHAIN {ID: "FCHAIN_CalendarIntegrationFlow"})
MATCH (f10:FUNC {ID: "FUNC_FormatActivityWithProjectName"})
MATCH (f11:FUNC {ID: "FUNC_CreateCalendarEvent"})
MATCH (f12:FUNC {ID: "FUNC_VerifyCalendarStorage"})
MATCH (f13:FUNC {ID: "FUNC_MarkForRetentionTracking"})
CREATE (fc3)-[:compose]->(f10)
CREATE (fc3)-[:compose]->(f11)
CREATE (fc3)-[:compose]->(f12)
CREATE (fc3)-[:compose]->(f13)

MATCH (fc4:FCHAIN {ID: "FCHAIN_WellnessMonitoringFlow"})
MATCH (f14:FUNC {ID: "FUNC_TrackSessionDuration"})
MATCH (f15:FUNC {ID: "FUNC_DetectLongSessions"})
MATCH (f16:FUNC {ID: "FUNC_TriggerBreakNotification"})
MATCH (f17:FUNC {ID: "FUNC_LaunchAppleTimer"})
CREATE (fc4)-[:compose]->(f14)
CREATE (fc4)-[:compose]->(f15)
CREATE (fc4)-[:compose]->(f16)
CREATE (fc4)-[:compose]->(f17)

MATCH (fc5:FCHAIN {ID: "FCHAIN_DataRetentionFlow"})
MATCH (f18:FUNC {ID: "FUNC_ScheduleDailyCleanupCheck"})
MATCH (f19:FUNC {ID: "FUNC_IdentifyExpiredLogs"})
MATCH (f20:FUNC {ID: "FUNC_DeleteLocalData"})
MATCH (f21:FUNC {ID: "FUNC_VerifyOnlyCalendarRemains"})
CREATE (fc5)-[:compose]->(f18)
CREATE (fc5)-[:compose]->(f19)
CREATE (fc5)-[:compose]->(f20)
CREATE (fc5)-[:compose]->(f21)

// 16. Create Flow Relationships within ActivityDetectionFlow
MATCH (f1:FUNC {ID: "FUNC_MonitorAppFocus"})
MATCH (f2:FUNC {ID: "FUNC_LogActivityContinuously"})
MATCH (f3:FUNC {ID: "FUNC_Wait15MinStabilization"})
MATCH (f4:FUNC {ID: "FUNC_TriggerClassificationPrompt"})
CREATE (f1)-[:flow {
    FlowDescr: "App focus changes trigger continuous activity logging",
    FlowDef: "focus_change_events → log_activity(timestamp, app, window)"
}]->(f2)
CREATE (f2)-[:flow {
    FlowDescr: "Logged activities feed into stabilization delay logic", 
    FlowDef: "activity_log → check_stability_duration(15_minutes)"
}]->(f3)
CREATE (f3)-[:flow {
    FlowDescr: "Stabilized activities trigger classification prompts",
    FlowDef: "stable_activity_block → trigger_user_prompt(activity)"
}]->(f4)

// 17. Create Flow Relationships within ClassificationFlow
MATCH (f5:FUNC {ID: "FUNC_AnalyzeApplication"})
MATCH (f6:FUNC {ID: "FUNC_ApplySmartDefaults"})
MATCH (f7:FUNC {ID: "FUNC_ScanGitForProjectName"})
MATCH (f8:FUNC {ID: "FUNC_PromptUserOnlyIfUncertain"})
MATCH (f9:FUNC {ID: "FUNC_FinalizeClassification"})
CREATE (f5)-[:flow {
    FlowDescr: "Application analysis feeds into smart default classification",
    FlowDef: "app_analysis → apply_classification_rules(confidence_level)"
}]->(f6)
CREATE (f6)-[:flow {
    FlowDescr: "Default classification triggers git project scanning for coding activities",
    FlowDef: "classification_result → scan_git_repos() if coding_activity"
}]->(f7)
CREATE (f7)-[:flow {
    FlowDescr: "Git analysis results determine if user prompt is needed",
    FlowDef: "project_name + confidence → prompt_user() if uncertain"
}]->(f8)
CREATE (f8)-[:flow {
    FlowDescr: "User input or auto-classification finalizes activity categorization",
    FlowDef: "user_selection || auto_classification → final_classification"
}]->(f9)

// 18. Create Flow Relationships within CalendarIntegrationFlow
MATCH (f10:FUNC {ID: "FUNC_FormatActivityWithProjectName"})
MATCH (f11:FUNC {ID: "FUNC_CreateCalendarEvent"})
MATCH (f12:FUNC {ID: "FUNC_VerifyCalendarStorage"})
MATCH (f13:FUNC {ID: "FUNC_MarkForRetentionTracking"})
CREATE (f10)-[:flow {
    FlowDescr: "Formatted activity description generates calendar event",
    FlowDef: "activity_title + project_name → create_calendar_entry()"
}]->(f11)
CREATE (f11)-[:flow {
    FlowDescr: "Calendar event creation triggers storage verification",
    FlowDef: "calendar_event → verify_storage_success(event_id)"
}]->(f12)
CREATE (f12)-[:flow {
    FlowDescr: "Verified calendar storage enables local log retention tracking",
    FlowDef: "storage_confirmed → mark_for_cleanup(activity_log, 7_days)"
}]->(f13)

// 19. Create Flow Relationships within WellnessMonitoringFlow
MATCH (f14:FUNC {ID: "FUNC_TrackSessionDuration"})
MATCH (f15:FUNC {ID: "FUNC_DetectLongSessions"})
MATCH (f16:FUNC {ID: "FUNC_TriggerBreakNotification"})
MATCH (f17:FUNC {ID: "FUNC_LaunchAppleTimer"})
CREATE (f14)-[:flow {
    FlowDescr: "Session duration tracking identifies long activity blocks",
    FlowDef: "session_duration → check_long_session(1_hour_threshold)"
}]->(f15)
CREATE (f15)-[:flow {
    FlowDescr: "Long session detection triggers break reminder notifications",
    FlowDef: "long_session_detected → send_break_notification()"
}]->(f16)
CREATE (f16)-[:flow {
    FlowDescr: "Break notification offers Apple Timer for break countdown",
    FlowDef: "break_notification → launch_timer(break_duration) if user_accepts"
}]->(f17)

// 20. Create Flow Relationships within DataRetentionFlow
MATCH (f18:FUNC {ID: "FUNC_ScheduleDailyCleanupCheck"})
MATCH (f19:FUNC {ID: "FUNC_IdentifyExpiredLogs"})
MATCH (f20:FUNC {ID: "FUNC_DeleteLocalData"})
MATCH (f21:FUNC {ID: "FUNC_VerifyOnlyCalendarRemains"})
CREATE (f18)-[:flow {
    FlowDescr: "Daily cleanup schedule triggers expired log identification",
    FlowDef: "daily_cleanup_trigger → identify_logs_older_than(7_days)"
}]->(f19)
CREATE (f19)-[:flow {
    FlowDescr: "Expired log identification leads to secure data deletion",
    FlowDef: "expired_logs_list → secure_delete(local_files)"
}]->(f20)
CREATE (f20)-[:flow {
    FlowDescr: "Local data deletion is verified to ensure only calendar data remains",
    FlowDef: "deletion_complete → verify_calendar_only_retention()"
}]->(f21)

// 21. Create Actor Flow Relationships
MATCH (a4:ACTOR {ID: "ACTOR_SystemMonitor"})
MATCH (f1:FUNC {ID: "FUNC_MonitorAppFocus"})
CREATE (a4)-[:flow {
    FlowDescr: "System monitor provides app focus events",
    FlowDef: "accessibility_api.focus_change → monitor_app_focus()"
}]->(f1)

MATCH (f8:FUNC {ID: "FUNC_PromptUserOnlyIfUncertain"})
MATCH (a1:ACTOR {ID: "ACTOR_User"})
CREATE (f8)-[:flow {
    FlowDescr: "Classification prompt displayed to user",
    FlowDef: "show_dropdown(classification_options) → user_selection"
}]->(a1)

MATCH (a3:ACTOR {ID: "ACTOR_GitSystem"})
MATCH (f7:FUNC {ID: "FUNC_ScanGitForProjectName"})
CREATE (a3)-[:flow {
    FlowDescr: "Git system provides commit data for project analysis", 
    FlowDef: "git_log() → extract_project_name(commits)"
}]->(f7)

MATCH (f11:FUNC {ID: "FUNC_CreateCalendarEvent"})
MATCH (a2:ACTOR {ID: "ACTOR_CalendarApp"})
CREATE (f11)-[:flow {
    FlowDescr: "Calendar event sent to Calendar.app for storage",
    FlowDef: "calendar_event → calendar_app.create_event()"
}]->(a2)

MATCH (f16:FUNC {ID: "FUNC_TriggerBreakNotification"})
MATCH (a5:ACTOR {ID: "ACTOR_NotificationCenter"})
CREATE (f16)-[:flow {
    FlowDescr: "Break reminder sent via notification center",
    FlowDef: "break_reminder → notification_center.show()"
}]->(a5)

MATCH (f17:FUNC {ID: "FUNC_LaunchAppleTimer"})
MATCH (a6:ACTOR {ID: "ACTOR_AppleTimer"})
CREATE (f17)-[:flow {
    FlowDescr: "Timer launch request sent to Apple Timer",
    FlowDef: "timer_request → apple_timer.start(duration)"
}]->(a6)

// 22. Verify the architecture is loaded
MATCH (sys:SYS {Name: "PersonalActivityTracker"})
OPTIONAL MATCH (sys)-[:compose]->(n)
RETURN sys.Name as System, 
       labels(n) as ComponentTypes, 
       count(n) as ComponentCount
ORDER BY ComponentTypes;