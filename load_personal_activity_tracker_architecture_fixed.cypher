// Load PersonalActivityTracker Architecture into Neo4j - FIXED VERSION
// Architecture: Mac activity tracking app with proper Actor boundary compliance
// Fix: All FCHAINs now have Actor beginning AND end, config-driven classification

// 1. Create System
CREATE (sys:SYS {
    Name: "PersonalActivityTracker",
    ID: "SYS_PersonalActivityTracker",
    Descr: "Mac app for personal activity tracking with calendar integration and wellness monitoring"
})

// 2. Create Use Cases
CREATE (uc1:UC {
    Name: "TrackActivities",
    ID: "UC_TrackActivities", 
    Descr: "Complete automated activity tracking from monitoring through calendar storage"
})

CREATE (uc2:UC {
    Name: "ManageConfiguration",
    ID: "UC_ManageConfiguration",
    Descr: "Manage classification rules through config files and log analysis utilities"
})

CREATE (uc3:UC {
    Name: "ProvideWellnessNudges",
    ID: "UC_ProvideWellnessNudges", 
    Descr: "Monitor session duration, detect long blocks >1h, trigger break notifications and Apple Timer integration"
})

CREATE (uc4:UC {
    Name: "ManageDataRetention",
    ID: "UC_ManageDataRetention",
    Descr: "Separate daily cleanup cycle to delete local logs after 7 days, ensuring only calendar entries remain"
})

// 3. Create Actors
CREATE (a1:ACTOR {
    Name: "User",
    ID: "ACTOR_User",
    Descr: "Person using the app who manages configuration and uses log analysis utilities"
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
    Descr: "macOS system orchestrator (main) that initiates all monitoring, wellness tracking, and retention processes"
})

CREATE (a5:ACTOR {
    Name: "NotificationCenter",
    ID: "ACTOR_NotificationCenter",
    Descr: "macOS notification system for break reminders"
})

CREATE (a6:ACTOR {
    Name: "AppleTimer",
    ID: "ACTOR_AppleTimer",
    Descr: "macOS built-in timer for break countdown functionality"
})

CREATE (a7:ACTOR {
    Name: "ConfigFile",
    ID: "ACTOR_ConfigFile",
    Descr: "Configuration file storage for classification rules and user preferences"
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
    Descr: "Rules for automatic activity classification including user-defined config overrides",
    Struct: '{"applicationPattern": "string", "autoClassification": "string?", "confidence": "number", "requiresUserInput": "boolean", "configOverride": "string?", "userDefined": "boolean", "priority": "number"}'
})

CREATE (s3:SCHEMA {
    Name: "CalendarEvent",
    ID: "SCHEMA_CalendarEvent",
    Descr: "Calendar entry format for permanent activity storage", 
    Struct: '{"title": "string", "startDate": "datetime", "endDate": "datetime", "notes": "string", "calendarId": "string"}'
})

CREATE (s4:SCHEMA {
    Name: "ConfigurationData",
    ID: "SCHEMA_ConfigurationData",
    Descr: "Structure for configuration file with classification rules and preferences",
    Struct: '{"version": "string", "classificationRules": ["ClassificationRule"], "wellnessSettings": {"breakReminderEnabled": "boolean", "sessionThresholdMinutes": "number"}, "retentionSettings": {"localLogRetentionDays": "number"}}'
})

// 5. Create Function Chains with proper Actor boundaries
CREATE (fc1:FCHAIN {
    Name: "MainActivityFlow",
    ID: "FCHAIN_MainActivityFlow",
    Descr: "Complete automated activity tracking from system monitoring through calendar storage"
})

CREATE (fc2:FCHAIN {
    Name: "ConfigurationManagement",
    ID: "FCHAIN_ConfigurationManagement",
    Descr: "Config-driven classification rule management with log analysis utilities"
})

CREATE (fc3:FCHAIN {
    Name: "WellnessMonitoringFlow",
    ID: "FCHAIN_WellnessMonitoringFlow", 
    Descr: "Session tracking, break detection, and notification workflow"
})

CREATE (fc4:FCHAIN {
    Name: "DataRetentionFlow",
    ID: "FCHAIN_DataRetentionFlow",
    Descr: "Daily cleanup cycle for 7-day log retention and privacy compliance"
})

// 6. Create Functions for MainActivityFlow
CREATE (f1:FUNC {
    Name: "InitiateActivityLogging",
    ID: "FUNC_InitiateActivityLogging",
    Descr: "System startup function to begin activity monitoring"
})

CREATE (f2:FUNC {
    Name: "MonitorAppFocus",
    ID: "FUNC_MonitorAppFocus",
    Descr: "Continuously track active application and window focus changes"
})

CREATE (f3:FUNC {
    Name: "LogActivityContinuously", 
    ID: "FUNC_LogActivityContinuously",
    Descr: "Record all activity data immediately without waiting for classification"
})

CREATE (f4:FUNC {
    Name: "Wait15MinStabilization",
    ID: "FUNC_Wait15MinStabilization",
    Descr: "Delay classification until 15 minutes of stable activity detected"
})

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
    Name: "CheckConfigurationRules",
    ID: "FUNC_CheckConfigurationRules",
    Descr: "Read config file for user-defined classification overrides and rules"
})

CREATE (f9:FUNC {
    Name: "FinalizeClassification",
    ID: "FUNC_FinalizeClassification",
    Descr: "Complete activity classification using config rules and smart defaults"
})

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

// 7. Create Functions for ConfigurationManagement
CREATE (f14:FUNC {
    Name: "GenerateConfigFromLogs",
    ID: "FUNC_GenerateConfigFromLogs",
    Descr: "Analyze activity logs to identify patterns and suggest classification rules"
})

CREATE (f15:FUNC {
    Name: "UpdateClassificationRules",
    ID: "FUNC_UpdateClassificationRules",
    Descr: "Modify configuration with new or updated classification rules"
})

CREATE (f16:FUNC {
    Name: "SaveConfiguration",
    ID: "FUNC_SaveConfiguration",
    Descr: "Write updated configuration rules to config file"
})

// 8. Create Functions for WellnessMonitoringFlow
CREATE (f17:FUNC {
    Name: "InitiateWellnessTracking",
    ID: "FUNC_InitiateWellnessTracking",
    Descr: "System startup function to begin wellness session monitoring"
})

CREATE (f18:FUNC {
    Name: "TrackSessionDuration",
    ID: "FUNC_TrackSessionDuration",
    Descr: "Monitor continuous activity duration for wellness notifications"
})

CREATE (f19:FUNC {
    Name: "DetectLongSessions", 
    ID: "FUNC_DetectLongSessions",
    Descr: "Identify activity blocks exceeding 1 hour duration"
})

CREATE (f20:FUNC {
    Name: "TriggerBreakNotification",
    ID: "FUNC_TriggerBreakNotification",
    Descr: "Send break reminder notification for long activity sessions"
})

CREATE (f21:FUNC {
    Name: "LaunchAppleTimer",
    ID: "FUNC_LaunchAppleTimer",
    Descr: "Start macOS timer for break countdown or next work session"
})

// 9. Create Functions for DataRetentionFlow
CREATE (f22:FUNC {
    Name: "InitiateRetentionCheck",
    ID: "FUNC_InitiateRetentionCheck",
    Descr: "System startup function to schedule daily retention cleanup"
})

CREATE (f23:FUNC {
    Name: "ScheduleDailyCleanupCheck",
    ID: "FUNC_ScheduleDailyCleanupCheck",
    Descr: "Run daily background task to identify expired local logs"
})

CREATE (f24:FUNC {
    Name: "IdentifyExpiredLogs",
    ID: "FUNC_IdentifyExpiredLogs", 
    Descr: "Find local activity logs older than 7 days with confirmed calendar storage"
})

CREATE (f25:FUNC {
    Name: "DeleteLocalData",
    ID: "FUNC_DeleteLocalData",
    Descr: "Permanently remove expired local logs while preserving calendar entries"
})

CREATE (f26:FUNC {
    Name: "VerifyOnlyCalendarRemains",
    ID: "FUNC_VerifyOnlyCalendarRemains",
    Descr: "Confirm successful data deletion and calendar-only retention"
})

// 10. Connect System to Use Cases
MATCH (sys:SYS {ID: "SYS_PersonalActivityTracker"})
MATCH (uc1:UC {ID: "UC_TrackActivities"})
MATCH (uc2:UC {ID: "UC_ManageConfiguration"})
MATCH (uc3:UC {ID: "UC_ProvideWellnessNudges"})
MATCH (uc4:UC {ID: "UC_ManageDataRetention"})
CREATE (sys)-[:compose]->(uc1)
CREATE (sys)-[:compose]->(uc2)
CREATE (sys)-[:compose]->(uc3)
CREATE (sys)-[:compose]->(uc4)

// 11. CRITICAL: Connect System to Actors
MATCH (sys:SYS {ID: "SYS_PersonalActivityTracker"})
MATCH (a1:ACTOR {ID: "ACTOR_User"})
MATCH (a2:ACTOR {ID: "ACTOR_CalendarApp"})
MATCH (a3:ACTOR {ID: "ACTOR_GitSystem"})
MATCH (a4:ACTOR {ID: "ACTOR_SystemMonitor"})
MATCH (a5:ACTOR {ID: "ACTOR_NotificationCenter"})
MATCH (a6:ACTOR {ID: "ACTOR_AppleTimer"})
MATCH (a7:ACTOR {ID: "ACTOR_ConfigFile"})
CREATE (sys)-[:compose]->(a1)
CREATE (sys)-[:compose]->(a2)
CREATE (sys)-[:compose]->(a3)
CREATE (sys)-[:compose]->(a4)
CREATE (sys)-[:compose]->(a5)
CREATE (sys)-[:compose]->(a6)
CREATE (sys)-[:compose]->(a7)

// 12. CRITICAL: Connect System to Schemas
MATCH (sys:SYS {ID: "SYS_PersonalActivityTracker"})
MATCH (s1:SCHEMA {ID: "SCHEMA_ActivityBlock"})
MATCH (s2:SCHEMA {ID: "SCHEMA_ClassificationRule"})
MATCH (s3:SCHEMA {ID: "SCHEMA_CalendarEvent"})
MATCH (s4:SCHEMA {ID: "SCHEMA_ConfigurationData"})
CREATE (sys)-[:compose]->(s1)
CREATE (sys)-[:compose]->(s2)
CREATE (sys)-[:compose]->(s3)
CREATE (sys)-[:compose]->(s4)

// 13. Connect Use Cases to Function Chains
MATCH (uc1:UC {ID: "UC_TrackActivities"})
MATCH (uc2:UC {ID: "UC_ManageConfiguration"})
MATCH (uc3:UC {ID: "UC_ProvideWellnessNudges"})
MATCH (uc4:UC {ID: "UC_ManageDataRetention"})
MATCH (fc1:FCHAIN {ID: "FCHAIN_MainActivityFlow"})
MATCH (fc2:FCHAIN {ID: "FCHAIN_ConfigurationManagement"})
MATCH (fc3:FCHAIN {ID: "FCHAIN_WellnessMonitoringFlow"})
MATCH (fc4:FCHAIN {ID: "FCHAIN_DataRetentionFlow"})
CREATE (uc1)-[:compose]->(fc1)
CREATE (uc2)-[:compose]->(fc2)
CREATE (uc3)-[:compose]->(fc3)
CREATE (uc4)-[:compose]->(fc4)

// 14. Connect Function Chains to Functions
MATCH (fc1:FCHAIN {ID: "FCHAIN_MainActivityFlow"})
MATCH (f1:FUNC {ID: "FUNC_InitiateActivityLogging"})
MATCH (f2:FUNC {ID: "FUNC_MonitorAppFocus"})
MATCH (f3:FUNC {ID: "FUNC_LogActivityContinuously"})
MATCH (f4:FUNC {ID: "FUNC_Wait15MinStabilization"})
MATCH (f5:FUNC {ID: "FUNC_AnalyzeApplication"})
MATCH (f6:FUNC {ID: "FUNC_ApplySmartDefaults"})
MATCH (f7:FUNC {ID: "FUNC_ScanGitForProjectName"})
MATCH (f8:FUNC {ID: "FUNC_CheckConfigurationRules"})
MATCH (f9:FUNC {ID: "FUNC_FinalizeClassification"})
MATCH (f10:FUNC {ID: "FUNC_FormatActivityWithProjectName"})
MATCH (f11:FUNC {ID: "FUNC_CreateCalendarEvent"})
MATCH (f12:FUNC {ID: "FUNC_VerifyCalendarStorage"})
MATCH (f13:FUNC {ID: "FUNC_MarkForRetentionTracking"})
CREATE (fc1)-[:compose]->(f1)
CREATE (fc1)-[:compose]->(f2)
CREATE (fc1)-[:compose]->(f3)
CREATE (fc1)-[:compose]->(f4)
CREATE (fc1)-[:compose]->(f5)
CREATE (fc1)-[:compose]->(f6)
CREATE (fc1)-[:compose]->(f7)
CREATE (fc1)-[:compose]->(f8)
CREATE (fc1)-[:compose]->(f9)
CREATE (fc1)-[:compose]->(f10)
CREATE (fc1)-[:compose]->(f11)
CREATE (fc1)-[:compose]->(f12)
CREATE (fc1)-[:compose]->(f13)

MATCH (fc2:FCHAIN {ID: "FCHAIN_ConfigurationManagement"})
MATCH (f14:FUNC {ID: "FUNC_GenerateConfigFromLogs"})
MATCH (f15:FUNC {ID: "FUNC_UpdateClassificationRules"})
MATCH (f16:FUNC {ID: "FUNC_SaveConfiguration"})
CREATE (fc2)-[:compose]->(f14)
CREATE (fc2)-[:compose]->(f15)
CREATE (fc2)-[:compose]->(f16)

MATCH (fc3:FCHAIN {ID: "FCHAIN_WellnessMonitoringFlow"})
MATCH (f17:FUNC {ID: "FUNC_InitiateWellnessTracking"})
MATCH (f18:FUNC {ID: "FUNC_TrackSessionDuration"})
MATCH (f19:FUNC {ID: "FUNC_DetectLongSessions"})
MATCH (f20:FUNC {ID: "FUNC_TriggerBreakNotification"})
MATCH (f21:FUNC {ID: "FUNC_LaunchAppleTimer"})
CREATE (fc3)-[:compose]->(f17)
CREATE (fc3)-[:compose]->(f18)
CREATE (fc3)-[:compose]->(f19)
CREATE (fc3)-[:compose]->(f20)
CREATE (fc3)-[:compose]->(f21)

MATCH (fc4:FCHAIN {ID: "FCHAIN_DataRetentionFlow"})
MATCH (f22:FUNC {ID: "FUNC_InitiateRetentionCheck"})
MATCH (f23:FUNC {ID: "FUNC_ScheduleDailyCleanupCheck"})
MATCH (f24:FUNC {ID: "FUNC_IdentifyExpiredLogs"})
MATCH (f25:FUNC {ID: "FUNC_DeleteLocalData"})
MATCH (f26:FUNC {ID: "FUNC_VerifyOnlyCalendarRemains"})
CREATE (fc4)-[:compose]->(f22)
CREATE (fc4)-[:compose]->(f23)
CREATE (fc4)-[:compose]->(f24)
CREATE (fc4)-[:compose]->(f25)
CREATE (fc4)-[:compose]->(f26)

// 15. Create Flow Relationships within MainActivityFlow (SystemMonitor → CalendarApp)
MATCH (f1:FUNC {ID: "FUNC_InitiateActivityLogging"})
MATCH (f2:FUNC {ID: "FUNC_MonitorAppFocus"})
MATCH (f3:FUNC {ID: "FUNC_LogActivityContinuously"})
MATCH (f4:FUNC {ID: "FUNC_Wait15MinStabilization"})
MATCH (f5:FUNC {ID: "FUNC_AnalyzeApplication"})
MATCH (f6:FUNC {ID: "FUNC_ApplySmartDefaults"})
MATCH (f7:FUNC {ID: "FUNC_ScanGitForProjectName"})
MATCH (f8:FUNC {ID: "FUNC_CheckConfigurationRules"})
MATCH (f9:FUNC {ID: "FUNC_FinalizeClassification"})
MATCH (f10:FUNC {ID: "FUNC_FormatActivityWithProjectName"})
MATCH (f11:FUNC {ID: "FUNC_CreateCalendarEvent"})
MATCH (f12:FUNC {ID: "FUNC_VerifyCalendarStorage"})
MATCH (f13:FUNC {ID: "FUNC_MarkForRetentionTracking"})

CREATE (f1)-[:flow {
    FlowDescr: "System initiation triggers app focus monitoring",
    FlowDef: "system_startup → start_monitoring()"
}]->(f2)

CREATE (f2)-[:flow {
    FlowDescr: "App focus changes trigger continuous activity logging",
    FlowDef: "focus_change_events → log_activity(timestamp, app, window)"
}]->(f3)

CREATE (f3)-[:flow {
    FlowDescr: "Logged activities feed into stabilization delay logic",
    FlowDef: "activity_log → check_stability_duration(15_minutes)"
}]->(f4)

CREATE (f4)-[:flow {
    FlowDescr: "Stabilized activities trigger application analysis",
    FlowDef: "stable_activity_block → analyze_application_type()"
}]->(f5)

CREATE (f5)-[:flow {
    FlowDescr: "Application analysis feeds into smart default classification",
    FlowDef: "app_analysis → apply_classification_rules(confidence_level)"
}]->(f6)

CREATE (f6)-[:flow {
    FlowDescr: "Default classification triggers git project scanning for coding activities",
    FlowDef: "classification_result → scan_git_repos() if coding_activity"
}]->(f7)

CREATE (f7)-[:flow {
    FlowDescr: "Git analysis results combined with config rule checking",
    FlowDef: "project_name + git_analysis → check_config_overrides()"
}]->(f8)

CREATE (f8)-[:flow {
    FlowDescr: "Config rules and analysis results finalize classification",
    FlowDef: "config_rules + analysis → final_classification"
}]->(f9)

CREATE (f9)-[:flow {
    FlowDescr: "Final classification triggers activity formatting",
    FlowDef: "final_classification → format_activity_title(project_name)"
}]->(f10)

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

// 16. Create Flow Relationships within ConfigurationManagement (User → ConfigFile)
MATCH (f14:FUNC {ID: "FUNC_GenerateConfigFromLogs"})
MATCH (f15:FUNC {ID: "FUNC_UpdateClassificationRules"})
MATCH (f16:FUNC {ID: "FUNC_SaveConfiguration"})

CREATE (f14)-[:flow {
    FlowDescr: "Log analysis generates configuration rule suggestions",
    FlowDef: "analyze_logs() → suggest_classification_rules()"
}]->(f15)

CREATE (f15)-[:flow {
    FlowDescr: "Updated rules are formatted for configuration storage",
    FlowDef: "classification_rules → format_config_data()"
}]->(f16)

// 17. Create Flow Relationships within WellnessMonitoringFlow (SystemMonitor → NotificationCenter/AppleTimer)
MATCH (f17:FUNC {ID: "FUNC_InitiateWellnessTracking"})
MATCH (f18:FUNC {ID: "FUNC_TrackSessionDuration"})
MATCH (f19:FUNC {ID: "FUNC_DetectLongSessions"})
MATCH (f20:FUNC {ID: "FUNC_TriggerBreakNotification"})
MATCH (f21:FUNC {ID: "FUNC_LaunchAppleTimer"})

CREATE (f17)-[:flow {
    FlowDescr: "System initiation triggers wellness session tracking",
    FlowDef: "system_startup → start_wellness_monitoring()"
}]->(f18)

CREATE (f18)-[:flow {
    FlowDescr: "Session duration tracking identifies long activity blocks",
    FlowDef: "session_duration → check_long_session(1_hour_threshold)"
}]->(f19)

CREATE (f19)-[:flow {
    FlowDescr: "Long session detection triggers break reminder notifications",
    FlowDef: "long_session_detected → send_break_notification()"
}]->(f20)

CREATE (f20)-[:flow {
    FlowDescr: "Break notification offers Apple Timer for break countdown",
    FlowDef: "break_notification → launch_timer(break_duration) if user_accepts"
}]->(f21)

// 18. Create Flow Relationships within DataRetentionFlow (SystemMonitor → SystemMonitor)
MATCH (f22:FUNC {ID: "FUNC_InitiateRetentionCheck"})
MATCH (f23:FUNC {ID: "FUNC_ScheduleDailyCleanupCheck"})
MATCH (f24:FUNC {ID: "FUNC_IdentifyExpiredLogs"})
MATCH (f25:FUNC {ID: "FUNC_DeleteLocalData"})
MATCH (f26:FUNC {ID: "FUNC_VerifyOnlyCalendarRemains"})

CREATE (f22)-[:flow {
    FlowDescr: "System initiation schedules daily retention checks",
    FlowDef: "system_startup → schedule_daily_cleanup()"
}]->(f23)

CREATE (f23)-[:flow {
    FlowDescr: "Daily cleanup schedule triggers expired log identification",
    FlowDef: "daily_cleanup_trigger → identify_logs_older_than(7_days)"
}]->(f24)

CREATE (f24)-[:flow {
    FlowDescr: "Expired log identification leads to secure data deletion",
    FlowDef: "expired_logs_list → secure_delete(local_files)"
}]->(f25)

CREATE (f25)-[:flow {
    FlowDescr: "Local data deletion is verified to ensure only calendar data remains",
    FlowDef: "deletion_complete → verify_calendar_only_retention()"
}]->(f26)

// 19. Create Actor Flow Relationships - MainActivityFlow: SystemMonitor → CalendarApp
MATCH (a4:ACTOR {ID: "ACTOR_SystemMonitor"})
MATCH (f1:FUNC {ID: "FUNC_InitiateActivityLogging"})
MATCH (f13:FUNC {ID: "FUNC_MarkForRetentionTracking"})
MATCH (a2:ACTOR {ID: "ACTOR_CalendarApp"})

CREATE (a4)-[:flow {
    FlowDescr: "System monitor initiates activity logging workflow",
    FlowDef: "main() → initiate_activity_logging()"
}]->(f1)

CREATE (f13)-[:flow {
    FlowDescr: "Retention tracking completion flows to calendar app",
    FlowDef: "retention_marked → calendar_storage_confirmed"
}]->(a2)

// 20. Create Actor Flow Relationships - ConfigurationManagement: User → ConfigFile
MATCH (a1:ACTOR {ID: "ACTOR_User"})
MATCH (f14:FUNC {ID: "FUNC_GenerateConfigFromLogs"})
MATCH (f16:FUNC {ID: "FUNC_SaveConfiguration"})
MATCH (a7:ACTOR {ID: "ACTOR_ConfigFile"})

CREATE (a1)-[:flow {
    FlowDescr: "User initiates configuration management utility",
    FlowDef: "user_command → generate_config_from_logs()"
}]->(f14)

CREATE (f16)-[:flow {
    FlowDescr: "Configuration saved to config file",
    FlowDef: "save_configuration() → config_file.write()"
}]->(a7)

// 21. Create Actor Flow Relationships - WellnessMonitoringFlow: SystemMonitor → NotificationCenter/AppleTimer
MATCH (a4:ACTOR {ID: "ACTOR_SystemMonitor"})
MATCH (f17:FUNC {ID: "FUNC_InitiateWellnessTracking"})
MATCH (f20:FUNC {ID: "FUNC_TriggerBreakNotification"})
MATCH (f21:FUNC {ID: "FUNC_LaunchAppleTimer"})
MATCH (a5:ACTOR {ID: "ACTOR_NotificationCenter"})
MATCH (a6:ACTOR {ID: "ACTOR_AppleTimer"})

CREATE (a4)-[:flow {
    FlowDescr: "System monitor initiates wellness tracking",
    FlowDef: "main() → initiate_wellness_tracking()"
}]->(f17)

CREATE (f20)-[:flow {
    FlowDescr: "Break reminder sent via notification center",
    FlowDef: "break_reminder → notification_center.show()"
}]->(a5)

CREATE (f21)-[:flow {
    FlowDescr: "Timer launch request sent to Apple Timer",
    FlowDef: "timer_request → apple_timer.start(duration)"
}]->(a6)

// 22. Create Actor Flow Relationships - DataRetentionFlow: SystemMonitor → SystemMonitor
MATCH (a4:ACTOR {ID: "ACTOR_SystemMonitor"})
MATCH (f22:FUNC {ID: "FUNC_InitiateRetentionCheck"})
MATCH (f26:FUNC {ID: "FUNC_VerifyOnlyCalendarRemains"})

CREATE (a4)-[:flow {
    FlowDescr: "System monitor initiates retention check workflow",
    FlowDef: "main() → initiate_retention_check()"
}]->(f22)

CREATE (f26)-[:flow {
    FlowDescr: "Retention verification completion confirms to system monitor",
    FlowDef: "verification_complete → system_monitor.confirm()"
}]->(a4)

// 23. Create External Actor Integration Flows
MATCH (a3:ACTOR {ID: "ACTOR_GitSystem"})
MATCH (f7:FUNC {ID: "FUNC_ScanGitForProjectName"})
MATCH (a7:ACTOR {ID: "ACTOR_ConfigFile"})
MATCH (f8:FUNC {ID: "FUNC_CheckConfigurationRules"})
MATCH (f11:FUNC {ID: "FUNC_CreateCalendarEvent"})
MATCH (a2:ACTOR {ID: "ACTOR_CalendarApp"})

CREATE (a3)-[:flow {
    FlowDescr: "Git system provides commit data for project analysis",
    FlowDef: "git_log() → extract_project_name(commits)"
}]->(f7)

CREATE (a7)-[:flow {
    FlowDescr: "Config file provides classification rules",
    FlowDef: "config_file.read() → get_classification_rules()"
}]->(f8)

CREATE (f11)-[:flow {
    FlowDescr: "Calendar event sent to Calendar.app for storage",
    FlowDef: "calendar_event → calendar_app.create_event()"
}]->(a2)

// 24. Verify the fixed architecture is loaded
MATCH (sys:SYS {Name: "PersonalActivityTracker"})
OPTIONAL MATCH (sys)-[:compose]->(n)
RETURN sys.Name as System, 
       labels(n) as ComponentTypes, 
       count(n) as ComponentCount
ORDER BY ComponentTypes;

// 25. Verify FCHAIN Actor boundaries compliance
MATCH (fc:FCHAIN)
OPTIONAL MATCH (fc)-[:compose]->(f:FUNC)
OPTIONAL MATCH (start_actor:ACTOR)-[:flow]->(first_func:FUNC)
WHERE (fc)-[:compose]->(first_func)
OPTIONAL MATCH (last_func:FUNC)-[:flow]->(end_actor:ACTOR)
WHERE (fc)-[:compose]->(last_func)
RETURN fc.Name as FChain,
       start_actor.Name as StartActor,
       end_actor.Name as EndActor,
       count(f) as FunctionCount
ORDER BY fc.Name;