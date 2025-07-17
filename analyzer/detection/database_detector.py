#!/usr/bin/env python3
"""
Database Detector for Code Architecture Analyzer

Specialized detector for identifying database usage patterns in Python code.
Detects SQLite, PostgreSQL, MongoDB, Redis, SQLAlchemy, Django ORM and other database connections.
"""

import re
import logging
from typing import Dict, List, Optional, Any, Set, Tuple
from dataclasses import dataclass

from .pattern_matcher import BaseDetector, PatternRule, DetectionMatch, ActorType, ConfidenceLevel
from ..core.ast_parser import ASTParseResult, FunctionInfo, ImportInfo

logger = logging.getLogger(__name__)


@dataclass
class DatabasePattern:
    """Database-specific pattern information."""
    library_name: str
    database_type: str
    connection_methods: List[str]
    query_methods: List[str]
    confidence: float
    orm_support: bool = False
    async_support: bool = False
    nosql: bool = False


class DatabaseDetector(BaseDetector):
    """
    Specialized detector for database connection patterns.
    
    Identifies usage of database libraries and extracts information about
    connections, queries, ORM usage, and database operations.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize database detector with configuration."""
        super().__init__(config)
        
        # Database-specific configuration
        db_config = self.config.get('detection', {}).get('database', {})
        self.detect_connection_strings = db_config.get('detect_connection_strings', True)
        self.detect_sql_queries = db_config.get('detect_sql_queries', True)
        self.min_connection_confidence = db_config.get('min_connection_confidence', 0.8)
        
        # SQL keywords pattern for detecting SQL queries in strings
        self.sql_keywords = {
            'SELECT', 'INSERT', 'UPDATE', 'DELETE', 'CREATE', 'DROP', 'ALTER',
            'INDEX', 'TABLE', 'DATABASE', 'SCHEMA', 'FROM', 'WHERE', 'JOIN',
            'GROUP BY', 'ORDER BY', 'HAVING', 'UNION'
        }
        
        # Connection string patterns
        self.connection_patterns = [
            re.compile(r'postgresql://[^\s"\']+', re.IGNORECASE),
            re.compile(r'mysql://[^\s"\']+', re.IGNORECASE),
            re.compile(r'sqlite:///[^\s"\']+', re.IGNORECASE),
            re.compile(r'mongodb://[^\s"\']+', re.IGNORECASE),
            re.compile(r'redis://[^\s"\']+', re.IGNORECASE),
        ]
        
        # SQL query pattern (basic detection)
        self.sql_pattern = re.compile(
            r'\b(?:SELECT|INSERT|UPDATE|DELETE|CREATE|DROP|ALTER)\b.*?(?:FROM|INTO|TABLE|INDEX)\b',
            re.IGNORECASE | re.DOTALL
        )
        
        logger.debug("Database detector initialized")
    
    def _load_patterns(self) -> List[PatternRule]:
        """Load database detection patterns."""
        patterns = []
        
        # Get confidence scores from config
        base_confidence = self.config.get('deterministic', {}).get('pattern_confidence', {}).get('database', 0.90)
        
        # SQLite patterns (built-in)
        patterns.append(PatternRule(
            name="sqlite3_builtin",
            actor_type=ActorType.DATABASE,
            confidence=base_confidence,
            imports=["sqlite3"],
            function_calls=[
                "sqlite3.connect", "sqlite3.Connection", "connection.execute",
                "connection.executemany", "connection.executescript", "cursor.execute",
                "cursor.executemany", "cursor.fetchone", "cursor.fetchall", "cursor.fetchmany"
            ],
            keywords=["sqlite3", "cursor", "execute", "fetch"]
        ))
        
        # PostgreSQL patterns (psycopg2)
        patterns.append(PatternRule(
            name="psycopg2_postgresql",
            actor_type=ActorType.DATABASE,
            confidence=base_confidence,
            imports=["psycopg2", "psycopg2.*", "psycopg2.pool"],
            function_calls=[
                "psycopg2.connect", "psycopg2.Connection", "cursor.execute",
                "cursor.executemany", "cursor.fetchone", "cursor.fetchall",
                "connection.commit", "connection.rollback", "SimpleConnectionPool"
            ],
            keywords=["psycopg2", "postgresql", "postgres", "cursor"]
        ))
        
        # MySQL patterns
        patterns.append(PatternRule(
            name="mysql_connector",
            actor_type=ActorType.DATABASE,
            confidence=base_confidence,
            imports=["mysql.connector", "mysql.*", "MySQLdb", "pymysql"],
            function_calls=[
                "mysql.connector.connect", "MySQLdb.connect", "pymysql.connect",
                "cursor.execute", "cursor.executemany", "cursor.fetchone", "cursor.fetchall"
            ],
            keywords=["mysql", "MySQLdb", "pymysql"]
        ))
        
        # Neo4j patterns
        patterns.append(PatternRule(
            name="neo4j_graph_database",
            actor_type=ActorType.DATABASE,
            confidence=base_confidence,
            imports=["neo4j", "neo4j.*", "neo4j.exceptions"],
            function_calls=[
                "GraphDatabase.driver", "neo4j.GraphDatabase.driver", "driver.session",
                "session.run", "session.execute_read", "session.execute_write",
                "tx.run", "result.single", "result.data", "result.consume",
                "session.begin_transaction", "session.close", "driver.close"
            ],
            keywords=["neo4j", "GraphDatabase", "cypher", "session", "tx", "graph"]
        ))
        
        # MongoDB patterns (pymongo)
        patterns.append(PatternRule(
            name="pymongo_mongodb",
            actor_type=ActorType.DATABASE,
            confidence=base_confidence,
            imports=["pymongo", "pymongo.*", "bson"],
            function_calls=[
                "pymongo.MongoClient", "MongoClient", "db.collection", "collection.find",
                "collection.find_one", "collection.insert_one", "collection.insert_many",
                "collection.update_one", "collection.update_many", "collection.delete_one",
                "collection.delete_many", "collection.aggregate", "collection.count_documents"
            ],
            keywords=["mongodb", "mongo", "pymongo", "collection"]
        ))
        
        # Redis patterns
        patterns.append(PatternRule(
            name="redis_cache",
            actor_type=ActorType.DATABASE,
            confidence=base_confidence,
            imports=["redis", "redis.*"],
            function_calls=[
                "redis.Redis", "redis.StrictRedis", "redis.ConnectionPool",
                "redis_client.get", "redis_client.set", "redis_client.delete",
                "redis_client.exists", "redis_client.expire", "redis_client.lpush",
                "redis_client.rpush", "redis_client.lpop", "redis_client.rpop"
            ],
            keywords=["redis", "cache", "key-value"]
        ))
        
        # SQLAlchemy ORM patterns
        patterns.append(PatternRule(
            name="sqlalchemy_orm",
            actor_type=ActorType.DATABASE,
            confidence=base_confidence,
            imports=["sqlalchemy", "sqlalchemy.*"],
            function_calls=[
                "sqlalchemy.create_engine", "create_engine", "Session", "sessionmaker",
                "session.query", "session.add", "session.commit", "session.rollback",
                "query.filter", "query.all", "query.first", "query.count",
                "Column", "Integer", "String", "DateTime", "ForeignKey"
            ],
            keywords=["sqlalchemy", "orm", "session", "query"]
        ))
        
        # Django ORM patterns
        patterns.append(PatternRule(
            name="django_orm",
            actor_type=ActorType.DATABASE,
            confidence=base_confidence - 0.05,
            imports=["django.db", "django.db.*", "django.contrib.*"],
            function_calls=[
                "models.Model", "objects.all", "objects.filter", "objects.get",
                "objects.create", "objects.update", "objects.delete",
                "save", "delete", "Q", "F", "Prefetch"
            ],
            keywords=["django", "models", "objects", "orm"],
            class_names=["Model", "Manager"]
        ))
        
        # Peewee ORM patterns
        patterns.append(PatternRule(
            name="peewee_orm",
            actor_type=ActorType.DATABASE,
            confidence=base_confidence - 0.1,
            imports=["peewee", "peewee.*"],
            function_calls=[
                "peewee.Model", "Model.select", "Model.get", "Model.create",
                "Model.save", "Model.delete", "SqliteDatabase", "PostgresqlDatabase",
                "MySQLDatabase"
            ],
            keywords=["peewee", "orm", "model"]
        ))
        
        # Tortoise ORM (async) patterns
        patterns.append(PatternRule(
            name="tortoise_orm",
            actor_type=ActorType.DATABASE,
            confidence=base_confidence - 0.1,
            imports=["tortoise", "tortoise.*"],
            function_calls=[
                "tortoise.models.Model", "Tortoise.init", "Tortoise.generate_schemas",
                "model.save", "model.delete", "Model.filter", "Model.all",
                "Model.get", "Model.create"
            ],
            keywords=["tortoise", "async", "orm"]
        ))
        
        # Cassandra patterns
        patterns.append(PatternRule(
            name="cassandra_driver",
            actor_type=ActorType.DATABASE,
            confidence=base_confidence - 0.05,
            imports=["cassandra", "cassandra.*"],
            function_calls=[
                "cassandra.cluster.Cluster", "Cluster", "session.execute",
                "session.prepare", "SimpleStatement", "PreparedStatement"
            ],
            keywords=["cassandra", "cluster", "keyspace"]
        ))
        
        # Elasticsearch patterns
        patterns.append(PatternRule(
            name="elasticsearch_client",
            actor_type=ActorType.DATABASE,
            confidence=base_confidence - 0.05,
            imports=["elasticsearch", "elasticsearch.*"],
            function_calls=[
                "elasticsearch.Elasticsearch", "Elasticsearch", "es.index",
                "es.search", "es.get", "es.delete", "es.update", "es.bulk"
            ],
            keywords=["elasticsearch", "search", "index", "document"]
        ))
        
        logger.info(f"Loaded {len(patterns)} database detection patterns")
        return patterns
    
    def _detect_function_calls(self, functions: List[FunctionInfo]) -> List[DetectionMatch]:
        """Detect database usage from function calls."""
        return self._detect_database_calls(functions)
    
    def _detect_content_patterns(self, ast_result: ASTParseResult) -> List[DetectionMatch]:
        """Detect database connection strings and SQL queries."""
        if self.detect_connection_strings or self.detect_sql_queries:
            return self._detect_database_strings(ast_result)
        return []
    
    
    def _detect_database_calls(self, functions: List[FunctionInfo]) -> List[DetectionMatch]:
        """Detect database operations within functions."""
        matches = []
        
        for func_info in functions:
            for call in func_info.calls:
                # Check against database patterns
                for pattern in self.patterns:
                    if self._match_database_call(call, pattern):
                        match = DetectionMatch(
                            actor_type=ActorType.DATABASE,
                            confidence=pattern.confidence,
                            pattern_name=pattern.name,
                            evidence={
                                'function_call': call,
                                'operation_type': self._extract_operation_type(call),
                                'containing_function': func_info.name,
                                'is_async': func_info.is_async
                            },
                            context={
                                'detection_method': 'database_call_analysis',
                                'database_type': self._get_database_type(pattern.name),
                                'library_type': self._get_library_type(pattern.name),
                                'call_context': 'function'
                            },
                            line_numbers=[func_info.line_number],
                            function_name=func_info.name
                        )
                        matches.append(match)
        
        return matches
    
    def _detect_database_strings(self, ast_result: ASTParseResult) -> List[DetectionMatch]:
        """Detect database connection strings and SQL queries in code."""
        matches = []
        
        try:
            # Read the file content to search for database strings
            with open(ast_result.file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Detect connection strings
            if self.detect_connection_strings:
                for pattern in self.connection_patterns:
                    for conn_match in pattern.finditer(content):
                        connection_string = conn_match.group()
                        line_number = content[:conn_match.start()].count('\n') + 1
                        
                        match = DetectionMatch(
                            actor_type=ActorType.DATABASE,
                            confidence=self.min_connection_confidence,
                            pattern_name="connection_string",
                            evidence={
                                'connection_string': connection_string,
                                'string_type': 'connection'
                            },
                            context={
                                'detection_method': 'connection_string_analysis',
                                'database_type': self._infer_db_type_from_connection(connection_string)
                            },
                            line_numbers=[line_number]
                        )
                        matches.append(match)
            
            # Detect SQL queries
            if self.detect_sql_queries:
                for sql_match in self.sql_pattern.finditer(content):
                    sql_query = sql_match.group().strip()
                    line_number = content[:sql_match.start()].count('\n') + 1
                    
                    # Skip very short matches that might be false positives
                    if len(sql_query) < 10:
                        continue
                    
                    match = DetectionMatch(
                        actor_type=ActorType.DATABASE,
                        confidence=0.7,  # Lower confidence for SQL string detection
                        pattern_name="sql_query",
                        evidence={
                            'sql_query': sql_query[:100],  # Truncate for brevity
                            'query_type': self._extract_sql_operation(sql_query),
                            'string_type': 'sql'
                        },
                        context={
                            'detection_method': 'sql_string_analysis',
                            'query_complexity': 'simple' if len(sql_query) < 50 else 'complex'
                        },
                        line_numbers=[line_number]
                    )
                    matches.append(match)
        
        except Exception as e:
            logger.warning(f"Failed to detect database strings in {ast_result.file_path}: {e}")
        
        return matches
    
    
    def _match_database_call(self, call: str, pattern: PatternRule) -> bool:
        """Check if function call matches database pattern."""
        if not pattern.function_calls:
            return False
        
        for call_pattern in pattern.function_calls:
            if call == call_pattern:
                return True
            if call.endswith(call_pattern.split('.')[-1]):  # Match method name
                return True
            if call_pattern in call:  # Substring match for complex calls
                return True
        
        return False
    
    def _extract_operation_type(self, call: str) -> Optional[str]:
        """Extract database operation type from function call."""
        call_lower = call.lower()
        
        # Query operations
        if any(op in call_lower for op in ['select', 'find', 'get', 'fetch', 'query']):
            return 'READ'
        
        # Insert operations
        if any(op in call_lower for op in ['insert', 'create', 'add', 'save']):
            return 'CREATE'
        
        # Update operations
        if any(op in call_lower for op in ['update', 'modify', 'set']):
            return 'UPDATE'
        
        # Delete operations
        if any(op in call_lower for op in ['delete', 'remove', 'drop']):
            return 'DELETE'
        
        # Connection operations
        if any(op in call_lower for op in ['connect', 'connection']):
            return 'CONNECT'
        
        # Transaction operations
        if any(op in call_lower for op in ['commit', 'rollback', 'transaction']):
            return 'TRANSACTION'
        
        return 'UNKNOWN'
    
    def _extract_sql_operation(self, sql: str) -> str:
        """Extract SQL operation type from query string."""
        sql_upper = sql.upper().strip()
        
        if sql_upper.startswith('SELECT'):
            return 'SELECT'
        elif sql_upper.startswith('INSERT'):
            return 'INSERT'
        elif sql_upper.startswith('UPDATE'):
            return 'UPDATE'
        elif sql_upper.startswith('DELETE'):
            return 'DELETE'
        elif sql_upper.startswith('CREATE'):
            return 'CREATE'
        elif sql_upper.startswith('DROP'):
            return 'DROP'
        elif sql_upper.startswith('ALTER'):
            return 'ALTER'
        else:
            return 'UNKNOWN'
    
    def _get_database_type(self, pattern_name: str) -> str:
        """Get database type from pattern name."""
        db_type_map = {
            'sqlite3_builtin': 'sqlite',
            'psycopg2_postgresql': 'postgresql',
            'mysql_connector': 'mysql',
            'pymongo_mongodb': 'mongodb',
            'redis_cache': 'redis',
            'sqlalchemy_orm': 'relational',
            'django_orm': 'relational',
            'peewee_orm': 'relational',
            'tortoise_orm': 'relational',
            'cassandra_driver': 'cassandra',
            'elasticsearch_client': 'elasticsearch'
        }
        return db_type_map.get(pattern_name, 'unknown')
    
    def _get_library_type(self, pattern_name: str) -> str:
        """Get library type from pattern name."""
        library_map = {
            'sqlite3_builtin': 'builtin',
            'psycopg2_postgresql': 'driver',
            'mysql_connector': 'driver',
            'pymongo_mongodb': 'driver',
            'redis_cache': 'client',
            'sqlalchemy_orm': 'orm',
            'django_orm': 'orm',
            'peewee_orm': 'orm',
            'tortoise_orm': 'orm',
            'cassandra_driver': 'driver',
            'elasticsearch_client': 'client'
        }
        return library_map.get(pattern_name, 'unknown')
    
    def _infer_db_type_from_connection(self, connection_string: str) -> str:
        """Infer database type from connection string."""
        conn_lower = connection_string.lower()
        
        if conn_lower.startswith('postgresql://'):
            return 'postgresql'
        elif conn_lower.startswith('mysql://'):
            return 'mysql'
        elif conn_lower.startswith('sqlite:///'):
            return 'sqlite'
        elif conn_lower.startswith('mongodb://'):
            return 'mongodb'
        elif conn_lower.startswith('redis://'):
            return 'redis'
        else:
            return 'unknown'
    
    def _enhance_matches(self, matches: List[DetectionMatch], ast_result: ASTParseResult) -> None:
        """Enhance matches with additional context and information."""
        for match in matches:
            # Add module context
            match.module_name = ast_result.module_name
            
            # Enhance evidence based on detection method
            if match.context.get('detection_method') == 'database_import_analysis':
                self._enhance_import_match(match, ast_result)
            elif match.context.get('detection_method') == 'database_call_analysis':
                self._enhance_call_match(match, ast_result)
            elif match.context.get('detection_method') in ['connection_string_analysis', 'sql_string_analysis']:
                self._enhance_string_match(match)
    
    def _enhance_import_match(self, match: DetectionMatch, ast_result: ASTParseResult) -> None:
        """Enhance import-based matches with usage context."""
        imported_module = match.evidence.get('import_module', '')
        
        # Count usage frequency in functions
        usage_count = 0
        for func_info in ast_result.functions:
            for call in func_info.calls:
                if imported_module in call or match.evidence.get('import_name', '') in call:
                    usage_count += 1
        
        match.evidence['usage_frequency'] = usage_count
        match.context['import_usage'] = 'active' if usage_count > 0 else 'imported_only'
        
        # Determine if it's ORM usage
        if any(orm_indicator in imported_module.lower() for orm_indicator in ['sqlalchemy', 'django', 'peewee', 'tortoise']):
            match.context['orm_usage'] = True
    
    def _enhance_call_match(self, match: DetectionMatch, ast_result: ASTParseResult) -> None:
        """Enhance call-based matches with function context."""
        function_name = match.function_name
        
        if function_name:
            # Find the function info
            func_info = next((f for f in ast_result.functions if f.name == function_name), None)
            if func_info:
                match.evidence['function_complexity'] = func_info.complexity
                match.evidence['function_args'] = func_info.args
                match.evidence['is_async_function'] = func_info.is_async
                
                # Check for transaction patterns
                transaction_calls = [call for call in func_info.calls 
                                   if any(trans in call.lower() for trans in ['commit', 'rollback', 'transaction'])]
                match.context['has_transactions'] = len(transaction_calls) > 0
                
                # Check for error handling patterns
                error_handling = any('except' in call or 'try' in call for call in func_info.calls)
                match.context['has_error_handling'] = error_handling
    
    def _enhance_string_match(self, match: DetectionMatch) -> None:
        """Enhance string-based matches with string analysis."""
        if match.evidence.get('string_type') == 'connection':
            connection_string = match.evidence.get('connection_string', '')
            # Extract additional connection info if needed
            match.context['connection_security'] = 'ssl' if 'ssl' in connection_string.lower() else 'plain'
        
        elif match.evidence.get('string_type') == 'sql':
            sql_query = match.evidence.get('sql_query', '')
            # Analyze SQL complexity
            if any(keyword in sql_query.upper() for keyword in ['JOIN', 'UNION', 'SUBQUERY']):
                match.context['query_complexity'] = 'complex'
            elif any(keyword in sql_query.upper() for keyword in ['WHERE', 'ORDER BY', 'GROUP BY']):
                match.context['query_complexity'] = 'medium'
            else:
                match.context['query_complexity'] = 'simple'
    
    def get_database_statistics(self, matches: List[DetectionMatch]) -> Dict[str, Any]:
        """Get database-specific statistics from matches."""
        stats = {
            'total_database_matches': len(matches),
            'database_type_distribution': {},
            'library_type_distribution': {},
            'operation_distribution': {},
            'orm_usage': 0,
            'async_usage': 0,
            'connection_strings': 0,
            'sql_queries': 0
        }
        
        for match in matches:
            # Database type distribution
            db_type = match.context.get('database_type', 'unknown')
            stats['database_type_distribution'][db_type] = stats['database_type_distribution'].get(db_type, 0) + 1
            
            # Library type distribution
            lib_type = match.context.get('library_type', 'unknown')
            stats['library_type_distribution'][lib_type] = stats['library_type_distribution'].get(lib_type, 0) + 1
            
            # Operation distribution
            operation = match.evidence.get('operation_type')
            if operation:
                stats['operation_distribution'][operation] = stats['operation_distribution'].get(operation, 0) + 1
            
            # ORM usage
            if match.context.get('orm_usage', False):
                stats['orm_usage'] += 1
            
            # Async usage
            if match.evidence.get('is_async', False):
                stats['async_usage'] += 1
            
            # Connection strings and SQL queries
            if match.evidence.get('string_type') == 'connection':
                stats['connection_strings'] += 1
            elif match.evidence.get('string_type') == 'sql':
                stats['sql_queries'] += 1
        
        return stats