#!/usr/bin/env python3
"""
Streaming Analysis for Code Architecture Analyzer

Provides streaming analysis capabilities for large files and datasets
to reduce memory usage and improve processing efficiency.
"""

import logging
from typing import Dict, List, Optional, Any, Callable, Generator, Iterator, Tuple
from dataclasses import dataclass
from abc import ABC, abstractmethod
import time

logger = logging.getLogger(__name__)


@dataclass
class ChunkResult:
    """Result of processing a data chunk."""
    chunk_id: str
    success: bool
    data: Any = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = None


class ChunkProcessor(ABC):
    """Abstract base class for chunk processors."""
    
    @abstractmethod
    def process_chunk(self, chunk_id: str, chunk_data: Any) -> ChunkResult:
        """Process a single chunk of data."""
        pass
    
    @abstractmethod
    def merge_results(self, results: List[ChunkResult]) -> Any:
        """Merge results from multiple chunks."""
        pass


class StreamingAnalyzer:
    """
    Streaming analyzer for processing large files and datasets.
    
    Processes data in chunks to minimize memory usage while maintaining
    analysis accuracy and completeness.
    """
    
    def __init__(self, 
                 chunk_size: int = 1000,
                 overlap_size: int = 0,
                 max_memory_mb: float = 100.0):
        """
        Initialize streaming analyzer.
        
        Args:
            chunk_size: Size of each processing chunk
            overlap_size: Overlap between chunks for context
            max_memory_mb: Maximum memory usage threshold
        """
        self.chunk_size = chunk_size
        self.overlap_size = overlap_size
        self.max_memory_mb = max_memory_mb
        
        # Processing statistics
        self.chunks_processed = 0
        self.total_items_processed = 0
        self.processing_errors = 0
        
        logger.info(f"Streaming analyzer initialized: chunk_size={chunk_size}, "
                   f"overlap={overlap_size}, max_memory={max_memory_mb}MB")
    
    def process_file_streaming(self, 
                             file_path: str,
                             processor: ChunkProcessor,
                             line_based: bool = True) -> Any:
        """
        Process a file using streaming approach.
        
        Args:
            file_path: Path to file to process
            processor: Chunk processor instance
            line_based: Whether to chunk by lines or bytes
            
        Returns:
            Merged results from all chunks
        """
        chunk_results = []
        
        try:
            if line_based:
                chunks = self._chunk_file_by_lines(file_path)
            else:
                chunks = self._chunk_file_by_bytes(file_path)
            
            for chunk_id, chunk_data in chunks:
                try:
                    result = processor.process_chunk(chunk_id, chunk_data)
                    chunk_results.append(result)
                    
                    self.chunks_processed += 1
                    if hasattr(chunk_data, '__len__'):
                        self.total_items_processed += len(chunk_data)
                    
                    if not result.success:
                        self.processing_errors += 1
                        logger.warning(f"Chunk {chunk_id} processing failed: {result.error}")
                
                except Exception as e:
                    self.processing_errors += 1
                    error_result = ChunkResult(
                        chunk_id=chunk_id,
                        success=False,
                        error=str(e)
                    )
                    chunk_results.append(error_result)
                    logger.error(f"Chunk {chunk_id} processing exception: {e}")
            
            # Merge all results
            return processor.merge_results(chunk_results)
            
        except Exception as e:
            logger.error(f"Streaming file processing failed: {e}")
            raise
    
    def process_data_streaming(self,
                             data: List[Any],
                             processor: ChunkProcessor) -> Any:
        """
        Process data list using streaming approach.
        
        Args:
            data: Data list to process
            processor: Chunk processor instance
            
        Returns:
            Merged results from all chunks
        """
        chunk_results = []
        
        try:
            for chunk_id, chunk_data in self._chunk_data(data):
                try:
                    result = processor.process_chunk(chunk_id, chunk_data)
                    chunk_results.append(result)
                    
                    self.chunks_processed += 1
                    self.total_items_processed += len(chunk_data)
                    
                    if not result.success:
                        self.processing_errors += 1
                        logger.warning(f"Chunk {chunk_id} processing failed: {result.error}")
                
                except Exception as e:
                    self.processing_errors += 1
                    error_result = ChunkResult(
                        chunk_id=chunk_id,
                        success=False,
                        error=str(e)
                    )
                    chunk_results.append(error_result)
                    logger.error(f"Chunk {chunk_id} processing exception: {e}")
            
            # Merge all results
            return processor.merge_results(chunk_results)
            
        except Exception as e:
            logger.error(f"Streaming data processing failed: {e}")
            raise
    
    def _chunk_file_by_lines(self, file_path: str) -> Generator[Tuple[str, List[str]], None, None]:
        """Chunk file by lines."""
        chunk_lines = []
        chunk_count = 0
        overlap_buffer = []
        
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                for line_num, line in enumerate(f, 1):
                    chunk_lines.append(line.rstrip('\n\r'))
                    
                    if len(chunk_lines) >= self.chunk_size:
                        # Create chunk with overlap from previous chunk
                        full_chunk = overlap_buffer + chunk_lines
                        chunk_id = f"chunk_{chunk_count}_{line_num-len(chunk_lines)+1}_{line_num}"
                        
                        yield chunk_id, full_chunk
                        
                        # Prepare overlap for next chunk
                        if self.overlap_size > 0:
                            overlap_buffer = chunk_lines[-self.overlap_size:]
                        else:
                            overlap_buffer = []
                        
                        chunk_lines = []
                        chunk_count += 1
                
                # Process remaining lines
                if chunk_lines:
                    full_chunk = overlap_buffer + chunk_lines
                    chunk_id = f"chunk_{chunk_count}_final"
                    yield chunk_id, full_chunk
                    
        except Exception as e:
            logger.error(f"Error chunking file {file_path}: {e}")
            raise
    
    def _chunk_file_by_bytes(self, file_path: str, byte_chunk_size: int = 1024*1024) -> Generator[Tuple[str, bytes], None, None]:
        """Chunk file by bytes."""
        chunk_count = 0
        
        try:
            with open(file_path, 'rb') as f:
                while True:
                    chunk_data = f.read(byte_chunk_size)
                    if not chunk_data:
                        break
                    
                    chunk_id = f"byte_chunk_{chunk_count}"
                    yield chunk_id, chunk_data
                    chunk_count += 1
                    
        except Exception as e:
            logger.error(f"Error chunking file {file_path} by bytes: {e}")
            raise
    
    def _chunk_data(self, data: List[Any]) -> Generator[Tuple[str, List[Any]], None, None]:
        """Chunk data list."""
        chunk_count = 0
        overlap_buffer = []
        
        for i in range(0, len(data), self.chunk_size):
            chunk_data = data[i:i + self.chunk_size]
            
            # Create chunk with overlap from previous chunk
            full_chunk = overlap_buffer + chunk_data
            chunk_id = f"data_chunk_{chunk_count}_{i}_{min(i + self.chunk_size, len(data))}"
            
            yield chunk_id, full_chunk
            
            # Prepare overlap for next chunk
            if self.overlap_size > 0 and len(chunk_data) >= self.overlap_size:
                overlap_buffer = chunk_data[-self.overlap_size:]
            else:
                overlap_buffer = []
            
            chunk_count += 1
    
    def get_streaming_stats(self) -> Dict[str, Any]:
        """Get streaming processing statistics."""
        return {
            'chunks_processed': self.chunks_processed,
            'total_items_processed': self.total_items_processed,
            'processing_errors': self.processing_errors,
            'error_rate': self.processing_errors / max(1, self.chunks_processed),
            'chunk_size': self.chunk_size,
            'overlap_size': self.overlap_size,
            'max_memory_mb': self.max_memory_mb
        }


# Example chunk processors for common analysis tasks
class ASTChunkProcessor(ChunkProcessor):
    """Chunk processor for AST parsing."""
    
    def __init__(self, ast_parser):
        self.ast_parser = ast_parser
        self.results = []
    
    def process_chunk(self, chunk_id: str, chunk_data: List[str]) -> ChunkResult:
        """Process a chunk of source code lines."""
        try:
            # Join lines back into source code
            source_code = '\n'.join(chunk_data)
            
            # Parse with AST parser
            ast_result = self.ast_parser.parse_source(source_code, chunk_id)
            
            return ChunkResult(
                chunk_id=chunk_id,
                success=True,
                data=ast_result,
                metadata={'lines_processed': len(chunk_data)}
            )
            
        except Exception as e:
            return ChunkResult(
                chunk_id=chunk_id,
                success=False,
                error=str(e),
                metadata={'lines_processed': len(chunk_data)}
            )
    
    def merge_results(self, results: List[ChunkResult]) -> Dict[str, Any]:
        """Merge AST results from multiple chunks."""
        merged_functions = []
        merged_classes = []
        merged_imports = []
        total_lines = 0
        errors = []
        
        for result in results:
            if result.success and result.data:
                ast_data = result.data
                merged_functions.extend(ast_data.get('functions', []))
                merged_classes.extend(ast_data.get('classes', []))
                merged_imports.extend(ast_data.get('imports', []))
            
            if result.metadata:
                total_lines += result.metadata.get('lines_processed', 0)
            
            if not result.success:
                errors.append(result.error)
        
        return {
            'functions': merged_functions,
            'classes': merged_classes,
            'imports': merged_imports,
            'total_lines_processed': total_lines,
            'chunk_count': len(results),
            'errors': errors
        }


class ActorDetectionChunkProcessor(ChunkProcessor):
    """Chunk processor for actor detection."""
    
    def __init__(self, pattern_matcher):
        self.pattern_matcher = pattern_matcher
    
    def process_chunk(self, chunk_id: str, chunk_data: List[str]) -> ChunkResult:
        """Process a chunk for actor detection."""
        try:
            # Convert lines to mock AST result for detection
            mock_ast = {
                'module_name': chunk_id,
                'source_lines': chunk_data,
                'functions': [],  # Would need actual function extraction
                'imports': []     # Would need actual import extraction
            }
            
            # Run actor detection
            detected_actors = self.pattern_matcher.detect_patterns(mock_ast)
            
            return ChunkResult(
                chunk_id=chunk_id,
                success=True,
                data=detected_actors,
                metadata={'lines_processed': len(chunk_data)}
            )
            
        except Exception as e:
            return ChunkResult(
                chunk_id=chunk_id,
                success=False,
                error=str(e),
                metadata={'lines_processed': len(chunk_data)}
            )
    
    def merge_results(self, results: List[ChunkResult]) -> Dict[str, Any]:
        """Merge actor detection results from multiple chunks."""
        all_actors = []
        total_lines = 0
        errors = []
        
        for result in results:
            if result.success and result.data:
                all_actors.extend(result.data)
            
            if result.metadata:
                total_lines += result.metadata.get('lines_processed', 0)
            
            if not result.success:
                errors.append(result.error)
        
        # Deduplicate actors by name and type
        unique_actors = {}
        for actor in all_actors:
            key = f"{actor.get('name', '')}_{actor.get('type', '')}"
            if key not in unique_actors:
                unique_actors[key] = actor
        
        return {
            'detected_actors': list(unique_actors.values()),
            'total_actors': len(unique_actors),
            'total_lines_processed': total_lines,
            'chunk_count': len(results),
            'errors': errors
        }


# Convenience functions
def stream_process_large_file(file_path: str, 
                            processor: ChunkProcessor,
                            chunk_size: int = 1000,
                            overlap_size: int = 50) -> Any:
    """
    Convenience function to stream process a large file.
    
    Args:
        file_path: Path to file to process
        processor: Chunk processor instance  
        chunk_size: Lines per chunk
        overlap_size: Lines of overlap between chunks
        
    Returns:
        Merged processing results
    """
    analyzer = StreamingAnalyzer(chunk_size=chunk_size, overlap_size=overlap_size)
    return analyzer.process_file_streaming(file_path, processor, line_based=True)


def stream_process_data_list(data: List[Any],
                           processor: ChunkProcessor,
                           chunk_size: int = 100) -> Any:
    """
    Convenience function to stream process a large data list.
    
    Args:
        data: Data list to process
        processor: Chunk processor instance
        chunk_size: Items per chunk
        
    Returns:
        Merged processing results
    """
    analyzer = StreamingAnalyzer(chunk_size=chunk_size)
    return analyzer.process_data_streaming(data, processor)