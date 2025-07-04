# src/innovation_catalyst/utils/logging.py
"""
Advanced logging system for Innovation Catalyst Agent.
Provides structured logging with performance monitoring and error tracking.
"""

import logging
import logging.handlers
import sys
import time
import functools
import traceback
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Union
from datetime import datetime
from contextlib import contextmanager
import json

from .config import get_config

class PerformanceLogger:
    """
    Performance monitoring and logging utility.
    
    Features:
        - Function execution timing
        - Memory usage tracking
        - Performance metrics collection
        - Structured performance data
    """
    
    def __init__(self, logger_name: str = "performance"):
        self.logger = logging.getLogger(logger_name)
        self.metrics: Dict[str, list] = {}
    
    def time_function(self, func_name: Optional[str] = None):
        """Decorator to time function execution."""
        def decorator(func: Callable) -> Callable:
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                name = func_name or f"{func.__module__}.{func.__name__}"
                start_time = time.perf_counter()
                
                try:
                    result = func(*args, **kwargs)
                    success = True
                    error = None
                except Exception as e:
                    success = False
                    error = str(e)
                    raise
                finally:
                    end_time = time.perf_counter()
                    duration = end_time - start_time
                    
                    # Log performance data
                    self.log_performance(
                        function_name=name,
                        duration=duration,
                        success=success,
                        error=error,
                        args_count=len(args),
                        kwargs_count=len(kwargs)
                    )
                
                return result
            return wrapper
        return decorator
    
    def log_performance(
        self,
        function_name: str,
        duration: float,
        success: bool,
        error: Optional[str] = None,
        **metadata
    ) -> None:
        """Log performance metrics."""
        perf_data = {
            "timestamp": datetime.utcnow().isoformat(),
            "function": function_name,
            "duration_seconds": round(duration, 4),
            "success": success,
            "error": error,
            **metadata
        }
        
        # Store metrics
        if function_name not in self.metrics:
            self.metrics[function_name] = []
        self.metrics[function_name].append(perf_data)
        
        # Log based on performance and success
        if not success:
            self.logger.error(f"Function failed: {function_name}", extra=perf_data)
        elif duration > 10.0:  # Slow function threshold
            self.logger.warning(f"Slow function: {function_name}", extra=perf_data)
        else:
            self.logger.debug(f"Function completed: {function_name}", extra=perf_data)
    
    @contextmanager
    def time_block(self, block_name: str):
        """Context manager for timing code blocks."""
        start_time = time.perf_counter()
        try:
            yield
            success = True
            error = None
        except Exception as e:
            success = False
            error = str(e)
            raise
        finally:
            end_time = time.perf_counter()
            duration = end_time - start_time
            self.log_performance(
                function_name=f"block:{block_name}",
                duration=duration,
                success=success,
                error=error
            )
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get summary of performance metrics."""
        summary = {}
        
        for func_name, metrics in self.metrics.items():
            durations = [m["duration_seconds"] for m in metrics]
            successes = [m["success"] for m in metrics]
            
            summary[func_name] = {
                "call_count": len(metrics),
                "success_rate": sum(successes) / len(successes) if successes else 0,
                "avg_duration": sum(durations) / len(durations) if durations else 0,
                "min_duration": min(durations) if durations else 0,
                "max_duration": max(durations) if durations else 0,
                "total_duration": sum(durations)
            }
        
        return summary

class StructuredFormatter(logging.Formatter):
    """
    Custom formatter for structured logging.
    
    Features:
        - JSON structured output
        - Consistent field naming
        - Error context preservation
        - Performance data inclusion
    """
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record as structured JSON."""
        # Base log data
        log_data = {
            "timestamp": datetime.fromtimestamp(record.created).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }
        
        # Add exception information if present
        if record.exc_info:
            log_data["exception"] = {
                "type": record.exc_info[0].__name__,
                "message": str(record.exc_info[1]),
                "traceback": traceback.format_exception(*record.exc_info)
            }
        
        # Add extra fields from record
        for key, value in record.__dict__.items():
            if key not in {
                'name', 'msg', 'args', 'levelname', 'levelno', 'pathname',
                'filename', 'module', 'exc_info', 'exc_text', 'stack_info',
                'lineno', 'funcName', 'created', 'msecs', 'relativeCreated',
                'thread', 'threadName', 'processName', 'process', 'message'
            }:
                log_data[key] = value
        
        return json.dumps(log_data, default=str, ensure_ascii=False)

class LoggingManager:
    """
    Centralized logging management for Innovation Catalyst Agent.
    
    Features:
        - Multiple output formats (console, file, structured)
        - Log rotation and retention
        - Performance logging integration
        - Error tracking and alerting
        - Configuration-driven setup
    """
    
    def __init__(self):
        self.config = get_config()
        self.performance_logger = PerformanceLogger()
        self._setup_logging()
    
    def _setup_logging(self) -> None:
        """Setup logging configuration."""
        # Create logs directory
        self.config.logs_dir.mkdir(parents=True, exist_ok=True)
        
        # Configure root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(getattr(logging, self.config.log_level.upper()))
        
        # Clear existing handlers
        root_logger.handlers.clear()
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        console_handler.setFormatter(console_formatter)
        root_logger.addHandler(console_handler)
        
        # File handler with rotation
        file_handler = logging.handlers.RotatingFileHandler(
            filename=self.config.logs_dir / "innovation_catalyst.log",
            maxBytes=10 * 1024 * 1024,  # 10MB
            backupCount=5,
            encoding='utf-8'
        )
        file_handler.setLevel(logging.DEBUG)
        file_formatter = logging.Formatter(self.config.log_format)
        file_handler.setFormatter(file_formatter)
        root_logger.addHandler(file_handler)
        
        # Structured JSON handler
        json_handler = logging.handlers.RotatingFileHandler(
            filename=self.config.logs_dir / "innovation_catalyst.json",
            maxBytes=10 * 1024 * 1024,  # 10MB
            backupCount=5,
            encoding='utf-8'
        )
        json_handler.setLevel(logging.DEBUG)
        json_handler.setFormatter(StructuredFormatter())
        root_logger.addHandler(json_handler)
        
        # Error handler (separate file for errors)
        error_handler = logging.handlers.RotatingFileHandler(
            filename=self.config.logs_dir / "errors.log",
            maxBytes=5 * 1024 * 1024,  # 5MB
            backupCount=3,
            encoding='utf-8'
        )
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(file_formatter)
        root_logger.addHandler(error_handler)
        
        # Performance handler
        perf_handler = logging.handlers.RotatingFileHandler(
            filename=self.config.logs_dir / "performance.log",
            maxBytes=5 * 1024 * 1024,  # 5MB
            backupCount=3,
            encoding='utf-8'
        )
        perf_handler.setLevel(logging.DEBUG)
        perf_handler.setFormatter(StructuredFormatter())
        
        # Add performance handler to performance logger
        self.performance_logger.logger.addHandler(perf_handler)
        self.performance_logger.logger.setLevel(logging.DEBUG)
    
    def get_logger(self, name: str) -> logging.Logger:
        """Get a logger instance."""
        return logging.getLogger(name)
    
    def get_performance_logger(self) -> PerformanceLogger:
        """Get the performance logger instance."""
        return self.performance_logger

# Global logging manager
logging_manager = LoggingManager()

def get_logger(name: str) -> logging.Logger:
    """Get a logger instance."""
    return logging_manager.get_logger(name)

def get_performance_logger() -> PerformanceLogger:
    """Get the performance logger instance."""
    return logging_manager.get_performance_logger()

def log_function_performance(func_name: Optional[str] = None):
    """Decorator for logging function performance."""
    return logging_manager.performance_logger.time_function(func_name)

@contextmanager
def log_block_performance(block_name: str):
    """Context manager for logging block performance."""
    with logging_manager.performance_logger.time_block(block_name):
        yield
