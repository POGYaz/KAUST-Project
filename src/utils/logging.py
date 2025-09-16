"""
Logging utilities for the recommendation system.

This module provides standardized logging configuration and utilities
for consistent logging across all components of the system.
"""

import logging
import sys
from pathlib import Path
from typing import Optional, Union

from rich.console import Console
from rich.logging import RichHandler


def setup_logging(
    level: Union[str, int] = "INFO",
    log_file: Optional[Union[str, Path]] = None,
    rich_console: bool = True,
    format_string: Optional[str] = None,
) -> logging.Logger:
    """
    Set up logging configuration with optional file output and rich formatting.
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
        log_file: Optional path to log file for persistent logging.
        rich_console: Whether to use Rich for enhanced console output.
        format_string: Custom format string for log messages.
        
    Returns:
        Configured logger instance.
    """
    # Convert string level to logging constant
    if isinstance(level, str):
        level = getattr(logging, level.upper())
    
    # Create root logger
    logger = logging.getLogger()
    logger.setLevel(level)
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Set up console handler
    if rich_console:
        console = Console(stderr=True)
        console_handler = RichHandler(
            console=console,
            show_time=True,
            show_path=False,
            rich_tracebacks=True,
        )
    else:
        console_handler = logging.StreamHandler(sys.stderr)
        
    # Set console formatter
    if format_string is None:
        if rich_console:
            # Rich handler has built-in formatting
            console_formatter = logging.Formatter("%(message)s")
        else:
            console_formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
    else:
        console_formatter = logging.Formatter(format_string)
    
    console_handler.setFormatter(console_formatter)
    console_handler.setLevel(level)
    logger.addHandler(console_handler)
    
    # Set up file handler if specified
    if log_file is not None:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file)
        file_formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s"
        )
        file_handler.setFormatter(file_formatter)
        file_handler.setLevel(level)
        logger.addHandler(file_handler)
    
    return logger


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance for a specific module or component.
    
    Args:
        name: Name of the logger (typically __name__).
        
    Returns:
        Logger instance.
    """
    return logging.getLogger(name)


class LoggingContext:
    """
    Context manager for temporary logging configuration changes.
    
    Useful for temporarily changing log levels or adding handlers
    for specific operations.
    """
    
    def __init__(
        self,
        logger: Optional[logging.Logger] = None,
        level: Optional[Union[str, int]] = None,
        handlers: Optional[list[logging.Handler]] = None,
    ):
        """
        Initialize logging context manager.
        
        Args:
            logger: Logger to modify (default: root logger).
            level: Temporary logging level.
            handlers: Additional handlers to add temporarily.
        """
        self.logger = logger or logging.getLogger()
        self.new_level = level
        self.new_handlers = handlers or []
        
        # Store original state
        self.original_level = None
        self.original_handlers = []
    
    def __enter__(self) -> logging.Logger:
        """Enter the logging context."""
        # Store original state
        self.original_level = self.logger.level
        self.original_handlers = self.logger.handlers.copy()
        
        # Apply new configuration
        if self.new_level is not None:
            if isinstance(self.new_level, str):
                self.new_level = getattr(logging, self.new_level.upper())
            self.logger.setLevel(self.new_level)
        
        for handler in self.new_handlers:
            self.logger.addHandler(handler)
        
        return self.logger
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit the logging context and restore original state."""
        # Remove temporary handlers
        for handler in self.new_handlers:
            if handler in self.logger.handlers:
                self.logger.removeHandler(handler)
        
        # Restore original level
        if self.original_level is not None:
            self.logger.setLevel(self.original_level)


def log_function_call(logger: Optional[logging.Logger] = None):
    """
    Decorator to log function calls with arguments and return values.
    
    Args:
        logger: Logger instance to use (default: creates logger from function module).
        
    Returns:
        Decorator function.
    """
    def decorator(func):
        nonlocal logger
        if logger is None:
            logger = logging.getLogger(func.__module__)
        
        def wrapper(*args, **kwargs):
            # Log function entry
            logger.debug(
                f"Calling {func.__name__} with args={args}, kwargs={kwargs}"
            )
            
            try:
                # Execute function
                result = func(*args, **kwargs)
                
                # Log successful completion
                logger.debug(f"{func.__name__} completed successfully")
                
                return result
                
            except Exception as e:
                # Log exception
                logger.error(f"{func.__name__} failed with exception: {e}")
                raise
        
        return wrapper
    return decorator


def log_execution_time(logger: Optional[logging.Logger] = None):
    """
    Decorator to log function execution time.
    
    Args:
        logger: Logger instance to use (default: creates logger from function module).
        
    Returns:
        Decorator function.
    """
    def decorator(func):
        nonlocal logger
        if logger is None:
            logger = logging.getLogger(func.__module__)
        
        def wrapper(*args, **kwargs):
            import time
            
            start_time = time.time()
            
            try:
                result = func(*args, **kwargs)
                execution_time = time.time() - start_time
                
                logger.info(
                    f"{func.__name__} completed in {execution_time:.2f} seconds"
                )
                
                return result
                
            except Exception as e:
                execution_time = time.time() - start_time
                logger.error(
                    f"{func.__name__} failed after {execution_time:.2f} seconds: {e}"
                )
                raise
        
        return wrapper
    return decorator
