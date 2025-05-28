"""
General utility functions.
"""

import os
import logging
from datetime import datetime
from pathlib import Path
from typing import Union


def setup_logging(level: str = "INFO", 
                 log_file: Union[str, Path] = None,
                 format_string: str = None) -> logging.Logger:
    """
    Setup logging configuration.
    
    Parameters:
    -----------
    level : str
        Logging level
    log_file : str or Path, optional
        Path to log file
    format_string : str, optional
        Custom format string
        
    Returns:
    --------
    logging.Logger
        Configured logger
    """
    if format_string is None:
        format_string = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        
    # Convert string level to logging constant
    numeric_level = getattr(logging, level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f'Invalid log level: {level}')
        
    # Configure logging
    handlers = [logging.StreamHandler()]
    
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_path))
        
    logging.basicConfig(
        level=numeric_level,
        format=format_string,
        handlers=handlers
    )
    
    return logging.getLogger(__name__)


def ensure_dir(directory: Union[str, Path]) -> Path:
    """
    Ensure directory exists, create if it doesn't.
    
    Parameters:
    -----------
    directory : str or Path
        Directory path
        
    Returns:
    --------
    Path
        Directory path as Path object
    """
    dir_path = Path(directory)
    dir_path.mkdir(parents=True, exist_ok=True)
    return dir_path


def get_timestamp(format_string: str = "%Y%m%d_%H%M%S") -> str:
    """
    Get current timestamp as string.
    
    Parameters:
    -----------
    format_string : str
        Timestamp format string
        
    Returns:
    --------
    str
        Formatted timestamp
    """
    return datetime.now().strftime(format_string)


def validate_file_path(file_path: Union[str, Path], 
                      must_exist: bool = True) -> Path:
    """
    Validate and return file path.
    
    Parameters:
    -----------
    file_path : str or Path
        File path to validate
    must_exist : bool
        Whether file must exist
        
    Returns:
    --------
    Path
        Validated file path
        
    Raises:
    -------
    FileNotFoundError
        If file doesn't exist and must_exist is True
    ValueError
        If path is not a file
    """
    path = Path(file_path)
    
    if must_exist and not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
        
    if path.exists() and not path.is_file():
        raise ValueError(f"Path is not a file: {path}")
        
    return path


def validate_dir_path(dir_path: Union[str, Path], 
                     create_if_missing: bool = False) -> Path:
    """
    Validate and return directory path.
    
    Parameters:
    -----------
    dir_path : str or Path
        Directory path to validate
    create_if_missing : bool
        Whether to create directory if it doesn't exist
        
    Returns:
    --------
    Path
        Validated directory path
        
    Raises:
    -------
    NotADirectoryError
        If path exists but is not a directory
    FileNotFoundError
        If directory doesn't exist and create_if_missing is False
    """
    path = Path(dir_path)
    
    if path.exists() and not path.is_dir():
        raise NotADirectoryError(f"Path is not a directory: {path}")
        
    if not path.exists():
        if create_if_missing:
            path.mkdir(parents=True, exist_ok=True)
        else:
            raise FileNotFoundError(f"Directory not found: {path}")
            
    return path


def get_size_string(size_bytes: int) -> str:
    """
    Convert byte size to human readable string.
    
    Parameters:
    -----------
    size_bytes : int
        Size in bytes
        
    Returns:
    --------
    str
        Human readable size string
    """
    if size_bytes == 0:
        return "0B"
        
    size_names = ["B", "KB", "MB", "GB", "TB"]
    i = 0
    
    while size_bytes >= 1024 and i < len(size_names) - 1:
        size_bytes /= 1024.0
        i += 1
        
    return f"{size_bytes:.1f} {size_names[i]}"


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """
    Safely divide two numbers, returning default if denominator is zero.
    
    Parameters:
    -----------
    numerator : float
        Numerator
    denominator : float
        Denominator
    default : float
        Default value if denominator is zero
        
    Returns:
    --------
    float
        Division result or default
    """
    try:
        return numerator / denominator if denominator != 0 else default
    except (ZeroDivisionError, TypeError):
        return default


def chunks(lst: list, chunk_size: int):
    """
    Yield successive chunks of specified size from list.
    
    Parameters:
    -----------
    lst : list
        List to chunk
    chunk_size : int
        Size of each chunk
        
    Yields:
    -------
    list
        Chunks of the original list
    """
    for i in range(0, len(lst), chunk_size):
        yield lst[i:i + chunk_size]


def flatten_dict(d: dict, parent_key: str = '', sep: str = '.') -> dict:
    """
    Flatten nested dictionary.
    
    Parameters:
    -----------
    d : dict
        Dictionary to flatten
    parent_key : str
        Parent key prefix
    sep : str
        Separator for nested keys
        
    Returns:
    --------
    dict
        Flattened dictionary
    """
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def get_memory_usage() -> Dict[str, float]:
    """
    Get current memory usage statistics.
    
    Returns:
    --------
    Dict[str, float]
        Memory usage statistics in MB
    """
    try:
        import psutil
        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()
        
        return {
            'rss': memory_info.rss / 1024 / 1024,  # MB
            'vms': memory_info.vms / 1024 / 1024,  # MB
            'percent': process.memory_percent()
        }
    except ImportError:
        return {'error': 'psutil not available'}


def get_cpu_usage() -> float:
    """
    Get current CPU usage percentage.
    
    Returns:
    --------
    float
        CPU usage percentage
    """
    try:
        import psutil
        return psutil.cpu_percent(interval=1)
    except ImportError:
        return -1.0