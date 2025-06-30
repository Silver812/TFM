import logging
from typing import List
from pathlib import Path
from typing import Any, List, Protocol

logger = logging.getLogger(__name__)

class Individual(Protocol):
    """Protocol for Inspyred individuals."""
    
    candidate: List[float]
    fitness: Any
    birth_generation: int

def individual_to_commandline(individual_weights: List[float], precision: int = 8) -> str:
    """
    Convert a list of weights to a comma-separated string with specified precision.
    
    Args:
        individual_weights: List of float weights
        precision: Decimal precision for output
        
    Returns:
        Comma-separated string of formatted weights
    """
    
    if not isinstance(individual_weights, list):
        logger.error(f"Expected list of weights, got {type(individual_weights).__name__}")
        return ""
        
    return ",".join(f"{weight:.{precision}f}" for weight in individual_weights)


def ensure_directory_exists(dir_path: Path) -> None:
    """
    Create directory if it doesn't exist.
    
    Args:
        dir_path: Path to ensure exists
        
    Raises:
        OSError: If directory creation fails
    """
    
    try:
        dir_path.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        logger.error(f"Failed to create directory {dir_path}: {e}")
        raise  # Re-raise since directory creation is critical