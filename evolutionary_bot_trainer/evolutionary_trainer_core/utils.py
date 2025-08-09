import logging
from pathlib import Path
from typing import List, TYPE_CHECKING, Any, Protocol, Optional

# Using this constant to avoid runtime circular imports
if TYPE_CHECKING:
    from .config import Config

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


def generate_config_summary(config: "Config", session_id: Optional[str] = None, duration_sec: Optional[float] = None) -> List[str]:
    """Generates a list of strings summarizing the experiment configuration."""
    lines = []

    if session_id:
        lines.append(f"Session ID: {session_id}_s{config.seed}")

    if duration_sec is not None:
        lines.append(f"Duration: {duration_sec:.2f}s")

    # Main EA parameters
    lines.extend(
        [
            f"Training Runs: {config.num_training}",
            f"Population Size: {config.pop_size}",
            f"Max Evaluations/Run: {config.max_evaluations}",
            f"Number of Weights: {config.num_weights}",
            f"Mutation Rate: {config.mutation_rate}",
            f"Crossover Rate: {config.crossover_rate}",
        ]
    )

    # Execution and seeding
    lines.extend(
        [
            f"Threads for Parallelism: {config.num_threads}",
            f"Master Seed: {config.seed}",
            f"Deterministic Training Runs (Seeded): {config.seeded_training}",
            f"Deterministic Matches (Seeded): {config.seeded_match}",
        ]
    )

    # Evaluation mode
    lines.append(f"Evaluation Mode: {config.evaluation_mode}")
    if config.evaluation_mode == "hybrid":
        lines.append(f"  Hybrid Schedule: {config.hybrid_schedule_str}")

    # Base game counts and opponents
    opponents = ", ".join(config.opponent_bots) if config.opponent_bots else "None"
    lines.extend(
        [
            f"Base Games (Fixed): {config.num_games}",
            f"Opponent Bots (Fixed): {opponents}",
            f"Base Games (Coevo): {config.coevo_games_per_pairing}",
        ]
    )

    # Hall of fame parameters
    lines.append(f"Global HoF Size (Inter-Run): {config.hof_size}")
    if config.hof_size > 0:
        lines.append(f"  Games vs Each Global HoF Member: {config.hof_num_games}")

    lines.append(f"Intra-Run HoF Size: {config.intra_run_hof_size}")
    if config.intra_run_hof_size > 0:
        lines.extend(
            [
                f"  Games vs Each Intra-Run HoF Member: {config.intra_run_hof_num_games}",
                f"  Pruning Percentage: {config.hof_pruning_percentage * 100:.0f}%",
                f"  Pruning Frequency (Generations): {config.hof_pruning_frequency_gens}",
            ]
        )

    # Game and logging parameters
    lines.extend(
        [
            f"Turn Timeout: {config.turn_timeout}s",
            f"Weight Precision: {config.weight_precision}",
            f"Match Logs: {config.match_logs}",
            f"Debug Mode: {config.debug}",
        ]
    )

    return lines
