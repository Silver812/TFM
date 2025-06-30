import sys
import logging
import argparse
from typing import List, Tuple, Optional
from pydantic import BaseModel, Field, field_validator, model_validator, ValidationError

if sys.version_info >= (3, 11):
    from typing import Self
else:
    from typing_extensions import Self

logger = logging.getLogger(__name__)

# Default configuration values
DEFAULT_NUM_GAMES = 100
DEFAULT_NUM_WEIGHTS = 20
DEFAULT_POP_SIZE = 10
DEFAULT_NUM_THREADS = 8
DEFAULT_MAX_EVALUATIONS = 120
DEFAULT_NUM_TRAINING = 3
DEFAULT_DEBUG_MODE = False
DEFAULT_VERBOSE_MODE = False
DEFAULT_LOG_LEVEL = "INFO"
DEFAULT_TRAINING_LOGS = True
DEFAULT_SEEDED_MATCH = False  # Every match will use the same seed if enabled
DEFAULT_SEEDED_TRAINING = True
DEFAULT_SEED = 123
DEFAULT_TURN_TIMEOUT = 1
DEFAULT_MATCH_LOGS = "NONE"
DEFAULT_OPPONENT_BOTS = ["PatronFavorsBot", "MaxAgentsBot", "MaxPrestigeBot", "DecisionTreeBot"]
DEFAULT_EVALUATION_MODE = "hybrid"
DEFAULT_HYBRID_SCHEDULE = "fixed:0.4,coevolution:0.2,fixed:0.4"
DEFAULT_COEVO_PAIRING = "round_robin"
DEFAULT_COEVO_GAMES = 3
DEFAULT_GLOBAL_HOF_SIZE = 0  # 0 to disable global HoF
DEFAULT_GLOBAL_HOF_NUM_GAMES = 0
DEFAULT_WEIGHT_PRECISION = 8
DEFAULT_INTRA_HOF_SIZE = 3
DEFAULT_INTRA_HOF_NUM_GAMES = 3
DEFAULT_INTRA_HOF_PRUNING_PERCENTAGE = 0.35
DEFAULT_INTRA_HOF_PRUNING_FREQUENCY_GENS = 8

#! Note: Total matches per generation are MUCH higher in coevolution moden than fixed mode due to round-robin pairing


def parse_hybrid_schedule(schedule_str: Optional[str]) -> List[Tuple[str, float]]:
    """Parse a hybrid schedule string into a list of (mode, proportion) tuples."""

    if not schedule_str:
        return []

    segments = []
    total_proportion = 0.0

    try:
        parts = schedule_str.strip().lower().split(",")
        for part_idx, part in enumerate(parts):
            if ":" not in part:
                raise ValueError(f"Segment '{part}' is missing a ':' separator.")

            mode, proportion_str = part.split(":", 1)
            mode = mode.strip()
            proportion = float(proportion_str.strip())

            if mode not in ["fixed", "coevolution"]:
                raise ValueError(f"Invalid mode '{mode}' in hybrid schedule.")

            if not (0.0 < proportion <= 1.0):
                raise ValueError(f"Proportion '{proportion}' must be > 0.0 and <= 1.0.")

            segments.append((mode, proportion))
            total_proportion += proportion

        # Normalize proportions if needed
        if abs(total_proportion - 1.0) > 1e-5:
            logger.warning(f"Hybrid schedule proportions sum to {total_proportion:.4f}, normalizing to 1.0")
            if total_proportion > 1e-9:
                segments = [(mode, prop / total_proportion) for mode, prop in segments]
            elif segments:
                raise ValueError("Sum of proportions is too small to normalize.")

    except Exception as e:
        error_msg = f"Error parsing hybrid schedule '{schedule_str}': {e}"
        logger.error(error_msg)
        raise ValueError(error_msg) from e

    return segments


class Config(BaseModel):
    """Configuration for evolutionary bot training."""

    debug: bool = DEFAULT_DEBUG_MODE
    verbose_gamerunner: bool = DEFAULT_VERBOSE_MODE
    num_games: int = Field(DEFAULT_NUM_GAMES, gt=0)
    num_weights: int = Field(DEFAULT_NUM_WEIGHTS, gt=0)
    pop_size: int = Field(DEFAULT_POP_SIZE, ge=2)
    num_threads: int = Field(DEFAULT_NUM_THREADS, gt=0)
    max_evaluations: int = Field(DEFAULT_MAX_EVALUATIONS, gt=0)
    num_training: int = Field(DEFAULT_NUM_TRAINING, gt=0)
    seed: int = DEFAULT_SEED
    turn_timeout: int = Field(DEFAULT_TURN_TIMEOUT, ge=0)
    match_logs: str = DEFAULT_MATCH_LOGS
    training_logs: bool = DEFAULT_TRAINING_LOGS
    opponent_bots: List[str] = Field(default_factory=lambda: list(DEFAULT_OPPONENT_BOTS))
    seeded_match: bool = DEFAULT_SEEDED_MATCH
    seeded_training: bool = DEFAULT_SEEDED_TRAINING
    evaluation_mode: str = DEFAULT_EVALUATION_MODE
    hybrid_schedule_str: Optional[str] = DEFAULT_HYBRID_SCHEDULE
    coevo_pairing_strategy: str = DEFAULT_COEVO_PAIRING
    coevo_games_per_pairing: int = Field(DEFAULT_COEVO_GAMES, ge=0)
    hof_size: int = Field(DEFAULT_GLOBAL_HOF_SIZE, ge=0)
    hof_num_games: int = Field(DEFAULT_GLOBAL_HOF_NUM_GAMES, ge=0)
    log_level: str = DEFAULT_LOG_LEVEL.upper()
    weight_precision: int = DEFAULT_WEIGHT_PRECISION
    intra_run_hof_size: int = Field(DEFAULT_INTRA_HOF_SIZE, ge=0)
    intra_run_hof_num_games: int = Field(DEFAULT_INTRA_HOF_NUM_GAMES, ge=0)
    hof_pruning_percentage: float = Field(DEFAULT_INTRA_HOF_PRUNING_PERCENTAGE, ge=0.0, le=1.0)
    hof_pruning_frequency_gens: int = Field(DEFAULT_INTRA_HOF_PRUNING_FREQUENCY_GENS, ge=0)
    hybrid_schedule_parsed: List[Tuple[str, float]] = Field(default_factory=list)

    @field_validator("match_logs")
    @classmethod
    def validate_match_logs(cls, v: str) -> str:
        if v.upper() not in ["NONE", "P1", "P2", "BOTH"]:
            raise ValueError("match_logs must be one of NONE, P1, P2, BOTH")
        return v.upper()

    @field_validator("evaluation_mode")
    @classmethod
    def validate_evaluation_mode(cls, v: str) -> str:
        if v.lower() not in ["fixed", "coevolution", "hybrid"]:
            raise ValueError("evaluation_mode must be one of fixed, coevolution, hybrid")
        return v.lower()

    @field_validator("coevo_pairing_strategy")
    @classmethod
    def validate_coevo_pairing_strategy(cls, v: str) -> str:
        if v.lower() not in ["round_robin"]:
            raise ValueError("coevo_pairing_strategy must be 'round_robin'")
        return v.lower()

    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if v.upper() not in valid_levels:
            raise ValueError(f"log_level must be one of {valid_levels}")
        return v.upper()

    @model_validator(mode="after")
    def check_and_derive_dependent_fields(self) -> Self:
        # Process hybrid schedule
        if self.evaluation_mode == "hybrid":
            if not self.hybrid_schedule_str:
                raise ValueError("hybrid_schedule_str is required when evaluation_mode is 'hybrid'")
            self.hybrid_schedule_parsed = parse_hybrid_schedule(self.hybrid_schedule_str)
        elif self.hybrid_schedule_str and self.hybrid_schedule_str != DEFAULT_HYBRID_SCHEDULE:
            logger.warning(f"hybrid_schedule_str provided but evaluation_mode is '{self.evaluation_mode}', ignoring")
            self.hybrid_schedule_parsed = []
        else:
            self.hybrid_schedule_parsed = []

        # Validate max_evaluations against pop_size
        if self.max_evaluations < self.pop_size:
            logger.warning(f"max_evaluations ({self.max_evaluations}) < pop_size ({self.pop_size}), adjusting")
            self.max_evaluations = self.pop_size

        # Validate opponent_bots for fixed mode
        if self.evaluation_mode == "fixed" and not self.opponent_bots:
            logger.warning("Fixed evaluation with no opponent_bots, defaulting to ['PatronFavorsBot']")
            self.opponent_bots = ["PatronFavorsBot"]

        return self


def get_config_from_args() -> Config:
    """Parse command-line arguments and create a validated Config."""
    parser = argparse.ArgumentParser(description="Evolutionary Bot Trainer")

    # Training parameters
    parser.add_argument(
        "--num_games", type=int, default=DEFAULT_NUM_GAMES, help=f"Base games per candidate in 'fixed' mode (default: {DEFAULT_NUM_GAMES})"
    )
    parser.add_argument(
        "--num_weights", type=int, default=DEFAULT_NUM_WEIGHTS, help=f"Number of weights for bot (default: {DEFAULT_NUM_WEIGHTS})"
    )
    parser.add_argument("--pop_size", type=int, default=DEFAULT_POP_SIZE, help=f"Population size (default: {DEFAULT_POP_SIZE})")
    parser.add_argument("--num_threads", type=int, default=DEFAULT_NUM_THREADS, help=f"Parallel threads (default: {DEFAULT_NUM_THREADS})")
    parser.add_argument(
        "--max_evaluations", type=int, default=DEFAULT_MAX_EVALUATIONS, help=f"Max evaluations per run (default: {DEFAULT_MAX_EVALUATIONS})"
    )
    parser.add_argument(
        "--num_training", type=int, default=DEFAULT_NUM_TRAINING, help=f"Number of training runs (default: {DEFAULT_NUM_TRAINING})"
    )

    # Logging parameters
    parser.add_argument(
        "--debug", action=argparse.BooleanOptionalAction, default=DEFAULT_DEBUG_MODE, help="Enable debug logging for script logic"
    )
    parser.add_argument(
        "--verbose_gamerunner", action=argparse.BooleanOptionalAction, default=DEFAULT_VERBOSE_MODE, help="Enable verbose GameRunner output"
    )
    parser.add_argument(
        "--log_level",
        type=str,
        default=DEFAULT_LOG_LEVEL,
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help=f"Logging level (default: {DEFAULT_LOG_LEVEL})",
    )
    parser.add_argument(
        "--training_logs",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_TRAINING_LOGS,
        help=f"Enable CSV training logs (default: {DEFAULT_TRAINING_LOGS})",
    )

    # Game parameters
    parser.add_argument(
        "--seeded_match", action=argparse.BooleanOptionalAction, default=DEFAULT_SEEDED_MATCH, help="Use session seed for all matches"
    )
    parser.add_argument(
        "--seeded_training",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_SEEDED_TRAINING,
        help="Use deterministic seeds for each training run based on the master seed.",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help=f"Main random seed (default: {DEFAULT_SEED})")
    parser.add_argument(
        "--turn_timeout",
        type=int,
        default=DEFAULT_TURN_TIMEOUT,
        help=f"Time limit for turns in seconds, 0=unlimited (default: {DEFAULT_TURN_TIMEOUT})",
    )
    parser.add_argument(
        "--match_logs",
        type=str,
        default=DEFAULT_MATCH_LOGS,
        help=f"GameRunner match logs: NONE, P1, P2, BOTH (default: {DEFAULT_MATCH_LOGS})",
    )
    parser.add_argument(
        "--opponent_bots", type=str, nargs="+", default=DEFAULT_OPPONENT_BOTS, help=f"Fixed opponents (default: {DEFAULT_OPPONENT_BOTS})"
    )

    # Evaluation parameters
    parser.add_argument(
        "--evaluation_mode",
        type=str,
        default=DEFAULT_EVALUATION_MODE,
        help=f"Evaluation mode: fixed, coevolution, hybrid (default: {DEFAULT_EVALUATION_MODE})",
    )
    parser.add_argument(
        "--hybrid_schedule_str", type=str, default=DEFAULT_HYBRID_SCHEDULE, help="Hybrid schedule format: 'mode:proportion,mode:proportion'"
    )
    parser.add_argument(
        "--coevo_pairing_strategy",
        type=str,
        default=DEFAULT_COEVO_PAIRING,
        help=f"Coevolution pairing strategy (default: {DEFAULT_COEVO_PAIRING})",
    )
    parser.add_argument(
        "--coevo_games_per_pairing", type=int, default=DEFAULT_COEVO_GAMES, help=f"Games per coevo pairing (default: {DEFAULT_COEVO_GAMES})"
    )

    # Hall of Fame parameters
    parser.add_argument(
        "--hof_size",
        type=int,
        default=DEFAULT_GLOBAL_HOF_SIZE,
        help=f"Hall of Fame size, 0 to disable (default: {DEFAULT_GLOBAL_HOF_SIZE})",
    )
    parser.add_argument(
        "--hof_num_games",
        type=int,
        default=DEFAULT_GLOBAL_HOF_NUM_GAMES,
        help=f"Games vs each HoF member (default: {DEFAULT_GLOBAL_HOF_NUM_GAMES})",
    )
    parser.add_argument(
        "--weight_precision",
        type=int,
        default=DEFAULT_WEIGHT_PRECISION,
        help=f"Weight precision in output (default: {DEFAULT_WEIGHT_PRECISION})",
    )
    parser.add_argument(
        "--intra_run_hof_size",
        type=int,
        default=DEFAULT_INTRA_HOF_SIZE,
        help=f"Max size of the intra-run Hall of Fame (0 to disable). Default: {DEFAULT_INTRA_HOF_SIZE}.",
    )
    parser.add_argument(
        "--hof_pruning_percentage",
        type=float,
        default=DEFAULT_INTRA_HOF_PRUNING_PERCENTAGE,
        help=f"Percentage of HoF to prune based on quality (0.0-1.0). Default: {DEFAULT_INTRA_HOF_PRUNING_PERCENTAGE}.",
    )
    parser.add_argument(
        "--hof_pruning_frequency_gens",
        type=int,
        default=DEFAULT_INTRA_HOF_PRUNING_FREQUENCY_GENS,
        help=f"Frequency in generations to prune the HoF. Default: {DEFAULT_INTRA_HOF_PRUNING_FREQUENCY_GENS}.",
    )
    parser.add_argument(
        "--intra_run_hof_num_games",
        type=int,
        default=DEFAULT_INTRA_HOF_NUM_GAMES,
        help=f"Games vs each intra-run HoF member. Default: {DEFAULT_INTRA_HOF_NUM_GAMES}.",
    )

    args = parser.parse_args()

    try:
        return Config(**vars(args))
    except ValidationError as e:
        print("Configuration Error(s):", file=sys.stderr)
        print(e, file=sys.stderr)
        parser.exit(2)


def log_config_summary(config: Config, main_logger: logging.Logger):
    """Log the configuration summary."""
    main_logger.info("\n--- Configuration Summary ---")

    # Main training parameters
    main_logger.info(f"  Number of Training Runs: {config.num_training}")
    main_logger.info(f"  Population Size: {config.pop_size}")
    main_logger.info(f"  Max Evaluations per Run: {config.max_evaluations}")
    main_logger.info(f"  Number of Weights: {config.num_weights}")

    # Execution and seeding
    main_logger.info(f"  Threads for Parallelism: {config.num_threads}")
    main_logger.info(f"  Master Seed: {config.seed}")
    main_logger.info(f"  Deterministic Training Runs (Seeded): {config.seeded_training}")
    main_logger.info(f"  Deterministic Matches (Seeded): {config.seeded_match}")

    # Evaluation mode
    main_logger.info(f"  Evaluation Mode: {config.evaluation_mode}")
    if config.evaluation_mode == "hybrid":
        main_logger.info(f"     Hybrid Schedule: {config.hybrid_schedule_parsed}")

    # Evaluation mode specifics
    is_fixed_active = config.evaluation_mode == "fixed" or (
        config.evaluation_mode == "hybrid" and any(s[0] == "fixed" for s in config.hybrid_schedule_parsed)
    )
    if is_fixed_active:
        opponents = ", ".join(config.opponent_bots) if config.opponent_bots else "None"
        main_logger.info(f"  Fixed Opponent Bots: {opponents}")
        main_logger.info(f"  Base Games vs Fixed Opponents: {config.num_games}")

    is_coevo_active = config.evaluation_mode == "coevolution" or (
        config.evaluation_mode == "hybrid" and any(s[0] == "coevolution" for s in config.hybrid_schedule_parsed)
    )
    if is_coevo_active:
        main_logger.info(f"  Coevo Pairing Strategy: {config.coevo_pairing_strategy}")
        main_logger.info(f"  Base Coevo Games per Pairing: {config.coevo_games_per_pairing}")

    # Hall of fame parameters
    main_logger.info(f"  Global HoF Size (Inter-Run): {config.hof_size}")
    if config.hof_size > 0:
        main_logger.info(f"     Games vs Each Global HoF Member: {config.hof_num_games}")

    main_logger.info(f"  Intra-Run HoF Size: {config.intra_run_hof_size}")
    if config.intra_run_hof_size > 0:
        main_logger.info(f"     Games vs Each Intra-Run HoF Member: {config.intra_run_hof_num_games}")
        main_logger.info(f"     Pruning Percentage: {config.hof_pruning_percentage * 100:.0f}%")
        main_logger.info(f"     Pruning Frequency (Generations): {config.hof_pruning_frequency_gens}")

    # Game &and logging parameters
    main_logger.info(f"  Turn Timeout: {config.turn_timeout} seconds")
    main_logger.info(f"  Match Logs (GameRunner): {config.match_logs}")
    main_logger.info(f"  Verbose Output (GameRunner): {config.verbose_gamerunner}")
    main_logger.info(f"  Training Logs Enabled (CSV): {config.training_logs}")
    main_logger.info(f"  Log Level: {config.log_level}")
    main_logger.info(f"  Debug Mode (Script): {config.debug}")
    main_logger.info(f"  Weight Precision: {config.weight_precision}")

    main_logger.info("  Note: Benchmark games often use a multiplier on base game values")
    main_logger.info("---------------------------\n")
