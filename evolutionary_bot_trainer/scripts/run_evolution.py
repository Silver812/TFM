import os
import gc
import sys
import csv
import time
import random
import logging
import numpy as np
import multiprocessing
from pathlib import Path

# Project root to path for imports
project_root_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root_dir))

# Evolutionary trainer imports
try:
    from evolutionary_trainer_core.evolution_logic import train_one_run
    from evolutionary_trainer_core.simulation import get_game_runner_path
    from evolutionary_trainer_core.benchmark import run_final_benchmark, save_best_weights
    from evolutionary_trainer_core.config import get_config_from_args, log_config_summary
    from evolutionary_trainer_core.utils import ensure_directory_exists, individual_to_commandline
except ImportError as e:
    print(f"Error importing necessary modules: {e}", file=sys.stderr)
    print("Please ensure all dependencies are installed and the project structure is correct.", file=sys.stderr)
    print(f"Current sys.path includes: {project_root_dir}", file=sys.stderr)
    sys.exit(1)

# Setup logging
logger = logging.getLogger()
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setFormatter(logging.Formatter("%(message)s"))
logger.addHandler(console_handler)


def setup_csv_logging(config, base_dir, session_id_str, csv_fieldnames):
    """Setup CSV logging for dataset creation."""

    if not config.training_logs:
        return None

    ensure_directory_exists(base_dir)

    csv_filename = f"training_log_{session_id_str}_s{config.seed}_m{config.evaluation_mode}.csv"
    csv_path = base_dir / csv_filename

    logger.info(f"Logging training data to CSV: {csv_path}")

    try:
        with open(csv_path, "w", newline="") as csvfile:
            # Header comments
            header_session_id = f"session_{session_id_str}_s{config.seed}_m{config.evaluation_mode}"
            csvfile.write(f"# Experiment Session ID: {header_session_id}\n")
            csvfile.write(f"# Date Generated: {time.strftime('%Y-%m-%d %H:%M:%S %Z')}\n")

            # Main EA parameters
            csvfile.write(f"# Number of Training Runs: {config.num_training}\n")
            csvfile.write(f"# Population Size: {config.pop_size}\n")
            csvfile.write(f"# Max Evaluations per Run: {config.max_evaluations}\n")
            csvfile.write(f"# Number of Weights: {config.num_weights}\n")

            # Execution and seeding
            csvfile.write(f"# Threads for Parallelism: {config.num_threads}\n")
            csvfile.write(f"# Master Seed: {config.seed}\n")
            csvfile.write(f"# Deterministic Training Runs (Seeded): {config.seeded_training}\n")
            csvfile.write(f"# Deterministic Matches (Seeded): {config.seeded_match}\n")

            # Evaluation mode
            csvfile.write(f"# Evaluation Mode: {config.evaluation_mode}\n")
            if config.evaluation_mode == "hybrid":
                csvfile.write(f"# Hybrid Schedule String: {config.hybrid_schedule_str}\n")

            is_fixed = config.evaluation_mode == "fixed" or (
                config.evaluation_mode == "hybrid" and any(s[0] == "fixed" for s in config.hybrid_schedule_parsed)
            )
            is_coevo = config.evaluation_mode == "coevolution" or (
                config.evaluation_mode == "hybrid" and any(s[0] == "coevolution" for s in config.hybrid_schedule_parsed)
            )

            if is_fixed:
                csvfile.write(f"# Base Games vs Fixed Opponents: {config.num_games}\n")
                csvfile.write(f"# Opponent Bots (Fixed Mode): {config.opponent_bots}\n")

            if is_coevo:
                csvfile.write(f"# Base Coevo Games per Peer Pairing: {config.coevo_games_per_pairing}\n")

            # Hall of fame parameters
            csvfile.write(f"# Global HoF Size (Inter-Run): {config.hof_size}\n")
            if config.hof_size > 0:
                csvfile.write(f"#    Games vs Each Global HoF Member: {config.hof_num_games}\n")

            csvfile.write(f"# Intra-Run HoF Size: {config.intra_run_hof_size}\n")
            if config.intra_run_hof_size > 0:
                csvfile.write(f"#    Games vs Each Intra-Run HoF Member: {config.intra_run_hof_num_games}\n")
                csvfile.write(f"#    Pruning Percentage: {config.hof_pruning_percentage * 100:.0f}%\n")
                csvfile.write(f"#    Pruning Frequency (Generations): {config.hof_pruning_frequency_gens}\n")

            # Game and logging parameters
            csvfile.write(f"# (Note: Final champion benchmark games often use a multiplier on base game values)\n")
            csvfile.write(f"# Turn Timeout (sec): {config.turn_timeout}\n")
            csvfile.write(f"# Match Logs (GameRunner): {config.match_logs}\n")
            csvfile.write(f"# Debug Mode (Script): {config.debug}\n")
            csvfile.write(f"# Verbose Mode (GameRunner): {config.verbose_gamerunner}\n")
            csvfile.write(f"# Script Log Level: {config.log_level}\n")
            csvfile.write(f"# Weight Precision: {config.weight_precision}\n")

            csvfile.write("# ---\n")
            writer = csv.DictWriter(csvfile, fieldnames=csv_fieldnames)
            writer.writeheader()

        return csv_path

    except IOError as e:
        logger.error(f"Could not open CSV log file for writing: {e}. Disabling CSV logging.")
        return None


def main():
    if os.name == "nt":
        # If running on Windows, this improves the multiprocessing compatibility
        multiprocessing.freeze_support()

    config = get_config_from_args()

    # Configure logging
    log_level = logging.DEBUG if config.debug else config.log_level
    logger.setLevel(log_level)
    console_handler.setLevel(log_level)

    if config.debug:
        logger.debug("Debug mode enabled for script logic.")

    # Print configuration summary
    log_config_summary(config, logger)

    # Set up execution environment
    game_runner_exe_path = get_game_runner_path(config)
    if not game_runner_exe_path:
        logger.critical("GameRunner executable path could not be determined. Exiting.")
        sys.exit(1)
    logger.info(f"Using GameRunner executable at: {game_runner_exe_path}")

    # Various setup tasks
    np.random.seed(config.seed)
    random.seed(config.seed)
    logger.info(f"Master PRNG initialized with seed: {config.seed}")
    session_start_time = time.time()
    session_id_str = time.strftime("%Y%m%d_%H%M%S")
    training_data_dir = project_root_dir / "training_data"

    # Dataset columns
    csv_fieldnames = [
        "train_run",
        "generation",
        "timestamp",
        "total_time_sec",
        "num_matches_info",
        "evaluations",
        "best_fitness",
        "avg_fitness",
        "median_fitness",
        "min_fitness",
        "std_fitness",
        "best_weights",
        "avg_weights",
        "best_indiv_win_breakdown",
    ]

    session_csv_path = setup_csv_logging(config, training_data_dir, session_id_str, csv_fieldnames)

    # Training
    run_champions_data = []
    global_hall_of_fame = []

    for i in range(config.num_training):
        # Unique seed for each run
        if config.seeded_training:
            # Deterministic path
            run_seed = config.seed + (i * 1000)
            logger.info(f"\n--- Starting Training Run {i+1}/{config.num_training} with seed: {run_seed} ---")
            run_prng = random.Random(run_seed)
        else:
            # Stochastic path
            logger.info(f"\n--- Starting Training Run {i+1}/{config.num_training} with random seed ---")
            run_prng = random.Random()

        # Execute training run
        best_individual, global_hall_of_fame = train_one_run(
            prng_for_ea=run_prng,
            config=config,
            run_number_idx=i,
            session_csv_path=session_csv_path,
            csv_fieldnames=csv_fieldnames,
            global_hall_of_fame_tuples=global_hall_of_fame,
            game_runner_exe_path=game_runner_exe_path,
        )

        if config.debug and global_hall_of_fame:
            logger.debug(f"End of Run {i + 1}, Global HoF size: {len(global_hall_of_fame)}")
            logger.debug(f"  Top HoF fitness: {global_hall_of_fame[0][0]:.2f}" if global_hall_of_fame else "N/A")

        # Add champion data
        if best_individual and best_individual.fitness is not None and best_individual.candidate:
            fitness = float(best_individual.fitness.values[0] if hasattr(best_individual.fitness, "values") else best_individual.fitness)
            run_champions_data.append(
                {
                    "weights": best_individual.candidate[: config.num_weights],
                    "original_run_fitness": fitness,
                    "run_index": i,
                }
            )
            logger.info(f"Run {i+1} champion fitness: {fitness:.2f}")
        else:
            logger.warning(f"Run {i+1} did not produce a valid champion individual.")

        gc.collect()  # Explicitly run garbage collection

    # Final champion clash
    best_overall_weights = None
    best_benchmark_fitness = -float("inf")
    final_benchmark_details = {}

    if run_champions_data:
        best_overall_weights, best_benchmark_fitness, final_benchmark_details = run_final_benchmark(
            config=config,
            champions=run_champions_data,
            hall_of_fame=[w for _, w in global_hall_of_fame],
            game_runner_path=game_runner_exe_path,
        )
    else:
        logger.info("No run champions were produced, skipping final benchmark.")

    # Dataset and weights saving
    total_duration_sec = time.time() - session_start_time
    duration_str = time.strftime("%H:%M:%S", time.gmtime(total_duration_sec))

    logger.info("\n--- Overall Training Summary ---")
    logger.info(f"Total session duration: {duration_str} ({total_duration_sec:.2f} seconds)")

    if best_overall_weights:
        logger.info(f"Best overall benchmark fitness: {best_benchmark_fitness:.2f}")
        logger.info(f"Best Weights (from benchmark):\n{individual_to_commandline(best_overall_weights, config.weight_precision)}")

        save_best_weights(
            config=config,
            best_weights=best_overall_weights,
            benchmark_fitness=best_benchmark_fitness,
            benchmark_details=final_benchmark_details,
            game_runner_dir=game_runner_exe_path.parent,
            session_start_time=session_id_str,
            total_duration_sec=total_duration_sec,
        )
    else:
        logger.info("Training complete. No best overall individual found from benchmark.")

    if session_csv_path and config.training_logs:
        logger.info(f"CSV log for this training session is in: {session_csv_path}")
    elif config.training_logs:
        logger.info("CSV logging was enabled but failed to initialize.")
    else:
        logger.info("CSV logging was disabled for this session.")


if __name__ == "__main__":
    main()
