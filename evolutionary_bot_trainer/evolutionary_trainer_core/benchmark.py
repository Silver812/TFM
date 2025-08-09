import time
import logging
import multiprocessing
import concurrent.futures
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional, Union

from .config import Config
from .utils import individual_to_commandline, generate_config_summary
from .evaluators import evaluate_fixed_mode_orchestrator, run_single_coevo_battle_task

logger = logging.getLogger(__name__)

# Multiplier for benchmark games relative to training
BENCHMARK_GAMES_MULTIPLIER = 5


def run_final_benchmark(
    config: Config,
    champions: List[Dict[str, Any]],  # Each: {"weights": list, "original_run_fitness": float, "run_index": int}
    hall_of_fame: List[List[float]],
    game_runner_path: Path,
) -> Tuple[Optional[List[float]], float, Dict[str, Dict[str, int]]]:
    """
    Run final benchmark to determine the best overall champion.

    Args:
        config: Training configuration
        champions: List of champion data from each run
        hall_of_fame: List of weights from hall of fame members
        game_runner_path: Path to GameRunner executable

    Returns:
        Tuple of (best weights, best fitness, detailed results)
    """

    if not champions:
        logger.info("No champions to benchmark")
        return None, -float("inf"), {}

    logger.info(f"\n--- Starting Final Benchmark of {len(champions)} Champions ---")

    best_weights = None
    best_fitness = -float("inf")
    all_champion_details = {}

    # Determine benchmark style based on evaluation mode
    benchmark_style = get_benchmark_style(config)

    # Shared dictionary for collecting the results
    with multiprocessing.Manager() as manager:
        details_collector = manager.dict()

        if benchmark_style == "fixed":
            best_weights, best_fitness = run_fixed_benchmark(
                config, champions, hall_of_fame, game_runner_path, details_collector, all_champion_details
            )
        elif benchmark_style == "coevolution":
            best_weights, best_fitness = run_coevo_benchmark(config, champions, hall_of_fame, game_runner_path, all_champion_details)
        else:
            logger.error(f"Unsupported benchmark style: {benchmark_style}")
            return None, -float("inf"), {}

    # Extract details for the best champion
    best_details = {}
    if best_weights:
        best_key = tuple(best_weights)
        best_details = all_champion_details.get(best_key, {})

    return best_weights, best_fitness, best_details


def get_benchmark_style(config: Config) -> str:
    """Determine which benchmark style to use based on config."""

    if config.evaluation_mode != "hybrid":
        return config.evaluation_mode
    else:
        return "fixed"


def run_fixed_benchmark(
    config: Config,
    champions: List[Dict[str, Any]],
    hall_of_fame: List[List[float]],
    game_runner_path: Path,
    details_collector: Union[Dict, Any],
    all_champion_details: Dict,
) -> Tuple[Optional[List[float]], float]:
    """Run benchmark using fixed opponents."""

    logger.info(f"Benchmark Style: Fixed (vs {config.opponent_bots} and HoF). " f"Games x{BENCHMARK_GAMES_MULTIPLIER}")

    # Adjust number of games for better statistical significance
    benchmark_config = config.model_copy(
        update={
            "num_games": config.num_games * BENCHMARK_GAMES_MULTIPLIER,
            "hof_num_games": config.hof_num_games * BENCHMARK_GAMES_MULTIPLIER,
        }
    )

    args = {
        "config": benchmark_config,
        "shared_details_store": details_collector,
        "hall_of_fame": hall_of_fame,
        "game_runner_exe_path_str": str(game_runner_path),
        "is_benchmark_call": True,
    }

    best_weights = None
    best_fitness = -float("inf")

    for champ in champions:
        logger.info(f"Benchmarking Champion from Run {champ['run_index'] + 1} (Fixed)...")

        # Evaluate champion against fixed opponents
        fitness_list = evaluate_fixed_mode_orchestrator([champ["weights"]], args)
        fitness = fitness_list[0] if fitness_list else -float("inf")

        champ["benchmark_fitness"] = fitness
        logger.info(f"  Run {champ['run_index'] + 1} Champion Fitness: {fitness:.2f}")

        # Store details
        champ_key = tuple(champ["weights"])
        if champ_key in details_collector:
            all_champion_details[champ_key] = dict(details_collector[champ_key])
            details_collector.clear()

        # Track best champion
        if fitness > best_fitness:
            best_fitness = fitness
            best_weights = champ["weights"]

    return best_weights, best_fitness


def run_coevo_benchmark(
    config: Config,
    champions: List[Dict[str, Any]],
    hall_of_fame: List[List[float]],
    game_runner_path: Path,
    all_champion_details: Dict,
) -> Tuple[Optional[List[float]], float]:
    """Run benchmark using a coevolution tournament."""

    logger.info(f"Benchmark Style: Coevolution Tournament (and HoF if active). " f"Games x{BENCHMARK_GAMES_MULTIPLIER}")

    # Adjust number of games for better statistical significance
    benchmark_config = config.model_copy(
        update={
            "coevo_games_per_pairing": config.coevo_games_per_pairing * BENCHMARK_GAMES_MULTIPLIER,
            "hof_num_games": config.hof_num_games * BENCHMARK_GAMES_MULTIPLIER,
        }
    )

    # Prepare battle matchups and run battles in parallel
    battle_tasks = prepare_coevo_battle_tasks(champions, hall_of_fame, benchmark_config, game_runner_path)
    results = execute_battle_tasks(battle_tasks, config.num_threads)
    champion_wins = {f"champ_{c['run_index']}": 0.0 for c in champions}

    for result in filter(None, results):
        p1_id, p1_wins, p2_id, p2_wins, battle_type, games, p1_key, p2_key = result

        # Track wins for champions
        if str(p1_id) in champion_wins:
            champion_wins[str(p1_id)] += p1_wins

        # Store battle details for p1
        p1_details = all_champion_details.setdefault(p1_key, {})
        p1_vs_p2 = p1_details.setdefault(str(p2_key), {"wins": 0, "played": 0})
        p1_vs_p2["wins"] += p1_wins
        p1_vs_p2["played"] += games

        # For peer battles between champions, also track p2's results
        if battle_type == "peer_benchmark" and str(p2_id) in champion_wins:
            champion_wins[str(p2_id)] += p2_wins

            # Store battle details for p2
            if isinstance(p2_key, tuple):
                p2_details = all_champion_details.setdefault(p2_key, {})
                p2_vs_p1 = p2_details.setdefault(str(p1_key), {"wins": 0, "played": 0})
                p2_vs_p1["wins"] += p2_wins
                p2_vs_p1["played"] += games

    # Find best champion
    best_weights = None
    best_fitness = -float("inf")

    for champ in champions:
        champ_id = f"champ_{champ['run_index']}"
        fitness = champion_wins.get(champ_id, 0.0)
        champ["benchmark_fitness"] = fitness

        logger.info(f"  Run {champ['run_index'] + 1} Champion Fitness: {fitness:.2f}")

        if fitness > best_fitness:
            best_fitness = fitness
            best_weights = champ["weights"]

    return best_weights, best_fitness


def prepare_coevo_battle_tasks(
    champions: List[Dict[str, Any]],
    hall_of_fame: List[List[float]],
    config: Config,
    game_runner_path: Path,
) -> List[Tuple]:
    """Prepare battle tasks for coevolution benchmark."""

    tasks = []

    # Champion vs champion battles (round-robin)
    for i in range(len(champions)):
        for j in range(i + 1, len(champions)):
            p1_champ = champions[i]
            p2_champ = champions[j]
            tasks.append(
                (
                    f"champ_{p1_champ['run_index']}",
                    p1_champ["weights"],
                    f"champ_{p2_champ['run_index']}",
                    p2_champ["weights"],
                    config.coevo_games_per_pairing,
                    config,
                    game_runner_path,
                    "peer_benchmark",
                )
            )

    # Champions vs hall of Fame
    if config.hof_size > 0 and hall_of_fame and config.hof_num_games > 0:
        for champ in champions:
            for i, hof_weights in enumerate(hall_of_fame):
                hof_id = f"hof_{i}_{hash(tuple(hof_weights))%10000:04x}"
                tasks.append(
                    (
                        f"champ_{champ['run_index']}",
                        champ["weights"],
                        hof_id,
                        hof_weights,
                        config.hof_num_games,
                        config,
                        game_runner_path,
                        "hof_benchmark",
                    )
                )

    return tasks


def execute_battle_tasks(battle_tasks: List[Tuple], num_threads: int) -> List[Tuple]:
    """Execute battle tasks in parallel."""

    results = []

    with concurrent.futures.ProcessPoolExecutor(max_workers=num_threads) as executor:
        future_to_task = {executor.submit(run_single_coevo_battle_task, *task): task for task in battle_tasks}

        for future in concurrent.futures.as_completed(future_to_task):
            try:
                results.append(future.result())
            except Exception as exc:
                task = future_to_task[future]
                logger.error(f"Benchmark battle {task[0]} vs {task[2]} failed: {exc}")

    return results


def save_best_weights(
    config: Config,
    all_champions_data: List[Dict[str, Any]],
    benchmark_details: Dict[str, Dict[str, int]],
    game_runner_dir: Path,
    session_start_time: str,
    total_duration_sec: float,
):
    """Saves all champion weights sorted by fitness"""

    output_file = game_runner_dir / "eb_best_weights.txt"
    logger.info(f"Saving all {len(all_champions_data)} champion weights to {output_file}")

    # Sort the champions to find the best one for the summary
    sorted_champions = sorted(all_champions_data, key=lambda c: c.get("benchmark_fitness", -1.0), reverse=True)
    best_champion_fitness = sorted_champions[0].get("benchmark_fitness", 0.0) if sorted_champions else 0.0

    try:
        with open(output_file, "w") as f:
            f.write("### Overall Best Champion Benchmark Summary ###\n")
            f.write(f"# Fitness (Win Rate %): {best_champion_fitness:.4f}\n")

            if benchmark_details:
                f.write("# Win Breakdown:\n")
                for opponent, stats in sorted(benchmark_details.items()):
                    f.write(f"#   vs {opponent}: {stats.get('wins',0)}/{stats.get('played',0)}\n")
            else:
                f.write("# Win Breakdown: No details available\n")
            f.write("\n")

            # Write the weights for each champion, sorted by rank
            f.write("### All Champion Weights (Sorted by Fitness) ###\n")
            for i, champ in enumerate(sorted_champions):
                fitness = champ.get("benchmark_fitness", 0.0)
                weights = champ.get("weights", [])

                f.write(f"# Rank {i + 1}, Fitness: {fitness:.4f}, Run: {champ.get('run_index', -1) + 1}\n")
                f.write(individual_to_commandline(weights, config.weight_precision) + "\n")
            f.write("\n")

            # Write the full configuration summary
            f.write("### Configuration ###\n")
            f.write(f"# Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")

            summary_lines = generate_config_summary(config, session_id=session_start_time, duration_sec=total_duration_sec)
            for line in summary_lines:
                f.write(f"# {line}\n")

    except IOError as e:
        logger.error(f"Failed to write weights file: {e}")
