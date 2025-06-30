import logging
from pathlib import Path
import concurrent.futures
from .config import Config
from .simulation import simulate_match
from .utils import individual_to_commandline, Individual
from typing import List, Dict, Any, Tuple, Optional, Union


logger = logging.getLogger(__name__)


def fixed_mode_worker(weights: List[float], args: Dict[str, Any]) -> float:
    """Worker function that evaluates a single candidate in fixed mode and returns its results."""
    config = args["config"]
    global_hof_weights = args.get("hall_of_fame", [])
    run_hall_of_fame = args.get("run_hall_of_fame")
    game_runner_path = Path(args["game_runner_exe_path_str"])

    if config.debug:
        logger.debug(f"Evaluating candidate: {individual_to_commandline(weights, config.weight_precision)[:30]}...")

    total_wins = 0.0
    total_games = 0

    # Evaluate against fixed opponents
    if config.opponent_bots:
        total_games += config.num_games
        num_opponents = len(config.opponent_bots)
        games_per_opponent = config.num_games // num_opponents
        remainder = config.num_games % num_opponents

        for i, bot_name in enumerate(config.opponent_bots):
            games_this_opponent = games_per_opponent + (1 if i < remainder else 0)
            if games_this_opponent == 0:
                continue

            matches_as_p1 = games_this_opponent // 2
            matches_as_p2 = games_this_opponent - matches_as_p1

            if matches_as_p1 > 0:
                p1_wins, _ = simulate_match(config, "EvolutionaryBot", bot_name, matches_as_p1, game_runner_path, p1_weights=weights)
                total_wins += p1_wins

            if matches_as_p2 > 0:
                _, p2_wins = simulate_match(config, bot_name, "EvolutionaryBot", matches_as_p2, game_runner_path, p2_weights=weights)
                total_wins += p2_wins

    # Evaluate against global Hall of Fame
    if config.hof_size > 0 and global_hof_weights and config.hof_num_games > 0:
        for hof_weights_item in global_hof_weights:
            total_games += config.hof_num_games
            
            matches_as_p1 = config.hof_num_games // 2
            matches_as_p2 = config.hof_num_games - matches_as_p1

            if matches_as_p1 > 0:
                p1_wins, _ = simulate_match(
                    config,
                    "EvolutionaryBot",
                    "EvolutionaryBot",
                    matches_as_p1,
                    game_runner_path,
                    p1_weights=weights,
                    p2_weights=hof_weights_item,
                )
                total_wins += p1_wins

            if matches_as_p2 > 0:
                _, p2_wins = simulate_match(
                    config,
                    "EvolutionaryBot",
                    "EvolutionaryBot",
                    matches_as_p2,
                    game_runner_path,
                    p1_weights=hof_weights_item,
                    p2_weights=weights,
                )
                total_wins += p2_wins

    # Evaluate against internal Hall of Fame
    if run_hall_of_fame and config.intra_run_hof_size > 0 and config.intra_run_hof_num_games > 0:
        for hof_member in run_hall_of_fame:
            total_games += config.intra_run_hof_num_games
            hof_weights_item = hof_member["weights"]
            
            matches_as_p1 = config.intra_run_hof_num_games // 2
            matches_as_p2 = config.intra_run_hof_num_games - matches_as_p1

            if matches_as_p1 > 0:
                p1_wins, _ = simulate_match(
                    config,
                    "EvolutionaryBot",
                    "EvolutionaryBot",
                    matches_as_p1,
                    game_runner_path,
                    p1_weights=weights,
                    p2_weights=hof_weights_item,
                )
                total_wins += p1_wins

            if matches_as_p2 > 0:
                _, p2_wins = simulate_match(
                    config,
                    "EvolutionaryBot",
                    "EvolutionaryBot",
                    matches_as_p2,
                    game_runner_path,
                    p1_weights=hof_weights_item,
                    p2_weights=weights,
                )
                total_wins += p2_wins

    win_rate = (total_wins / total_games) * 100 if total_games > 0 else 0.0
    return win_rate


def evaluate_fixed_mode_orchestrator(candidates: List[List[float]], args: Dict[str, Any]) -> List[float]:
    """Orchestrates the parallel evaluation for fixed mode."""

    worker_args = {
        "config": args["config"],
        "hall_of_fame": args.get("hall_of_fame", []),
        "run_hall_of_fame": args.get("run_hall_of_fame"),
        "game_runner_exe_path_str": args.get("game_runner_exe_path_str"),
    }
    fitness_map = {}  # This will map the original candidate to its fitness

    with concurrent.futures.ProcessPoolExecutor(max_workers=worker_args["config"].num_threads) as executor:
        future_to_index = {executor.submit(fixed_mode_worker, cand, worker_args): i for i, cand in enumerate(candidates)}

        for future in concurrent.futures.as_completed(future_to_index):
            index = future_to_index[future]
            try:
                fitness = future.result()
                fitness_map[index] = fitness
            except Exception as exc:
                logger.error(f"Candidate evaluation failed for index {index}: {exc}")
                fitness_map[index] = 0.0

    # Reorder the fitness values to match the original candidate order
    ordered_fitness = [fitness_map[i] for i in range(len(candidates))]
    return ordered_fitness


def run_single_coevo_battle_task(
    p1_id: Union[int, str],
    p1_weights: List[float],
    p2_id: Union[int, str],
    p2_weights: List[float],
    num_games: int,
    config: Config,
    game_runner_path: Path,
    battle_type: str,
) -> Tuple[Union[int, str], int, Union[int, str], int, str, int, str, str]:
    """Run a single 'mirror match' battle between two bots for coevolution."""

    if num_games <= 0:
        return p1_id, 0, p2_id, 0, battle_type, 0, "", ""

    # Split logic
    matches_as_p1 = num_games // 2
    matches_as_p2 = num_games - matches_as_p1

    total_p1_wins = 0
    total_p2_wins = 0

    # Run games with P1 as player 1
    if matches_as_p1 > 0:
        p1_wins_part1, p2_wins_part1 = simulate_match(
            config, "EvolutionaryBot", "EvolutionaryBot", matches_as_p1, game_runner_path,
            p1_weights=p1_weights, p2_weights=p2_weights
        )
        total_p1_wins += p1_wins_part1
        total_p2_wins += p2_wins_part1

    # Mirror games with P1 as player 2
    if matches_as_p2 > 0:
        p2_wins_part2, p1_wins_part2 = simulate_match(
            config, "EvolutionaryBot", "EvolutionaryBot", matches_as_p2, game_runner_path,
            p1_weights=p2_weights, p2_weights=p1_weights
        )
        total_p1_wins += p1_wins_part2
        total_p2_wins += p2_wins_part2

    # Keys are no longer needed by the main loop, but we return them for the benchmark
    p1_key = individual_to_commandline(p1_weights, config.weight_precision)
    p2_key = individual_to_commandline(p2_weights, config.weight_precision) if battle_type in ["peer", "peer_benchmark"] else str(p2_id)

    return p1_id, total_p1_wins, p2_id, total_p2_wins, battle_type, num_games, p1_key, p2_key


def evaluate_coevolution_orchestrator(candidates: List[List[float]], args: Dict[str, Any]) -> List[float]:
    """
    Orchestrate coevolution evaluation for a generation.

    Args:
        candidates: New offspring to evaluate
        args: Arguments with config and EA instance

    Returns:
        Fitness values for offspring candidates
    """
    config = args["config"]
    global_hof_weights = args.get("hall_of_fame", [])
    run_hall_of_fame = args.get("run_hall_of_fame")
    game_runner_path = Path(args["game_runner_exe_path_str"])
    ea = args.get("_ec")

    if not ea:
        logger.critical("EA instance (_ec) not found in coevolution args")
        return [0.0] * len(candidates)

    # Build competitor pool of parents + offspring
    parents = ea.population
    pool = []
    current_id = 0

    # Add parents to pool
    for parent in parents:
        if parent.candidate:
            parent_weights = parent.candidate[: config.num_weights]
            pool.append((current_id, "parent", parent_weights, parent))
            current_id += 1

    offspring_start_idx = current_id

    # Add offspring to pool
    for weights in candidates:
        trimmed_weights = weights[: config.num_weights]
        pool.append((current_id, "offspring", trimmed_weights, None))
        current_id += 1

    if not pool:
        logger.warning("Coevolution competitor pool is empty")
        return [0.0] * len(candidates)

    battle_tasks = []

    # Peer battles
    if config.coevo_pairing_strategy == "round_robin" and config.coevo_games_per_pairing > 0:
        for i in range(len(pool)):
            for j in range(i + 1, len(pool)):
                p1_id, _, p1_weights, _ = pool[i]
                p2_id, _, p2_weights, _ = pool[j]
                battle_tasks.append(
                    (p1_id, p1_weights, p2_id, p2_weights, config.coevo_games_per_pairing, config, game_runner_path, "peer")
                )

    # Global Hall of Fame battles
    if config.hof_size > 0 and global_hof_weights and config.hof_num_games > 0:
        for comp_id, _, comp_weights, _ in pool:
            for i, hof_w in enumerate(global_hof_weights):
                hof_id = f"hof_{i}_{hash(tuple(hof_w))%10000:04x}"
                battle_tasks.append((comp_id, comp_weights, hof_id, hof_w, config.hof_num_games, config, game_runner_path, "hof"))

    # Intra Hall of Fame battles
    if run_hall_of_fame and config.intra_run_hof_size > 0 and config.intra_run_hof_num_games > 0:
        for comp_id, _, comp_weights, _ in pool:
            for i, hof_member in enumerate(run_hall_of_fame):
                hof_weights = hof_member["weights"]
                # Use index 'i' to identify the HoF member
                hof_id = f"run_hof_{i}"
                battle_tasks.append(
                    (comp_id, comp_weights, hof_id, hof_weights, config.intra_run_hof_num_games, config, game_runner_path, "run_hof")
                )

    if not battle_tasks:
        logger.info("No coevolution battles scheduled")
        # Set zero fitness and return
        return handle_no_battles(pool, candidates)

    # Run battles in parallel
    wins_map, games_played_map = run_parallel_battles(battle_tasks, config, pool, run_hall_of_fame)

    # Assign fitness to parents and return offspring fitness
    return assign_fitness(pool, offspring_start_idx, len(candidates), wins_map, games_played_map, config.debug)


def handle_no_battles(
    pool: List[Tuple[int, str, List[float], Optional[Individual]]],
    candidates: List[List[float]],
) -> List[float]:
    """Handle case when no battles are scheduled by assigning zero fitness."""

    # Set parent fitness to 0
    for _, comp_type, _, parent in pool:
        if comp_type == "parent" and parent:
            parent.fitness = 0.0

    # Return zero fitness for all new candidates
    return [0.0] * len(candidates)


def run_parallel_battles(
    battle_tasks: List[Tuple], config: Config, pool: List[Tuple[int, str, List[float], Optional[Individual]]], run_hall_of_fame
) -> Tuple[Dict[Union[int, str], float], Dict[Union[int, str], int]]:
    """Run battles in parallel and process results."""

    if config.debug:
        logger.debug(f"Running {len(battle_tasks)} coevo battles with {config.num_threads} threads")

    wins_map: Dict[Union[int, str], float] = {comp[0]: 0.0 for comp in pool}
    games_played_map: Dict[Union[int, str], int] = {comp[0]: 0 for comp in pool}
    results = []

    # Execute battles
    with concurrent.futures.ProcessPoolExecutor(max_workers=config.num_threads) as executor:
        try:
            future_to_task = {executor.submit(run_single_coevo_battle_task, *task): task for task in battle_tasks}

            for future in concurrent.futures.as_completed(future_to_task):
                try:
                    result = future.result()
                    results.append(result)
                except Exception as exc:
                    task = future_to_task[future]
                    logger.error(f"Battle task between {task[0]} and {task[2]} failed: {exc}")
        except Exception as e:
            logger.critical(f"Parallel battle execution error: {e}")
            return {}, {}

    # Process results
    for result in results:
        if not result:
            continue

        p1_id, p1_wins, p2_id, p2_wins, battle_type, games, _, _ = result

        # Track defeats for intra run HoF
        if battle_type == "run_hof":
            if p2_wins < p1_wins:
                try:
                    # Extract index from ID like "run_hof_5"
                    hof_index = int(str(p2_id).split("_")[-1])
                    if run_hall_of_fame and 0 <= hof_index < len(run_hall_of_fame):
                        current_member = run_hall_of_fame[hof_index]
                        current_member["defeats"] += 1
                        run_hall_of_fame[hof_index] = current_member
                except (ValueError, IndexError) as e:
                    logger.error(f"Could not parse HoF index from ID {p2_id}: {e}")

        # Update P1 wins and games played
        wins_map[p1_id] = wins_map.get(p1_id, 0.0) + p1_wins
        games_played_map[p1_id] = games_played_map.get(p1_id, 0) + games

        # For peer battles, also update P2
        if battle_type == "peer":
            p2_pool_id = int(p2_id)
            wins_map[p2_pool_id] = wins_map.get(p2_pool_id, 0.0) + p2_wins
            games_played_map[p2_pool_id] = games_played_map.get(p2_pool_id, 0) + games

    return wins_map, games_played_map


def assign_fitness(
    pool: List[Tuple[int, str, List[float], Optional[Individual]]],
    offspring_start_idx: int,
    num_offspring: int,
    wins_map: Dict[Union[int, str], float],
    games_played_map: Dict[Union[int, str], int],
    debug: bool,
) -> List[float]:
    """Assign fitness to parents and prepare offspring fitness list."""

    offspring_fitness = [0.0] * num_offspring

    for pool_id, comp_type, _, parent in pool:
        wins = wins_map.get(pool_id, 0.0)
        games_played = games_played_map.get(pool_id, 0)

        # Winrate calculation
        fitness = (wins / games_played) * 100 if games_played > 0 else 0.0

        if comp_type == "parent" and parent:
            parent.fitness = fitness
        elif comp_type == "offspring":
            idx = pool_id - offspring_start_idx
            if 0 <= idx < len(offspring_fitness):
                offspring_fitness[idx] = fitness
            else:
                logger.error(f"Invalid offspring index {idx} from pool_id {pool_id}")

    if debug:
        logger.debug(f"Offspring fitness: {offspring_fitness}")

    return offspring_fitness
