import os
import logging
import multiprocessing
from pathlib import Path
import concurrent.futures
from .config import Config
from operator import itemgetter
from .simulation import simulate_match
from .utils import individual_to_commandline, Individual
from typing import List, Dict, Any, Tuple, Optional, Union


logger = logging.getLogger(__name__)


def _compute_worker_count(config: Config) -> int:
    # Treat config.num_threads as GameRunner thread count and cap Python workers conservatively
    gr_threads = max(1, int(getattr(config, "num_threads", 1)))
    per_task_threads = 2 * gr_threads + 1
    cpu = os.cpu_count() or 1
    safe = max(1, cpu // per_task_threads)
    
    # Respect user setting but never exceed safe bound
    return max(1, min(int(getattr(config, "num_threads", 1)), safe))


def _worker_init():
    
    # Single threaded math libs and a basic logger inside workers
    import os as _os, sys as _sys, logging as _logging

    _os.environ.setdefault("OMP_NUM_THREADS", "1")
    _os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    _os.environ.setdefault("MKL_NUM_THREADS", "1")
    _os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
    
    if not _logging.getLogger().handlers:
        _logging.basicConfig(level=_logging.INFO, format="%(message)s", stream=_sys.stdout)


def _handle_hof_logic(
    all_individuals: List[Dict[str, Any]],
    run_hall_of_fame: List[Dict[str, Any]],
    config: Config,
    global_gen: int,
    current_mode: str,
    match_cache: Optional[Dict[str, Any]] = None,
):
    """Handles HoF admission logic inside the evaluator where data is consistent."""
    if not all_individuals or config.intra_run_hof_size <= 0:
        return

    # Find the best individual from the combined pool of parents and offspring
    tournament_champion = max(all_individuals, key=itemgetter("fitness"))
    best_fitness = tournament_champion["fitness"]
    best_weights = tournament_champion["weights"][: config.num_weights]
    champion_type = tournament_champion.get("type", "unknown")

    # Create the key from the sliced weights
    champion_key = individual_to_commandline(best_weights, config.weight_precision)
    is_in_hof = any(champion_key == hof_entry.get("id") for hof_entry in run_hall_of_fame)

    if not is_in_hof:
        hof_entry = {"id": champion_key, "weights": best_weights, "fitness": best_fitness, "defeats": 0, "generation_added": global_gen}

        if len(run_hall_of_fame) < config.intra_run_hof_size:
            run_hall_of_fame.append(hof_entry)
            logger.info(
                f"HoF not full. Added new champion (type: {champion_type}, fitness: {best_fitness:.2f}). Size: {len(run_hall_of_fame)}"
            )
        else:
            if config.debug:
                print("\n--- EVALUATOR HOF DEBUG ---")
                print(f"[DEBUG] Generation: {global_gen}")
                print(f"[DEBUG] HoF is full. New champion (type: {champion_type}, fitness: {best_fitness:.2f}) is challenging.")
                print(f"[DEBUG] Champion Key: {champion_key[:90]}...")
                if current_mode == "coevolution" and match_cache:
                    challenger_results = match_cache.get(champion_key, {})
                    print(f"[DEBUG] Champion Key in Match Cache: {champion_key in match_cache}")
                    if challenger_results:
                        print("[DEBUG] Win rates against current HoF members:")
                        for i, hof_member in enumerate(run_hall_of_fame):
                            hof_key = hof_member["id"]  # Use the stored ID
                            if hof_key in challenger_results:
                                wins, games = challenger_results[hof_key]
                                win_rate = (wins / games) * 100 if games > 0 else 0
                                print(f"  - vs HoF member {i} (fit: {hof_member['fitness']:.2f}): {win_rate:.2f}% ({wins}/{games})")
                            else:
                                print(f"  - vs HoF member {i} (fit: {hof_member['fitness']:.2f}): No match data found.")
                print("--- END EVALUATOR HOF DEBUG ---\n")

            if current_mode == "coevolution":
                if not match_cache:
                    logger.warning("Co-evolution HoF logic requires match_cache, but it was not provided.")
                    return

                challenger_results = match_cache.get(champion_key, {})
                best_replacement_candidate_idx = None
                highest_win_rate = 50.0

                for i, hof_member in enumerate(run_hall_of_fame):
                    hof_key = hof_member["id"]
                    if hof_key in challenger_results:
                        wins, games_played = challenger_results[hof_key]
                        win_rate = (wins / games_played) * 100 if games_played > 0 else 0
                        if win_rate > highest_win_rate:
                            highest_win_rate = win_rate
                            best_replacement_candidate_idx = i

                if best_replacement_candidate_idx is not None:
                    member_to_replace = run_hall_of_fame[best_replacement_candidate_idx]
                    logger.info(
                        f"New champion defeated HoF member (fitness {member_to_replace['fitness']:.2f}) with a {highest_win_rate:.2f}% win rate."
                    )
                    logger.info("Replacing member in HoF.")
                    run_hall_of_fame[best_replacement_candidate_idx] = hof_entry
                else:
                    logger.info("New champion did not decisively beat any HoF members. HoF remains unchanged.")

            elif current_mode == "fixed":
                run_hall_of_fame.sort(key=itemgetter("fitness"))
                weakest_member = run_hall_of_fame[0]
                if best_fitness > weakest_member["fitness"]:
                    logger.info(
                        f"New champion (fitness {best_fitness:.2f}) is stronger than weakest HoF member (fitness {weakest_member['fitness']:.2f}). Replacing."
                    )
                    run_hall_of_fame[0] = hof_entry
                else:
                    logger.info(
                        f"New champion (fitness {best_fitness:.2f}) is not stronger than weakest HoF member. HoF remains unchanged."
                    )


def fixed_mode_worker(weights: List[float], args: Dict[str, Any]) -> Tuple[float, Dict[str, Dict[str, int]]]:
    """
    Worker function for fixed mode.
    Returns a tuple of (win_rate, breakdown_dictionary).
    """
    config: Config = args["config"]
    global_hof_weights: List[List[float]] = args.get("hall_of_fame", [])
    run_hall_of_fame: List[Dict[str, Any]] = args.get("run_hall_of_fame") or []
    game_runner_path = Path(args["game_runner_exe_path_str"])

    if config.debug:
        logger.debug(f"Evaluating candidate: {individual_to_commandline(weights, config.weight_precision)[:30]}...")

    total_wins = 0.0
    total_games = 0
    results_breakdown: Dict[str, Dict[str, int]] = {}

    # Evaluate against the static list of fixed opponents
    if config.opponent_bots and config.num_games > 0:
        num_opponents = len(config.opponent_bots)
        games_per_opponent = config.num_games // num_opponents
        remainder = config.num_games % num_opponents

        for i, bot_name in enumerate(config.opponent_bots):
            games_this_opponent = games_per_opponent + (1 if i < remainder else 0)
            if games_this_opponent == 0:
                continue

            total_games += games_this_opponent
            matches_as_p1 = games_this_opponent // 2
            matches_as_p2 = games_this_opponent - matches_as_p1
            wins_this_opponent = 0

            if matches_as_p1 > 0:
                p1_wins, _ = simulate_match(
                    config, "EvolutionaryBot", bot_name, matches_as_p1, game_runner_path, p1_weights=weights, threads=config.num_threads
                )
                wins_this_opponent += p1_wins

            if matches_as_p2 > 0:
                _, p2_wins = simulate_match(
                    config, bot_name, "EvolutionaryBot", matches_as_p2, game_runner_path, p2_weights=weights, threads=config.num_threads
                )
                wins_this_opponent += p2_wins

            total_wins += wins_this_opponent
            results_breakdown[bot_name] = {"wins": wins_this_opponent, "played": games_this_opponent}

    # Evaluate against the global hof
    if config.hof_size > 0 and global_hof_weights and config.hof_num_games > 0:
        for i, hof_weights in enumerate(global_hof_weights):
            total_games += config.hof_num_games
            matches_as_p1 = config.hof_num_games // 2
            matches_as_p2 = config.hof_num_games - matches_as_p1
            wins_vs_hof = 0

            if matches_as_p1 > 0:
                p1_wins, _ = simulate_match(
                    config,
                    "EvolutionaryBot",
                    "EvolutionaryBot",
                    matches_as_p1,
                    game_runner_path,
                    p1_weights=weights,
                    p2_weights=hof_weights,
                    threads=config.num_threads,
                )
                wins_vs_hof += p1_wins

            if matches_as_p2 > 0:
                _, p2_wins = simulate_match(
                    config,
                    "EvolutionaryBot",
                    "EvolutionaryBot",
                    matches_as_p2,
                    game_runner_path,
                    p1_weights=hof_weights,
                    p2_weights=weights,
                    threads=config.num_threads,
                )
                wins_vs_hof += p2_wins

            total_wins += wins_vs_hof
            hof_key = f"global_hof_{i}"
            results_breakdown[hof_key] = {"wins": wins_vs_hof, "played": config.hof_num_games}

    # Evaluate against the intra hof
    if run_hall_of_fame and config.intra_run_hof_size > 0 and config.intra_run_hof_num_games > 0:
        for i, hof_member in enumerate(run_hall_of_fame):
            total_games += config.intra_run_hof_num_games
            hof_weights_item = hof_member["weights"]
            matches_as_p1 = config.intra_run_hof_num_games // 2
            matches_as_p2 = config.intra_run_hof_num_games - matches_as_p1
            wins_vs_hof = 0

            if matches_as_p1 > 0:
                p1_wins, _ = simulate_match(
                    config,
                    "EvolutionaryBot",
                    "EvolutionaryBot",
                    matches_as_p1,
                    game_runner_path,
                    p1_weights=weights,
                    p2_weights=hof_weights_item,
                    threads=config.num_threads,
                )
                wins_vs_hof += p1_wins

            if matches_as_p2 > 0:
                _, p2_wins = simulate_match(
                    config,
                    "EvolutionaryBot",
                    "EvolutionaryBot",
                    matches_as_p2,
                    game_runner_path,
                    p1_weights=hof_weights_item,
                    p2_weights=weights,
                    threads=config.num_threads,
                )
                wins_vs_hof += p2_wins

            total_wins += wins_vs_hof
            hof_key = f"run_hof_{i}"
            results_breakdown[hof_key] = {"wins": wins_vs_hof, "played": config.intra_run_hof_num_games}

    win_rate = (total_wins / total_games) * 100 if total_games > 0 else 0.0
    return win_rate, results_breakdown


def evaluate_fixed_mode_orchestrator(candidates: List[List[float]], args: Dict[str, Any]) -> List[float]:
    """Orchestrates the parallel evaluation for fixed mode and caches results."""
    config = args["config"]
    run_hall_of_fame = args.get("run_hall_of_fame", [])
    ea = args.get("_ec")

    # Snapshot Manager proxies to plain lists for workers
    worker_args = {
        "config": config,
        "hall_of_fame": list(args.get("hall_of_fame", [])),
        "run_hall_of_fame": list(run_hall_of_fame) if run_hall_of_fame else [],
        "game_runner_exe_path_str": args.get("game_runner_exe_path_str"),
    }
    fitness_map = {}
    details_store = args.get("shared_details_store")

    workers = _compute_worker_count(config)
    ctx = multiprocessing.get_context("spawn")
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers, mp_context=ctx, initializer=_worker_init) as executor:
        future_to_index = {executor.submit(fixed_mode_worker, cand, worker_args): i for i, cand in enumerate(candidates)}

        for future in concurrent.futures.as_completed(future_to_index):
            index = future_to_index[future]
            try:
                # Unpack the tuple result from the worker
                fitness, breakdown = future.result()
                fitness_map[index] = fitness

                # Cache it
                if details_store is not None:
                    details_store[tuple(candidates[index])] = breakdown
            except Exception as exc:
                logger.error(f"Candidate evaluation failed for index {index}: {exc}")
                fitness_map[index] = 0.0

    # Reorder the fitness values to match the original candidate order
    ordered_fitness = [fitness_map.get(i, 0.0) for i in range(len(candidates))]

    if ea:
        gen_offset = args.get("run_generations_offset", 0)
        global_gen = gen_offset + ea.num_generations

        all_individuals = []

        # Parents in fixed mode are not revaluated, so use their existing fitness
        for p in ea.population:
            all_individuals.append({"weights": p.candidate, "fitness": p.fitness, "type": "parent"})
        for i, c in enumerate(candidates):
            all_individuals.append({"weights": c, "fitness": ordered_fitness[i], "type": "offspring"})

        current_mode = args.get("current_observer_segment_mode", "fixed")
        _handle_hof_logic(all_individuals, run_hall_of_fame, config, global_gen, current_mode)

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
            config,
            "EvolutionaryBot",
            "EvolutionaryBot",
            matches_as_p1,
            game_runner_path,
            p1_weights=p1_weights,
            p2_weights=p2_weights,
            threads=config.num_threads,
        )
        total_p1_wins += p1_wins_part1
        total_p2_wins += p2_wins_part1

    # Mirror games with P1 as player 2
    if matches_as_p2 > 0:
        p2_wins_part2, p1_wins_part2 = simulate_match(
            config,
            "EvolutionaryBot",
            "EvolutionaryBot",
            matches_as_p2,
            game_runner_path,
            p1_weights=p2_weights,
            p2_weights=p1_weights,
            threads=config.num_threads,
        )
        total_p1_wins += p1_wins_part2
        total_p2_wins += p2_wins_part2

    # Create unique keys for both players
    p1_key = individual_to_commandline(p1_weights, config.weight_precision)

    if battle_type in ["peer", "peer_benchmark", "hof_challenge", "hof", "run_hof"]:
        p2_key = individual_to_commandline(p2_weights, config.weight_precision)
    else:
        p2_key = str(p2_id)

    # The battle_type argument is passed in from the orchestrator
    return p1_id, total_p1_wins, p2_id, total_p2_wins, battle_type, num_games, p1_key, p2_key


def evaluate_coevolution_orchestrator(candidates: List[List[float]], args: Dict[str, Any]) -> List[float]:
    """Orchestrate coevolution evaluation for a generation."""
    config = args["config"]
    global_hof_weights = args.get("hall_of_fame", [])
    run_hall_of_fame = args.get("run_hall_of_fame", [])
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
            pool.append((current_id, "parent", parent.candidate[: config.num_weights], parent))
            current_id += 1

    offspring_start_idx = current_id

    # Add offspring to pool
    for weights in candidates:
        pool.append((current_id, "offspring", weights[: config.num_weights], None))
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
                hof_id = f"run_hof_{i}"
                battle_tasks.append(
                    (comp_id, comp_weights, hof_id, hof_weights, config.intra_run_hof_num_games, config, game_runner_path, "run_hof")
                )

    if not battle_tasks:
        logger.info("No coevolution battles scheduled")
        # Set zero fitness and return
        return handle_no_battles(pool, candidates)

    wins_map, games_played_map, match_cache = run_parallel_battles(battle_tasks, config, pool, run_hall_of_fame)

    # Pass the cache to the observer's details store
    if "shared_details_store" in args and args["shared_details_store"] is not None:
        args["shared_details_store"].clear()
        args["shared_details_store"].update(match_cache)

    # Assign fitness to parents and offspring
    offspring_fitness = assign_fitness(pool, offspring_start_idx, len(candidates), wins_map, games_played_map, config.debug)

    gen_offset = args.get("run_generations_offset", 0)
    global_gen = gen_offset + ea.num_generations

    all_individuals = []
    for p in parents:
        all_individuals.append({"weights": p.candidate, "fitness": p.fitness, "type": "parent"})
    for i, c in enumerate(candidates):
        all_individuals.append({"weights": c, "fitness": offspring_fitness[i], "type": "offspring"})

    current_mode = args.get("current_observer_segment_mode", "coevolution")
    _handle_hof_logic(all_individuals, run_hall_of_fame, config, global_gen, current_mode, match_cache)

    return offspring_fitness


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
) -> Tuple[Dict[Any, float], Dict[Any, int], Dict[str, Any]]:
    """Run battles in parallel and process results."""

    if config.debug:
        logger.debug(f"Running {len(battle_tasks)} coevo battles with {config.num_threads} threads")

    wins_map: Dict[Union[int, str], float] = {comp[0]: 0.0 for comp in pool}
    games_played_map: Dict[Union[int, str], int] = {comp[0]: 0 for comp in pool}
    results = []
    match_cache = {}

    # Execute battles
    workers = _compute_worker_count(config)
    ctx = multiprocessing.get_context("spawn")
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers, mp_context=ctx, initializer=_worker_init) as executor:
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
            return {}, {}, {}

    # Process results
    for result in results:
        if not result:
            continue

        p1_id, p1_wins, p2_id, p2_wins, battle_type, games, p1_key, p2_key = result

        # Store result from P1's perspective
        if p1_key not in match_cache:
            match_cache[p1_key] = {}
        match_cache[p1_key][p2_key] = (p1_wins, games)

        # Store result from P2's perspective
        if p2_key not in match_cache:
            match_cache[p2_key] = {}
        match_cache[p2_key][p1_key] = (p2_wins, games)

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

    return wins_map, games_played_map, match_cache


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
