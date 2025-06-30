import copy
import uuid
import math
import random
import logging
import inspyred
import numpy as np
import multiprocessing
from pathlib import Path
from multiprocessing.managers import ListProxy
from typing import List, Dict, Any, Tuple, Optional


from .config import Config
from .utils import Individual
from .observers import LoggingObserver
from .evaluators import evaluate_fixed_mode_orchestrator, evaluate_coevolution_orchestrator

logger = logging.getLogger(__name__)


def generate_weights(prng: random.Random, args: Dict[str, Any]) -> List[float]:
    """Generate random weights between 0 and 1."""
    config: Config = args["config"]
    return [prng.uniform(0, 1) for _ in range(config.num_weights)]


def train_one_run(
    prng_for_ea: random.Random,
    config: Config,
    run_number_idx: int,
    session_csv_path: Optional[Path],
    csv_fieldnames: List[str],
    global_hall_of_fame_tuples: List[Tuple[float, List[float]]],
    game_runner_exe_path: Path,
) -> Tuple[Optional[Individual], List[Tuple[float, List[float]]]]:
    """
    Run one training session of the evolutionary algorithm.

    Args:
        prng_for_ea: Random number generator for the EA
        config: Configuration parameters
        run_number_idx: Index of the current training run (0-indexed)
        session_csv_path: Path to CSV file for logging
        csv_fieldnames: Column names for CSV
        global_hall_of_fame_tuples: List of (fitness, weights) tuples from previous runs
        game_runner_exe_path: Path to GameRunner executable

    Returns:
        Tuple of (best individual, updated hall of fame)
    """

    logger.info(f"Starting Training Run {run_number_idx + 1}/{config.num_training} (HoF size: {len(global_hall_of_fame_tuples)})")

    # Get hall of fame weights
    hof_weights = [weights for _, weights in global_hall_of_fame_tuples]

    with multiprocessing.Manager() as manager:
        run_hall_of_fame = manager.list()  # This list is mutable, to track the best individuals of this train run
        ea = inspyred.ec.ES(prng_for_ea)

        # Set up observers
        observers = []
        custom_observer = None
        if config.training_logs and session_csv_path:
            custom_observer = LoggingObserver(
                config=config,
                csv_path=session_csv_path,
                csv_fields=csv_fieldnames,
                train_run=run_number_idx,
                run_hall_of_fame=run_hall_of_fame,
            )
            observers.append(custom_observer)

        if observers:
            ea.observer = observers  # type: ignore

        # Base arguments for evolution
        base_args = {
            "config": config,
            "train_run_idx": run_number_idx,
            "hall_of_fame": hof_weights,
            "run_hall_of_fame": run_hall_of_fame,
            "game_runner_exe_path_str": str(game_runner_exe_path),
        }

        # Determine evaluation schedule
        if config.evaluation_mode == "hybrid" and config.hybrid_schedule_parsed:
            schedule = config.hybrid_schedule_parsed
        else:
            schedule = [(config.evaluation_mode, 1.0)]

        eval_offset = 0
        gen_offset = 0

        # Process each schedule segment
        for seg_idx, (seg_mode, seg_proportion) in enumerate(schedule):
            if eval_offset >= config.max_evaluations:
                logger.info(f"Run {run_number_idx + 1}, Segment {seg_idx + 1}: Max evaluations reached. Skipping.")
                break

            # Calculate evaluations for this segment
            ideal_evals = math.ceil(config.max_evaluations * seg_proportion)
            remaining_evals = config.max_evaluations - eval_offset
            seg_evals = min(ideal_evals, remaining_evals)

            # Use all remaining evals for final segment
            if seg_idx == len(schedule) - 1:
                seg_evals = remaining_evals

            # Ensure minimum viable evals
            if seg_evals > 0:
                if seg_evals < config.pop_size:
                    seg_evals = config.pop_size
                else:
                    seg_evals = math.ceil(seg_evals / config.pop_size) * config.pop_size
                seg_evals = min(seg_evals, remaining_evals)

            if seg_evals <= 0:
                logger.info(f"Run {run_number_idx + 1}, Segment {seg_idx + 1} ({seg_mode}): No evaluations scheduled.")
                continue

            # Set up segment arguments
            seg_args = base_args.copy()
            seg_args["current_observer_segment_mode"] = seg_mode

            if seg_mode == "fixed":
                seg_args["current_observer_segment_games"] = config.num_games
            elif seg_mode == "coevolution":
                seg_args["current_observer_segment_games"] = config.coevo_games_per_pairing
            else:
                seg_args["current_observer_segment_games"] = "N/A"

            seg_args["run_evaluations_offset"] = eval_offset
            seg_args["run_generations_offset"] = gen_offset
            seg_args["max_evaluations"] = seg_evals

            logger.info(
                f"\nRun {run_number_idx + 1}, Segment {seg_idx + 1}/{len(schedule)}: Mode '{seg_mode}', "
                f"Evals: {seg_evals}, Eval offset: {eval_offset}, Gen offset: {gen_offset}"
            )

            # Select appropriate evaluator
            mp_args = None
            if seg_mode == "fixed":
                evaluator_func = evaluate_fixed_mode_orchestrator
                mp_evaluator = None
            elif seg_mode == "coevolution":
                evaluator_func = evaluate_coevolution_orchestrator
                mp_evaluator = None
            else:
                logger.error(f"Unknown segment mode '{seg_mode}'. Skipping segment.")
                continue

            # Set up terminator and reset counters
            ea.terminator = inspyred.ec.terminators.evaluation_termination
            ea.num_evaluations = 0

            # Run evolution for this segment
            ea.evolve(
                generator=generate_weights,
                evaluator=evaluator_func,
                mp_evaluator=mp_evaluator,
                mp_nprocs=config.num_threads if mp_evaluator else 1,
                mp_args=mp_args,
                pop_size=config.pop_size,
                maximize=True,
                bounder=inspyred.ec.Bounder(0, 1),
                **seg_args,
            )

            # Update offsets for next segment
            evals_in_segment = ea.num_evaluations
            gens_in_segment = ea.num_generations + 1

            eval_offset += evals_in_segment
            gen_offset += gens_in_segment

            if config.debug:
                logger.debug(
                    f"Segment {seg_idx + 1} completed: {evals_in_segment} evals, {gens_in_segment} gens. "
                    f"New offsets - eval: {eval_offset}, gen: {gen_offset}"
                )

            # Update hall of fame
            if ea.population and config.hof_size > 0:
                update_hall_of_fame(ea.population, config, global_hall_of_fame_tuples)
                base_args["hall_of_fame"] = [w for _, w in global_hall_of_fame_tuples]
                hof_weights = [w for _, w in global_hall_of_fame_tuples]

        # Find best individual from final population
        best_individual = None
        if ea.population:
            valid_individuals = [ind for ind in ea.population if ind.fitness is not None]
            if valid_individuals:
                best_individual = max(
                    valid_individuals, key=lambda x: float(x.fitness.values[0] if hasattr(x.fitness, "values") else x.fitness)
                )

        # Log best individual details
        if best_individual:
            best_fitness = float(
                best_individual.fitness.values[0] if hasattr(best_individual.fitness, "values") else best_individual.fitness
            )
            logger.info(f"\nRun {run_number_idx + 1} completed. Best fitness: {best_fitness:.2f}")
        else:
            logger.warning(f"Run {run_number_idx + 1} completed but no best individual found.")

        # Global hall of fame must be a standard list of tuples
        final_global_hof = copy.deepcopy(global_hall_of_fame_tuples)

        return best_individual, final_global_hof


def update_hall_of_fame(population: List[Individual], config: Config, global_hof: List[Tuple[float, List[float]]]) -> None:
    """Update the global hall of fame with individuals from the population."""

    sorted_pop = sorted(
        [ind for ind in population if ind.fitness is not None and ind.candidate is not None],
        key=lambda x: float(x.fitness.values[0] if hasattr(x.fitness, "values") else x.fitness),
        reverse=True,
    )

    for individual in sorted_pop:
        # Extract fitness and weights
        fitness = float(individual.fitness.values[0] if hasattr(individual.fitness, "values") else individual.fitness)
        weights = individual.candidate[: config.num_weights]

        # Check if already in HoF
        if any(np.array_equal(weights, hof_w) for _, hof_w in global_hof):
            continue

        # Add if HoF not full
        if len(global_hof) < config.hof_size:
            global_hof.append((fitness, weights))
            global_hof.sort(key=lambda x: x[0], reverse=True)
        else:
            # Replace worst member if this individual is better
            global_hof.sort(key=lambda x: x[0])
            if fitness > global_hof[0][0]:
                global_hof.pop(0)
                global_hof.append((fitness, weights))
                global_hof.sort(key=lambda x: x[0], reverse=True)
            else:
                global_hof.sort(key=lambda x: x[0], reverse=True)
                break
