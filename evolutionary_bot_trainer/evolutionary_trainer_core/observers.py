from operator import itemgetter
import numpy as np
import csv
import time
import logging
from pathlib import Path
from multiprocessing.managers import ListProxy
from typing import List, Dict, Any, Tuple, Optional, Union


from .config import Config
from .evaluators import run_single_coevo_battle_task
from .utils import individual_to_commandline, Individual

logger = logging.getLogger(__name__)


class LoggingObserver:
    """Observer that logs evolutionary algorithm progress to console and CSV."""

    def __init__(self, config: Config, csv_path: Optional[Path], csv_fields: List[str], train_run: int, run_hall_of_fame: ListProxy):
        self.config = config
        self.csv_path = csv_path
        self.csv_fields = csv_fields
        self.train_run = train_run
        self.run_start_time = time.time()
        self.last_gen_time = time.time()
        self.__name__ = self.__class__.__name__
        self.run_hall_of_fame = run_hall_of_fame
        self.details_from_last_eval = {}

    def __call__(self, population: List[Individual], num_generations: int, num_evaluations: int, args: Dict[str, Any]):
        """Called by evolutionary algorithm to log progress."""

        # Initialization
        current_time = time.time()
        gen_duration = current_time - self.last_gen_time
        self.last_gen_time = current_time

        eval_offset = args.get("run_evaluations_offset", 0)
        gen_offset = args.get("run_generations_offset", 0)
        global_evals = eval_offset + num_evaluations
        global_gen = gen_offset + num_generations

        details_store = self.details_from_last_eval

        # Get population and best individual
        valid_pop = [p for p in population if p.fitness is not None and p.candidate is not None]

        if not valid_pop:
            logger.warning(f"Run {self.train_run + 1} Gen {global_gen}: No valid individuals.")
            if details_store:
                details_store.clear()
            return

        # Define best_indiv once, early, to be used throughout the function
        best_indiv = max(valid_pop, key=lambda p: p.fitness)
        fitness_values = [p.fitness for p in valid_pop]
        best_fitness = best_indiv.fitness
        worst_fitness = min(fitness_values) if fitness_values else 0.0
        mean_fitness = np.mean(fitness_values) if fitness_values else 0.0
        median_fitness = np.median(fitness_values) if fitness_values else 0.0
        std_fitness = np.std(fitness_values) if fitness_values else 0.0

        # Logging
        if logging.getLogger().getEffectiveLevel() <= logging.INFO:
            print("\n" + "Generation Evaluation   Time (s)      Worst       Best     Median    Average    Std Dev")
            print("---------- ---------- ---------- ---------- ---------- ---------- ---------- ----------")
            print(
                f"{global_gen: >10} {global_evals: >10} "
                f"{gen_duration: >9.2f}s "
                f"{worst_fitness: >10.2f} {best_fitness: >10.2f} "
                f"{median_fitness: >10.2f} {mean_fitness: >10.2f} {std_fitness: >10.2f}"
            )

        # Hall of Fame logic
        if self.config.intra_run_hof_size > 0:
            # Bounded key for uniqueness checks
            bounded_weights = [max(0, min(w, 1)) for w in best_indiv.candidate]
            champion_key = individual_to_commandline(bounded_weights, self.config.weight_precision)
            is_in_hof = any(champion_key == hof_entry.get("id") for hof_entry in self.run_hall_of_fame)

            if not is_in_hof:
                hof_entry = {"id": champion_key, "weights": best_indiv.candidate, "defeats": 0}

                if len(self.run_hall_of_fame) < self.config.intra_run_hof_size:
                    self.run_hall_of_fame.append(hof_entry)
                    logger.info(f"Added new champion. Size: {len(self.run_hall_of_fame)}")
                else:

                    # Find weakest member to challenge
                    weakest_member = max(self.run_hall_of_fame, key=itemgetter("defeats"))
                    weakest_member_index = self.run_hall_of_fame.index(weakest_member)

                    # Match against weakest member
                    game_runner_path = Path(args["game_runner_exe_path_str"])

                    _, wins, _, _, _, played, _, _ = run_single_coevo_battle_task(
                        p1_id="champion",
                        p1_weights=best_indiv.candidate,
                        p2_id="weakest_hof",
                        p2_weights=weakest_member["weights"],
                        num_games=self.config.intra_run_hof_num_games,
                        config=self.config,
                        game_runner_path=game_runner_path,
                        battle_type="hof_challenge",
                    )

                    # Replace
                    if played > 0 and (wins / played) > 0.5:
                        logger.info(f"New champion WON the challenge ({wins}/{played}). Replacing weakest member.")
                        self.run_hall_of_fame[weakest_member_index] = hof_entry
                    else:
                        logger.info(f"New champion LOST the challenge ({wins}/{played}). HoF remains unchanged.")

            # Pruning logic
            perform_pruning = (
                self.config.hof_pruning_frequency_gens > 0 and global_gen > 0 and global_gen % self.config.hof_pruning_frequency_gens == 0
            )
            if perform_pruning and len(self.run_hall_of_fame) > 1:
                num_to_prune = int(len(self.run_hall_of_fame) * self.config.hof_pruning_percentage)
                if num_to_prune > 0:
                    self.run_hall_of_fame.sort(key=itemgetter("defeats"), reverse=True)
                    for _ in range(num_to_prune):
                        self.run_hall_of_fame.pop(0)
                    logger.info(f"Pruning {num_to_prune} weakest members. New size: {len(self.run_hall_of_fame)}")

        if self.config.debug and best_indiv.candidate:
            weights_str = individual_to_commandline(best_indiv.candidate, self.config.weight_precision)
            logger.debug(f"Best weights (Run {self.train_run + 1}, Gen {global_gen}): {weights_str[:70]}...")

        if self.config.training_logs and self.csv_path:
            segment_mode = args.get("current_observer_segment_mode", "N/A")
            segment_games = args.get("current_observer_segment_games", "N/A")
            self.log_to_csv(global_gen, global_evals, best_indiv, fitness_values, details_store, valid_pop, segment_mode, segment_games)

        if details_store:
            self.details_from_last_eval.clear()

    def log_to_csv(
        self,
        global_gen: int,
        global_evals: int,
        best_indiv: Optional[Individual],
        fitness_values: List[float],
        details_store: Optional[Dict[Tuple[float, ...], Dict[str, Dict[str, int]]]],
        valid_pop: List[Individual],
        segment_mode: str,
        segment_games: Union[int, str],
    ):
        """Write current generation data to CSV file."""

        if not self.csv_path:
            logger.error("CSV path is None. Cannot write CSV.")
            return

        # Initialize default values
        best_fitness = 0.0
        best_weights_str = ""
        win_breakdown = "N/A"

        # Process best individual data
        if best_indiv and best_indiv.candidate:
            # Extract best weights
            best_weights = best_indiv.candidate[: self.config.num_weights]
            best_weights_str = individual_to_commandline(best_weights, self.config.weight_precision)

            # Extract fitness
            best_fitness = float(best_indiv.fitness.values[0] if hasattr(best_indiv.fitness, "values") else best_indiv.fitness)

            # Get win breakdown for fixed mode
            if segment_mode == "fixed" and details_store:
                candidate_key = tuple(best_weights)
                details = details_store.get(candidate_key)

                if details:
                    parts = []
                    for opponent, stats in sorted(details.items(), key=lambda item: str(item[0])):
                        if not opponent.startswith("("):
                            parts.append(f"{opponent}:{stats.get('wins',0)}/{stats.get('played',0)}")

                    win_breakdown = ",".join(parts) if parts else "No fixed opponent details"
                else:
                    win_breakdown = f"Details not found for best individual"
            elif segment_mode == "coevolution":
                win_breakdown = "N/A (Coevolution)"

        # Calculate statistics
        avg_fitness = np.mean(fitness_values) if fitness_values else 0.0
        median_fitness = np.median(fitness_values) if fitness_values else 0.0
        min_fitness = min(fitness_values) if fitness_values else 0.0
        std_fitness = np.std(fitness_values) if fitness_values else 0.0

        # Calculate average weights
        avg_weights_str = ""
        if valid_pop:
            valid_weights = []
            for indiv in valid_pop:
                if indiv.candidate and len(indiv.candidate) >= self.config.num_weights:
                    valid_weights.append(indiv.candidate[: self.config.num_weights])

            if valid_weights:
                try:
                    mean_w = np.mean(valid_weights, axis=0).tolist()
                    avg_weights_str = individual_to_commandline(mean_w, self.config.weight_precision)
                except Exception as e:
                    logger.error(f"Error calculating average weights: {e}")
                    avg_weights_str = "Error"

        # Generate matches info string
        matches_info = ""
        if segment_mode == "fixed":
            matches_info = f"FixedSegBaseTotal:{segment_games}"
        elif segment_mode == "coevolution":
            matches_info = f"CoevoSegBasePeerPair:{segment_games}"
        else:
            matches_info = f"Mode({segment_mode}):BaseGames({segment_games})"

        if self.config.hof_size > 0 and self.config.hof_num_games > 0:
            matches_info += f";HoFBasePair:{self.config.hof_num_games}"

        # Create CSV data
        data = {
            "train_run": self.train_run + 1,
            "generation": global_gen,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "total_time_sec": round(time.time() - self.run_start_time, 2),
            "num_matches_info": matches_info,
            "evaluations": global_evals,
            "best_fitness": round(best_fitness, 3),
            "avg_fitness": round(avg_fitness, 3),
            "median_fitness": round(median_fitness, 3),
            "min_fitness": round(min_fitness, 3),
            "std_fitness": round(std_fitness, 3),
            "best_weights": best_weights_str,
            "avg_weights": avg_weights_str,
            "best_indiv_win_breakdown": win_breakdown,
        }

        # Remove fields not in fieldnames
        if "avg_weights" not in self.csv_fields and "avg_weights" in data:
            del data["avg_weights"]
        elif "avg_weights" in self.csv_fields and "avg_weights" not in data:
            data["avg_weights"] = ""

        # Write to CSV
        try:
            with open(self.csv_path, "a", newline="") as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=self.csv_fields)
                writer.writerow(data)
        except IOError as e:
            logger.error(f"Could not write to CSV file {self.csv_path}: {e}")
        except Exception as e:
            logger.error(f"Error writing CSV log for generation {global_gen}: {e}")
