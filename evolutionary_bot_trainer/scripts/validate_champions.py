import sys
import re
import itertools
import argparse
from pathlib import Path
import concurrent.futures

# Add project root to path to allow imports from evolutionary_trainer_core
project_root_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root_dir))

try:
    from evolutionary_trainer_core.simulation import get_game_runner_path, simulate_match
    from evolutionary_trainer_core.config import Config
except ImportError as e:
    print(f"Error: Could not import necessary modules. Make sure you are running the script from the 'scripts' directory.", file=sys.stderr)
    print(f"Details: {e}", file=sys.stderr)
    sys.exit(1)

LOG_TO_VALIDATE = ""
MODE = "coevo"
MATCHES = 5000
THREADS = 8


def find_latest_log_file(log_dir: Path) -> Path | None:
    """Finds the most recent .log file in the specified directory."""
    log_files = list(log_dir.glob("*.log"))
    if not log_files:
        return None
    return max(log_files, key=lambda p: p.stat().st_mtime)


def parse_weights_from_log(log_path: Path) -> list[str]:
    """Extracts all 'Best Weights' strings from a log file."""
    if not log_path.exists():
        print(f"Error: Log file not found at {log_path}", file=sys.stderr)
        return []

    print(f"\nParsing weights from: {log_path.name}")
    content = log_path.read_text()
    weights_pattern = re.compile(r"(?<=Best Weights \(from benchmark\):\n)(.*)")
    found_weights = [line.strip() for line in weights_pattern.findall(content)]
    print(f"Found {len(found_weights)} champion weight sets to validate.")
    return found_weights


def run_fixed_benchmark(all_weights: list, args: argparse.Namespace, sim_config: Config, game_runner_path: Path):
    """Runs benchmark against a fixed set of opponents."""
    results = []
    for i, weights_str in enumerate(all_weights):
        print(f"\n--- Benchmarking Champion {i+1}/{len(all_weights)} vs Fixed Opponents ---")
        champion_results = {"id": i + 1, "weights": weights_str, "total_wins": 0, "total_matches": 0, "breakdown": {}}
        try:
            weights_floats = [float(w) for w in weights_str.split(",")]
        except ValueError:
            print(f"Error: Could not parse weights for champion {i+1}. Skipping.")
            continue

        for opponent in args.opponents:
            print(f"  vs. {opponent} ({args.matches} matches)...", end="", flush=True)
            matches_as_p1 = args.matches // 2
            matches_as_p2 = args.matches - matches_as_p1
            total_wins_vs_opponent = 0

            if matches_as_p1 > 0:
                bot_wins_as_p1, _ = simulate_match(
                    sim_config, "EvolutionaryBot", opponent, matches_as_p1, game_runner_path, p1_weights=weights_floats, threads=args.threads
                )
                total_wins_vs_opponent += bot_wins_as_p1

            if matches_as_p2 > 0:
                _, bot_wins_as_p2 = simulate_match(
                    sim_config, opponent, "EvolutionaryBot", matches_as_p2, game_runner_path, p2_weights=weights_floats, threads=args.threads
                )
                total_wins_vs_opponent += bot_wins_as_p2

            print(f" Wins: {total_wins_vs_opponent}/{args.matches}")
            champion_results["total_wins"] += total_wins_vs_opponent
            champion_results["total_matches"] += args.matches
            if opponent in champion_results["breakdown"]:
                champion_results["breakdown"][opponent] += total_wins_vs_opponent
            else:
                champion_results["breakdown"][opponent] = total_wins_vs_opponent
        results.append(champion_results)

    print_fixed_summary(results, args)


def run_coevo_benchmark(all_weights: list, args: argparse.Namespace, sim_config: Config, game_runner_path: Path):
    """Runs a round-robin tournament between all found champions."""
    champions = {f"C{i+1}": [float(w) for w in weights_str.split(",")] for i, weights_str in enumerate(all_weights)}
    matchups = list(itertools.combinations(champions.keys(), 2))

    print(f"\n--- Starting Co-evolution Tournament: {len(champions)} champions, {len(matchups)} matchups ---")

    results_map = {champ_id: {} for champ_id in champions}

    with concurrent.futures.ProcessPoolExecutor(max_workers=args.threads) as executor:
        future_to_matchup = {}
        for p1_id, p2_id in matchups:
            print(f"  Scheduling: {p1_id} vs {p2_id} ({args.matches} matches)")
            future = executor.submit(
                simulate_match,
                sim_config,
                "EvolutionaryBot",
                "EvolutionaryBot",
                args.matches,
                game_runner_path,
                p1_weights=champions[p1_id],
                p2_weights=champions[p2_id],
            )
            future_to_matchup[future] = (p1_id, p2_id)

        for future in concurrent.futures.as_completed(future_to_matchup):
            p1_id, p2_id = future_to_matchup[future]
            try:
                p1_wins, p2_wins = future.result()
                results_map[p1_id][p2_id] = p1_wins
                results_map[p2_id][p1_id] = p2_wins
            except Exception as exc:
                print(f"\nMatchup {p1_id} vs {p2_id} generated an exception: {exc}")
                results_map[p1_id][p2_id] = -1
                results_map[p2_id][p1_id] = -1

    print_coevo_summary(results_map, args.matches)


def print_fixed_summary(results: list, args: argparse.Namespace):
    """Prints the summary for the fixed opponent benchmark."""
    print("\n" + "=" * 35)
    print("--- Fixed Benchmark Summary ---")
    print("=" * 35)

    if not results:
        print("No results to display.")
        return

    results.sort(key=lambda r: r["total_wins"] / r["total_matches"] if r["total_matches"] > 0 else 0, reverse=True)

    for result in results:
        win_rate = (result["total_wins"] / result["total_matches"]) * 100 if result["total_matches"] > 0 else 0
        print(f"\nChampion #{result['id']} - Overall Win Rate: {win_rate:.2f}% ({result['total_wins']}/{result['total_matches']})")
        for opponent, wins in result["breakdown"].items():
            print(f"  - vs {opponent:<20}: {wins}/{args.matches}")

    print("\n" + "=" * 35)
    print(f"Best Performing Champion: #{results[0]['id']}")
    print(f"Best Weights: {results[0]['weights']}")
    print("=" * 35 + "\n")


def print_coevo_summary(results_map: dict, matches: int):
    """Prints a summary matrix of the co-evolution tournament."""
    print("\n" + "=" * 60)
    print("--- Co-evolution Tournament Summary ---")
    print("=" * 60)

    if not results_map:
        print("No results to display.")
        return

    champ_ids = sorted(results_map.keys())
    header = f"{'vs':>5}" + "".join([f"{cid:>8}" for cid in champ_ids])
    print(header)
    print("-" * len(header))

    overall_wins = {cid: 0 for cid in champ_ids}

    for p1_id in champ_ids:
        row_str = f"{p1_id:<5}"
        for p2_id in champ_ids:
            if p1_id == p2_id:
                row_str += f"{'---':>8}"
                continue

            wins = results_map.get(p1_id, {}).get(p2_id, -1)
            if wins >= 0:
                win_rate = (wins / matches) * 100
                row_str += f"{win_rate:>7.1f}%"
                overall_wins[p1_id] += wins
            else:
                row_str += f"{'ERROR':>8}"
        print(row_str)

    print("-" * len(header))
    print("\n--- Overall Win Rates vs. Other Champions ---")

    total_matches = (len(champ_ids) - 1) * matches
    leaderboard = []
    for champ_id, total_wins in overall_wins.items():
        win_rate = (total_wins / total_matches) * 100 if total_matches > 0 else 0
        leaderboard.append((champ_id, win_rate, total_wins))

    leaderboard.sort(key=lambda x: x[1], reverse=True)

    print(f"{'Rank':<5} {'Champion':<10} {'Win Rate':<15} {'Total Wins':<15}")
    print("-" * 45)
    for i, (champ_id, win_rate, total_wins) in enumerate(leaderboard):
        print(f"{i+1:<5} {champ_id:<10} {win_rate:>7.2f}%         ({total_wins}/{total_matches})")

    print("\n" + "=" * 60)
    print(f"Tournament Winner: {leaderboard[0][0]}")
    print("=" * 60 + "\n")


def main():
    """Main function to parse args and dispatch to the correct benchmark mode."""
    parser = argparse.ArgumentParser(description="Validate champion weights from training logs.")
    parser.add_argument("log_file", nargs="?", default=None, help="Path to the training log file. If omitted, the latest log will be used.")
    parser.add_argument(
        "--mode",
        choices=["fixed", "coevo"],
        default=MODE,
        help="Choose validation mode: 'fixed' opponents or 'coevo' round-robin tournament.",
    )
    parser.add_argument("--matches", type=int, default=MATCHES, help="Number of matches to run per pairing.")
    parser.add_argument("--threads", type=int, default=THREADS, help="Number of CPU cores to use (for coevo mode).")
    parser.add_argument(
        "--opponents",
        nargs="+",
        default=["PatronFavorsBot", "MaxAgentsBot", "MaxPrestigeBot", "DecisionTreeBot"],
        help="List of opponent bot names (for fixed mode).",
    )
    args = parser.parse_args()

    log_dir = project_root_dir / "training_logs"
    log_to_parse = None

    if LOG_TO_VALIDATE:
        log_to_parse = log_dir / LOG_TO_VALIDATE
    elif args.log_file:
        log_to_parse = Path(args.log_file)
    else:
        print("No specific log file provided, finding the most recent...")
        log_to_parse = find_latest_log_file(log_dir)

    if not log_to_parse or not log_to_parse.exists():
        print(f"Error: No valid log file found or the specified file does not exist: {log_to_parse}", file=sys.stderr)
        return

    all_weights = parse_weights_from_log(log_to_parse)
    if not all_weights:
        return

    game_runner_path = get_game_runner_path(Config())  # type: ignore
    if not game_runner_path:
        print("Error: GameRunner executable not found. Cannot proceed.", file=sys.stderr)
        return

    sim_config = Config(turn_timeout=2, match_logs="NONE", debug=False)  # type: ignore

    if args.mode == "fixed":
        run_fixed_benchmark(all_weights, args, sim_config, game_runner_path)
    elif args.mode == "coevo":
        run_coevo_benchmark(all_weights, args, sim_config, game_runner_path)


if __name__ == "__main__":
    main()
