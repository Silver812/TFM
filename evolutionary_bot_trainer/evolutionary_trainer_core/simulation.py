import os, sys
import logging
import subprocess
from pathlib import Path
from typing import Tuple, Optional, Dict, List
from .config import Config
from .utils import individual_to_commandline

logger = logging.getLogger(__name__)


def get_game_runner_path(config: Config) -> Optional[Path]:
    """Determine the path to GameRunner executable."""

    script_dir = Path(__file__).resolve().parent  # evolutionary_trainer_core
    project_root = script_dir.parent  # evolutionary_bot_trainer
    workspace_dir = project_root.parent

    # Path to GameRunner executable
    runner_dir = Path("ScriptsOfTribute-Core/GameRunner/bin/Debug/net8.0")
    executable = "GameRunner.exe" if os.name == "nt" else "GameRunner"
    runner_path = workspace_dir / runner_dir / executable

    if config.debug:
        logger.debug(f"Attempting GameRunner path: {runner_path}")

    if not runner_path.exists():
        logger.error(f"GameRunner executable not found at {runner_path}")
        logger.error("Expected structure: workspace/ScriptsOfTribute-Core and workspace/evolutionary_bot_trainer")
        return None

    return runner_path


def simulate_match(
    current_config: Config,
    p1_bot_type: str,
    p2_bot_type: str,
    num_games: int,
    game_runner_exe_path: Path,
    p1_weights: Optional[List[float]] = None,
    p2_weights: Optional[List[float]] = None,
    threads: Optional[int] = None,
) -> Tuple[int, int]:
    """Simulate matches between two bots and return (p1_wins, p2_wins)."""

    if num_games <= 0:
        return 0, 0

    # Create a copy of the current environment to avoid modifying the global state
    match_env = os.environ.copy()

    # Convert weights to strings and add them to our custom environment copy
    if p1_weights:
        match_env["EVO_BOT_P1_WEIGHTS"] = individual_to_commandline(p1_weights, current_config.weight_precision)
    if p2_weights:
        match_env["EVO_BOT_P2_WEIGHTS"] = individual_to_commandline(p2_weights, current_config.weight_precision)
        
    # Keep .NET runtime lightweight per GameRunner process
    match_env.setdefault("DOTNET_gcServer", "0")                     # disable server GC
    match_env.setdefault("COMPlus_gcServer", "0")                    # older/runtime-specific
    match_env.setdefault("DOTNET_ThreadPool_ForceMinThreads", "1")   # small thread pool
    match_env.setdefault("COMPlus_ThreadPool_ForceMinWorkerThreads", "1")

    # Calculate process timeout
    estimated_turns_per_game = 70
    safety_buffer = 120

    if current_config.turn_timeout > 0:
        process_timeout = (current_config.turn_timeout * estimated_turns_per_game * num_games) + safety_buffer
    else:
        process_timeout = (3600 * num_games) + safety_buffer  # 1 hour per game + buffer

    try:
        # Build command
        cmd = [
            str(game_runner_exe_path),
            "--runs",
            str(num_games),
            "--timeout",
            str(current_config.turn_timeout),
            "--enable-logs",
            current_config.match_logs,
            p1_bot_type,
            p2_bot_type,
        ]

        if current_config.seeded_match:
            cmd.extend(["--seed", str(current_config.seed)])

        if threads is not None and threads > 0:
            cmd.extend(["--threads", str(threads)])

        # Log debug info if requested
        if current_config.debug:
            p1_str = match_env.get("EVO_BOT_P1_WEIGHTS")
            p2_str = match_env.get("EVO_BOT_P2_WEIGHTS")

            p1_info = f"{p1_str[:25]}..." if p1_str else "N/A"
            p2_info = f"{p2_str[:25]}..." if p2_str else "N/A"

            logger.debug(f"Simulating: {p1_bot_type} (P1: {p1_info}) vs {p2_bot_type} (P2: {p2_info}) " f"for {num_games} games.")
            logger.debug(f"Command: {' '.join(cmd)}")
            logger.debug(f"Timeout: {process_timeout}s")

        # Run simulation
        process = subprocess.run(
            cmd,
            env=match_env,
            cwd=str(game_runner_exe_path.parent),
            capture_output=True,
            text=True,
            timeout=process_timeout,
            check=False,
        )

        p1_wins = p2_wins = 0

        if current_config.debug:
            print(f"GameRunner Stdout for '{p1_bot_type} vs {p2_bot_type}':\n---\n{process.stdout}\n---")
            print(f"GameRunner Stderr for '{p1_bot_type} vs {p2_bot_type}':\n---\n{process.stderr}\n---")

        if process.returncode != 0:
            logger.warning(f"GameRunner process failed with code {process.returncode}")
            if current_config.debug or current_config.verbose_gamerunner:
                logger.warning(f"Command: {' '.join(cmd)}")
                logger.warning(f"Stderr: {process.stderr[:500].strip()}...")
        else:
            output = process.stdout

            if current_config.verbose_gamerunner:
                logger.info(f"GameRunner Output ({p1_bot_type} vs {p2_bot_type}):\n{output[:1000]}...")

            # Parse wins
            for line in output.splitlines():
                line_lower = line.lower()
                if "final amount of p1 win" in line_lower:
                    try:
                        p1_wins = int(line.split(":")[1].strip().split("/")[0])
                    except Exception as e:
                        logger.error(f"Error parsing P1 wins from '{line}': {e}")
                elif "final amount of p2 win" in line_lower:
                    try:
                        p2_wins = int(line.split(":")[1].strip().split("/")[0])
                    except Exception as e:
                        logger.error(f"Error parsing P2 wins from '{line}': {e}")

            # Infer P2 wins if not found directly (fallback)
            if p1_wins > 0 and p2_wins == 0 and "final amount of p2 win" not in output.lower():
                if p1_wins <= num_games:
                    p2_wins = num_games - p1_wins
                else:
                    logger.error(f"P1 wins ({p1_wins}) exceeds total games ({num_games}).")
                    p2_wins = 0

        if current_config.debug:
            logger.debug(f"Result: {p1_bot_type} won {p1_wins}/{num_games} vs {p2_bot_type} (P2 wins: {p2_wins})")

        return p1_wins, p2_wins

    except subprocess.TimeoutExpired:
        logger.warning(f"Match '{p1_bot_type} vs {p2_bot_type}' timed out after {process_timeout}s.")
        return 0, 0
    except Exception as e:
        logger.error(f"Error during match simulation: {e}")
        if current_config.debug:
            logger.exception("Traceback:")
        return 0, 0
