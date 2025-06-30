import re
import numpy as np
import pandas as pd
import seaborn as sns
from pathlib import Path
import matplotlib.pyplot as plt

# Specify a filename like "training_log_20250528_191808_s123_mfixed.csv" to load a specific file
# Leave empty to load the most recent training_log_*.csv file.
TRAINING_DATA = ""

# Styling
plt.style.use("seaborn-v0_8-whitegrid")
palette = sns.color_palette("viridis", 4)
COLORS = {
    "best": palette[0],  # type: ignore
    "average": palette[1],  # type: ignore
    "min": palette[2],  # type: ignore
    "std_dev": palette[3],  # type: ignore
    "background": "#FFFFFF",
    "grid": "#EAEAEA",
}

# Weight names
WEIGHT_NAMES = {
    0: "A_HEALTH_REDUCED",
    1: "A_KILLED",
    2: "A_OWN_AMOUNT",
    3: "A_ENEMY_AMOUNT",
    4: "CURSE_REMOVED",
    5: "C_TIER_POOL",
    6: "C_TIER_TAVERN",
    7: "C_GOLD_COST",
    8: "C_OWN_COMBO",
    9: "C_ENEMY_COMBO",
    10: "COIN_AMOUNT",
    11: "POWER_AMOUNT",
    12: "PRESTIGE_AMOUNT",
    13: "H_DRAFT",
    14: "T_TITHE",
    15: "T_BLACK_SACRAMENT",
    16: "T_AMBUSH",
    17: "T_BLACKMAIL",
    18: "T_IMPRISONMENT",
    19: "P_AMOUNT",
}

NUM_WEIGHTS = len(WEIGHT_NAMES)

# Non weight columns to exclude from weight analysis
EXCLUDE_COLUMNS = ["best_indiv_win_breakdown", "best_fitness", "best_weights", "avg_fitness", "avg_weights"]


# Helper Functions

def parse_weights(weight_str: str) -> list[float]:
    """Parse comma-separated weight strings."""
    
    return [float(w) for w in weight_str.strip('"').split(",")]


def add_parsed_weights(df: pd.DataFrame, source_col: str, prefix: str) -> pd.DataFrame:
    """Parse weight strings and add new columns with prefix."""
    
    parsed = df[source_col].apply(parse_weights)
    weights_array = np.array(parsed.tolist())

    for i, weight_values in enumerate(weights_array.T):
        col_name = f"{prefix}_{WEIGHT_NAMES.get(i, f'weight_{i}')}"
        df[col_name] = weight_values

    return df


def get_weight_columns(df: pd.DataFrame, prefix: str) -> list[str]:
    """Get weight column names, excluding fitness/original columns."""

    candidate_columns = [col for col in df.columns if col.startswith(f"{prefix}_")]
    return [col for col in candidate_columns if col not in EXCLUDE_COLUMNS]


def get_final_generation_data(df: pd.DataFrame) -> pd.DataFrame:
    """Extract data for the last generation of each training run."""
    
    return df.loc[df.groupby("train_run")["generation"].idxmax()]


def get_weight_labels(columns: list[str]) -> list[str]:
    """Generate display labels for weight columns."""
    
    labels = []
    for col in columns:
        if "_" in col:
            weight_name = col.split("_", 1)[1]
            labels.append(weight_name if weight_name in WEIGHT_NAMES.values() else col)
        else:
            labels.append(col)
    return labels


# Data Loading

def load_training_data(csv_path: Path) -> tuple[pd.DataFrame, str]:
    """Loads training data, skips comments, and parses weight columns."""
    
    print(f"Loading data from {csv_path}")

    # Read the file and extract metadata from comments
    with open(csv_path, "r") as f:
        header_comments = []
        for line in f:
            if line.startswith("#"):
                header_comments.append(line.strip())
            else:
                break

    df = pd.read_csv(csv_path, comment="#")
    experiment_id = csv_path.stem

    # Parse weights - let it crash if columns don't exist
    df = add_parsed_weights(df, "best_weights", "best")
    df = add_parsed_weights(df, "avg_weights", "avg")

    return df, experiment_id


# Plotting Functions

def plot_fitness_evolution(df: pd.DataFrame, output_dir: Path, experiment_id: str):
    """Plots the average fitness evolution across multiple training runs using Std Dev band."""
    
    print("Plotting average fitness evolution...")

    fig, ax = plt.subplots(figsize=(12, 7))
    fig.set_facecolor(COLORS["background"])
    ax.set_facecolor(COLORS["background"])

    grouped = df.groupby("generation")
    mean_fitness = grouped["avg_fitness"].mean()
    std_of_avg_fitness = grouped["avg_fitness"].std()
    mean_best_fitness = grouped["best_fitness"].mean()
    mean_min_fitness = grouped["min_fitness"].mean()
    x = mean_fitness.index

    # Plot lines
    ax.plot(x, mean_best_fitness, color=COLORS["best"], linewidth=2.5, label="Mean Best Fitness")
    ax.plot(x, mean_fitness, color=COLORS["average"], linewidth=2.0, label="Mean Average Fitness")
    ax.plot(x, mean_min_fitness, color=COLORS["min"], linewidth=1.5, linestyle="--", label="Mean Min Fitness")
    ax.fill_between(
        x,
        mean_fitness - std_of_avg_fitness,  # type: ignore
        mean_fitness + std_of_avg_fitness,  # type: ignore
        color=COLORS["std_dev"],
        alpha=0.20,
        label="Std Dev (Avg Fitness across runs)",
    )

    # Styling
    num_runs = df["train_run"].nunique()
    ax.set_title(f"Average Fitness Evolution Across {num_runs} Runs", fontsize=16, pad=15)
    ax.set_xlabel("Generation", fontsize=12, labelpad=8)
    ax.set_ylabel("Fitness (Wins)", fontsize=12, labelpad=8)
    ax.grid(True, linestyle="--", alpha=0.7, color=COLORS["grid"])
    ax.legend(fontsize=10, frameon=True, facecolor="white", edgecolor="gray", loc="best")

    # Stats text
    overall_max_best = df["best_fitness"].max()
    final_gen = x.max()
    stats_text = (
        f"Overall Stats ({num_runs} runs):\n"
        f"Max Best Fitness: {overall_max_best:.2f}\n"
        f"Final Gen ({final_gen}) Mean Best: {mean_best_fitness.iloc[-1]:.2f}\n"
        f"Final Gen ({final_gen}) Mean Avg: {mean_fitness.iloc[-1]:.2f}"
    )

    ax.text(
        0.02,
        0.97,
        stats_text,
        transform=ax.transAxes,
        fontsize=9,
        verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.5", facecolor=COLORS["background"], alpha=0.8, edgecolor="gray"),
    )

    # Save
    plt.tight_layout()
    output_file = output_dir / f"fitness_evolution_avg_{experiment_id}.png"
    plt.savefig(output_file, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  -> Saved to {output_file}")


def plot_average_weights_heatmap(df: pd.DataFrame, output_dir: Path, experiment_id: str, weight_type: str = "best"):
    """Plots a heatmap of weight values averaged across runs, evolving over generations."""
    
    print(f"Plotting average '{weight_type}' weights heatmap...")
    
    # Get appropriate weight columns for the specified type
    prefix = "best" if weight_type == "best" else "avg"
    weight_columns = get_weight_columns(df, prefix)

    # Group by generation and calculate mean weights across runs
    grouped = df.groupby("generation")
    mean_weights_over_time = grouped[weight_columns].mean(numeric_only=True)

    # Prepare data for heatmap (transpose for weights as rows)
    heatmap_data = mean_weights_over_time.values.T
    weight_labels = get_weight_labels(weight_columns)

    # Create figure and heatmap
    fig, ax = plt.subplots(figsize=(15, 8))
    fig.set_facecolor(COLORS["background"])
    ax.set_facecolor(COLORS["background"])

    sns.heatmap(
        heatmap_data,
        cmap="viridis",
        ax=ax,
        vmin=0,
        vmax=1,
        cbar_kws={"label": f"Average Weight Value ({weight_type.capitalize()})", "ticks": np.arange(0, 1.1, 0.1)},
        xticklabels=max(1, len(mean_weights_over_time.index) // 10),  # type: ignore
        yticklabels=weight_labels,  # type: ignore
    )

    # Set titles and labels
    ax.set_title(f"Evolution of Average '{weight_type.capitalize()}' Weights Across Runs", fontsize=16, pad=15)
    ax.set_xlabel("Generation", fontsize=12, labelpad=8)
    ax.set_ylabel("Weight Name", fontsize=12, labelpad=8)
    plt.yticks(rotation=0, fontsize=9)
    plt.xticks(rotation=0, ha="center", fontsize=9)

    # Save and close
    plt.tight_layout()
    output_file = output_dir / f"avg_weight_heatmap_{weight_type}_{experiment_id}.png"
    plt.savefig(output_file, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  -> Saved to {output_file}")


def plot_average_final_weights_bar(df: pd.DataFrame, output_dir: Path, experiment_id: str, weight_type: str = "best"):
    """Plots a bar chart of the average weight values in the final generation across all runs."""
    
    print(f"Plotting average final '{weight_type}' weights bar chart...")

    # Get weight data from final generation only
    prefix = "best" if weight_type == "best" else "avg"
    weight_columns = get_weight_columns(df, prefix)
    final_gen_data = get_final_generation_data(df)

    # Calculate average weights across all final generations
    avg_final_weights = final_gen_data[weight_columns].mean(numeric_only=True)
    weight_labels = get_weight_labels(avg_final_weights.index.tolist())

    # Create bar chart with color gradient
    fig, ax = plt.subplots(figsize=(14, 7))
    fig.set_facecolor(COLORS["background"])
    ax.set_facecolor(COLORS["background"])

    bar_colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(avg_final_weights)))  # type: ignore
    bars = ax.bar(weight_labels, avg_final_weights.values.tolist(), color=bar_colors)

    # Set titles, labels and grid
    ax.set_title(f"Average Final Generation '{weight_type.capitalize()}' Weights Across All Runs", fontsize=16, pad=15)
    ax.set_xlabel("Weight Name", fontsize=12, labelpad=8)
    ax.set_ylabel("Average Weight Value", fontsize=12, labelpad=8)
    plt.xticks(rotation=45, ha="right", fontsize=9)
    ax.yaxis.grid(True, linestyle="-", alpha=0.75, color="#BBBBBB")
    ax.xaxis.grid(False)
    ax.set_yticks(np.arange(0, 1.01, 0.05))
    ax.set_ylim(-0.02, 1.05)

    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2.0, height + 0.01, f"{height:.3f}", ha="center", va="bottom", fontsize=8)

    # Save and close
    plt.tight_layout()
    output_file = output_dir / f"avg_final_weights_bar_{weight_type}_{experiment_id}.png"
    plt.savefig(output_file, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  -> Saved to {output_file}")


def plot_final_weights_boxplot(df: pd.DataFrame, output_dir: Path, experiment_id: str, weight_type: str = "best"):
    """Plots box plots for each weight based on the final generation values across all runs."""
    
    print(f"Plotting final '{weight_type}' weights boxplot...")

    # Get final generation data for specified weight type
    prefix = "best" if weight_type == "best" else "avg"
    weight_columns = get_weight_columns(df, prefix)

    final_gen_data = get_final_generation_data(df)
    plot_data_raw = final_gen_data[weight_columns].select_dtypes(include=np.number)

    # Rename columns to readable weight labels
    weight_labels_map = {col: label for col, label in zip(plot_data_raw.columns, get_weight_labels(plot_data_raw.columns.tolist()))}
    plot_data = plot_data_raw.rename(columns=weight_labels_map)

    # Create boxplot
    fig, ax = plt.subplots(figsize=(16, 8))
    fig.set_facecolor(COLORS["background"])
    ax.set_facecolor(COLORS["background"])
    sns.boxplot(data=plot_data, ax=ax, palette="viridis", linewidth=1.5, fliersize=3)

    # Set titles, labels and grid
    ax.set_title(f"Distribution of Final Generation '{weight_type.capitalize()}' Weights Across Runs", fontsize=16, pad=15)
    ax.set_xlabel("Weight Name", fontsize=12, labelpad=8)
    ax.set_ylabel("Weight Value", fontsize=12, labelpad=8)
    plt.xticks(rotation=45, ha="right", fontsize=9)
    ax.yaxis.grid(True, linestyle="-", alpha=0.75, color="#BBBBBB")
    ax.xaxis.grid(False)
    ax.set_yticks(np.arange(0, 1.01, 0.05))
    ax.set_ylim(-0.02, 1.05)

    # Save and close
    plt.tight_layout()
    output_file = output_dir / f"final_weights_boxplot_{weight_type}_{experiment_id}.png"
    plt.savefig(output_file, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  -> Saved to {output_file}")


# Main Execution Logic

def find_latest_csv(training_data_dir: Path) -> Path | None:
    """Finds the most recent training_log_*.csv file based on timestamp in filename."""

    training_files = list(training_data_dir.glob("training_log_*.csv"))

    if not training_files:
        # Fallback to any CSV
        all_csv_files = list(training_data_dir.glob("*.csv"))
        return max(all_csv_files, key=lambda p: p.stat().st_mtime) if all_csv_files else None

    # Extract timestamp from filename and find the latest
    def extract_timestamp(filepath):
        filename = filepath.name
        match = re.search(r"training_log_(\d{8}_\d{6})_", filename)
        return match.group(1) if match else "00000000_000000"

    return max(training_files, key=extract_timestamp)


def main():
    """Main function to process data and generate plots."""
    script_path = Path(__file__).absolute()
    script_dir = script_path.parent.parent
    output_dir = script_dir / "plots"
    training_data_dir = script_dir / "training_data"
    output_dir.mkdir(exist_ok=True)

    if TRAINING_DATA:
        csv_to_process = training_data_dir / TRAINING_DATA
        if not csv_to_process.exists():
            print(f"Specified file not found: {csv_to_process}")
            return
    else:
        csv_to_process = find_latest_csv(training_data_dir)

    if not csv_to_process:
        print(f"No suitable CSV files found in '{training_data_dir}'!")
        return

    df, experiment_id = load_training_data(csv_to_process)

    print(f"\n--- Generating plots for {experiment_id} ---")
    plot_fitness_evolution(df, output_dir, experiment_id)

    for weight_type in ["best", "avg"]:
        plot_average_final_weights_bar(df, output_dir, experiment_id, weight_type)
        plot_average_weights_heatmap(df, output_dir, experiment_id, weight_type)
        plot_final_weights_boxplot(df, output_dir, experiment_id, weight_type)

    print("\n--- Plot generation complete! ---")
    print(f"Plots saved in: {output_dir}")


if __name__ == "__main__":
    main()
