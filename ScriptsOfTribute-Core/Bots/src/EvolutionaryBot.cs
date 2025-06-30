/*
 * MIT License
 * Copyright (c) 2025 David Castejón
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

using System.Globalization;

using ScriptsOfTribute;
using ScriptsOfTribute.AI;
using ScriptsOfTribute.Board;
using ScriptsOfTribute.Serializers;
using ScriptsOfTribute.Board.Cards;

namespace Bots;

/// <summary>
/// Enumeration for the various weights in the evolutionary scoring function.
/// These evolutionary bot weights are tuned by an evolutionary algorithm
/// </summary>
public enum EBW
{
	// Agent related weights
	A_HEALTH_REDUCED,
	A_KILLED,
	A_OWN_AMOUNT,
	A_ENEMY_AMOUNT,

	// Card pool and curse related weights
	CURSE_REMOVED,        // Penalty (or bonus for enemy) for curses in the deck
	C_TIER_POOL,          // Weight for card tier pool value
	C_TIER_TAVERN,        // Weight for card tier tavern value
	C_GOLD_COST,          // Weight for gold cost of a card
	C_OWN_COMBO,          // Bonus for our own combo potential (cards from same deck)
	C_ENEMY_COMBO,        // Bonus for enemy combo potential (to burden the enemy deck)
	COIN_AMOUNT,          // Weight for coin differences
	POWER_AMOUNT,         // Weight for power differences
	PRESTIGE_AMOUNT,      // Weight for prestige differences
	H_DRAFT,              // Weight for denying the enemy a card

	// Tavern specific card weights
	T_TITHE,              // Extra patron activation
	T_BLACK_SACRAMENT,    // Knockout enemy agent
	T_AMBUSH,             // x2 Knockout enemy agents
	T_BLACKMAIL,          // gaining power
	T_IMPRISONMENT,       // x2 gaining power

	// Patron related weights
	P_AMOUNT              // Weight for an action that changes a patron favor
}

/// <summary>
/// EvolutionaryBot is an AI agent for Scripts of Tribute that uses a parametric, greedy simulation approach.
/// For each available move, it simulates the resulting game state and computes a score. Then greedily selects the move with the highest score.
/// </summary>
public sealed class EvolutionaryBot : AI
{
	#region Class Variables

	/// <summary>
	/// Constant simulation seed used to ensure deterministic move simulations
	/// </summary>
	private const int SimulationSeed = 123;
	private SeededRandom _rng = null!; // Using null forgiving operator to avoid compiler warnings
	private bool _startOfGame;
	private PlayerEnum _myPlayerID;
	private string _patrons = null!;
	private readonly string _logPath = "patronsEBot.csv";
	private int _turnCounter;
	private int _moveCounter;
	private Dictionary<EBW, double> _weights = null!;

	// To check for too big values in the scoring function
	private List<(int turnNumber, int moveNumber, string componentName, double value)> _excessiveValues = new();

	#endregion

	#region Constructor and Initialization

	/// <summary>
	/// Initializes a new instance of the <see cref="EvolutionaryBot"/> class
	/// </summary>
	public EvolutionaryBot()
	{
		InitBot();
	}

	/// <summary>
	/// Gets the path to the weights file.
	/// </summary>
	private string GetWeightsFilePath()
	{
		// Try current directory first
		string path = "eb_best_weights.txt";

		if (File.Exists(path))
			return path;

		// Try executable directory next
		string? exePath = Path.GetDirectoryName(System.Reflection.Assembly.GetExecutingAssembly().Location);
		if (exePath != null)
		{
			path = Path.Combine(exePath, "eb_best_weights.txt");
			if (File.Exists(path))
				return path;
		}

		return "eb_best_weights.txt"; // Default fallback
	}

	/// <summary>
	/// Resets the bot's variables and sets default weights
	/// </summary>
	private void InitBot()
	{
		_rng = new SeededRandom(SimulationSeed);
		_startOfGame = true;
		_turnCounter = 0;
		_moveCounter = 0;
		_weights = new Dictionary<EBW, double>();
		string weightsFilePath = GetWeightsFilePath();
		_excessiveValues.Clear();

		// Attempt to load from the general environment variable (used for fixed opponent mode)
		string? defaultSingleEvoWeightsEnv = Environment.GetEnvironmentVariable("EVOLUTIONARY_BOT_WEIGHTS");
		string? weightStringToParse = null;
		string source = "None";

		if (!string.IsNullOrEmpty(defaultSingleEvoWeightsEnv))
		{
			weightStringToParse = defaultSingleEvoWeightsEnv;
			source = "EVOLUTIONARY_BOT_WEIGHTS (InitBot)";
		}

		if (!string.IsNullOrEmpty(weightStringToParse))
		{
			Log($"InitBot: Attempting to load weights from {source}.");

			string[] parts = weightStringToParse.Split(',');

			if (parts.Length == Enum.GetNames(typeof(EBW)).Length)
			{
				try
				{
					double[] weights = parts.Select(s => double.Parse(s.Trim(), CultureInfo.InvariantCulture)).ToArray();
					SetAgentWeights(weights);

					Log($"InitBot: Weights loaded from {source}.");
				}
				catch (Exception ex)
				{
					Log($"InitBot: Error parsing weights from EVOLUTIONARY_BOT_WEIGHTS: {ex.Message}");
				}
			}
			else
			{
				Log($"Expected {Enum.GetNames(typeof(EBW)).Length} weights but got {parts.Length} in EVOLUTIONARY_BOT_WEIGHTS");
			}
		}
		else
		{
			Log("InitBot: EVOLUTIONARY_BOT_WEIGHTS not set. Will attempt P1/P2 specific or file/default in Play() or later InitBot stages.");

			// Read weights from the weights file
			if (File.Exists(weightsFilePath))
			{
				try
				{
					string fileContents = File.ReadAllText(weightsFilePath);

					// Extract just the first line with weights (ignore comments)
					string weightsLine = fileContents.Split('\n')[0].Trim();
					string[] parts = weightsLine.Split(',');

					if (parts.Length == Enum.GetNames(typeof(EBW)).Length)
					{
						double[] weights = parts.Select(s => double.Parse(s.Trim(), CultureInfo.InvariantCulture)).ToArray();
						SetAgentWeights(weights);
						Log("Weights loaded from file eb_best_weights.txt");
					}
					else
					{
						Log($"Expected {Enum.GetNames(typeof(EBW)).Length} weights but got {parts.Length} in file");
					}
				}
				catch (Exception ex)
				{
					Log($"Error reading weights file: {ex.Message}");
				}
			}
			else
			{
				Log("Default weights set (weights file not found or invalid)");
				_weights = Enum.GetValues(typeof(EBW)).Cast<EBW>().ToDictionary(ebw => ebw, ebw => 1.0);
			}
		}
	}

	#endregion

	#region Move Simulation and Scoring

	/// <summary>
	/// For each available move, simulate its effect on the game state and select the move with the highest score
	/// </summary>
	private KeyValuePair<Move, double> GetBestMove(GameState gameState, List<Move> possibleMoves)
	{
		double bestScore = double.MinValue;

		// Start with a random move in case no simulation gives a better score
		Move bestMove = possibleMoves.PickRandom(_rng);

		foreach (var move in possibleMoves)
		{
			Log($"--> POSSIBLE {FormatMove(move)}");
			double score = 0;

			if (move.Command == CommandEnum.END_TURN)
			{
				score = -1;
			}
			else
			{
				try
				{
					// Simulate the move using a fixed seed for deterministic behavior
					var (newState, newMoves) = gameState.ApplyMove(move, SimulationSeed);
					score = ScoreMove(gameState, newState);
				}
				catch (Exception ex)
				{
					// If simulation fails, log the error and give a slight positive score
					Log($"Simulation failed for move {FormatMove(move)}: {ex.Message}");
					score = 1;
				}
			}

			if (score >= bestScore)
			{
				bestScore = score;
				bestMove = move;
			}
		}
		return new KeyValuePair<Move, double>(bestMove, bestScore);
	}

	/// <summary>
	/// Checks if any of the scoring components exceeds the specified threshold
	/// </summary>
	/// <param name="threshold">The threshold value to check against (default 50)</param>
	/// <param name="components">Dictionary of component names and their values</param>
	/// <returns>True if any component exceeds the threshold, otherwise false</returns>
	private bool HasExcessiveComponent(double threshold = 50, params (string name, double value)[] components)
	{
		bool foundExcessive = false;

		foreach (var (name, value) in components)
		{
			if (Math.Abs(value) > threshold)
			{
				// Store excessive value for end-of-game reporting instead of logging immediately
				_excessiveValues.Add((_turnCounter, _moveCounter, name, value));
				foundExcessive = true;
			}
		}

		return foundExcessive;
	}

	/// <summary>
	/// Computes the overall score of a move by comparing key aspects of the game state before and after the move
	/// </summary>
	private double ScoreMove(GameState before, SeededGameState after)
	{
		// Simulation failed
		if (after is null)
		{
			return -100;
		}

		// Check patron favor: if any player reaches 4 favors (excluding TREASURY), the game is effectively over
		Dictionary<PlayerEnum, int> patronFavorStates = GetPatronFavorStates(after.PatronStates.All);
		if (patronFavorStates[PlayerEnum.PLAYER1] == 4 || patronFavorStates[PlayerEnum.PLAYER2] == 4)
		{
			bool iWon = patronFavorStates[_myPlayerID] == 4;
			Log(iWon ? "WINNING MOVE (4 Patrons)" : "LOSING MOVE (4 Patrons)");
			return iWon ? double.MaxValue : double.MinValue;
		}

		// Calculate patron differences
		double statesDiff = GetPatronFavorStates(after.PatronStates.All)[_myPlayerID] - GetPatronFavorStates(before.PatronStates.All)[_myPlayerID];
		double patronDiff = statesDiff * _weights[EBW.P_AMOUNT];

		// Calculate differences in prestige
		double enemyPrestigeDiff = CalculateEnemyScorePrestige(before.EnemyPlayer, after.EnemyPlayer);
		double myPrestigeDiff = CalculateMyScorePrestige(before.CurrentPlayer, after.CurrentPlayer);

		// Calculate differences in coins and power
		int coinDifference = after.CurrentPlayer.Coins - before.CurrentPlayer.Coins;
		double coinScore = coinDifference * _weights[EBW.COIN_AMOUNT];

		int powerDifference = after.CurrentPlayer.Power - before.CurrentPlayer.Power;
		double powerScore = powerDifference * _weights[EBW.POWER_AMOUNT];

		// Compute agent-related score differences
		double enemyAgentScore = CalculateScoreAgents(before.EnemyPlayer.Agents, after.EnemyPlayer.Agents) * _weights[EBW.A_ENEMY_AMOUNT];
		double myAgentScore = CalculateScoreAgents(before.CurrentPlayer.Agents, after.CurrentPlayer.Agents) * _weights[EBW.A_OWN_AMOUNT];

		// Compute the change in card pool value for both players
		double enemyCardScore = CalculateEnemyScoreCards(before.EnemyPlayer, after.EnemyPlayer);
		double myCardScore = CalculateMyScoreCards(before.CurrentPlayer, after.CurrentPlayer);

		// Compute the special score from Tavern cards
		double tavernScore = CalculateTavernScore(before, after);

		// Combine all components to get the final move score
		double finalScore = patronDiff + (enemyPrestigeDiff - myPrestigeDiff) + (enemyAgentScore - myAgentScore)
			+ coinScore + powerScore + (enemyCardScore - myCardScore) + tavernScore;

		// Check for excessive values in the scoring components
		bool hasExcessiveValues = HasExcessiveComponent(50,
			("patronDiff", patronDiff),
			("enemyPrestigeDiff", enemyPrestigeDiff),
			("myPrestigeDiff", myPrestigeDiff),
			("enemyAgentScore", enemyAgentScore),
			("myAgentScore", myAgentScore),
			("coinScore", coinScore),
			("powerScore", powerScore),
			("enemyCardScore", enemyCardScore),
			("myCardScore", myCardScore),
			("tavernScore", tavernScore)
		);

		if (hasExcessiveValues)
		{
			Log("[ERROR]: Detected excessive score component values!\n\n\n");
		}

		Log($"Complete move score: Patrons({patronDiff:F2}) Prestige({enemyPrestigeDiff:F2} - {myPrestigeDiff:F2}), " +
						$"Agents({enemyAgentScore:F2} - {myAgentScore:F2}), Coins({coinScore:F2}), Power({powerScore:F2}), " +
						$"Cards({enemyCardScore:F2} - {myCardScore:F2}), Tavern({tavernScore:F2}) = {finalScore:F2}");

		return finalScore;
	}

	#endregion

	#region Scoring Helpers

	/// <summary>
	/// Returns a dictionary with the number of patron favors for each player (ignoring TREASURY)
	/// </summary>
	private Dictionary<PlayerEnum, int> GetPatronFavorStates(Dictionary<PatronId, PlayerEnum> patrons)
	{
		var favorStates = new Dictionary<PlayerEnum, int>
		{
			{ PlayerEnum.PLAYER1, 0 },
			{ PlayerEnum.PLAYER2, 0 },
			{ PlayerEnum.NO_PLAYER_SELECTED, 0 }
		};

		foreach (var entry in patrons)
		{
			if (entry.Key == PatronId.TREASURY)
				continue;
			favorStates[entry.Value]++;
		}

		return favorStates;
	}

	/// <summary>
	/// Calculates the change in prestige for the current player
	/// </summary>
	private double CalculateMyScorePrestige(FairSerializedPlayer before, SerializedPlayer after) =>
		(after.Prestige - before.Prestige) * _weights[EBW.PRESTIGE_AMOUNT];

	/// <summary>
	/// Calculates the change in prestige for the enemy player
	/// </summary>
	private double CalculateEnemyScorePrestige(FairSerializedEnemyPlayer before, SerializedPlayer after) =>
		(after.Prestige - before.Prestige) * _weights[EBW.PRESTIGE_AMOUNT];

	/// <summary>
	/// Calculates agent-related score differences based on health reduction and losses
	/// </summary>
	private double CalculateScoreAgents(List<SerializedAgent> before, List<SerializedAgent> after)
	{
		double healthReductionScore = 0;
		double killScore = 0;

		foreach (var agentBefore in before)
		{
			// Check if the agent survives after the move
			bool survived = after.Any(agentAfter => agentAfter.RepresentingCard.UniqueId == agentBefore.RepresentingCard.UniqueId);

			if (survived)
			{
				var agentAfter = after.First(a => a.RepresentingCard.UniqueId == agentBefore.RepresentingCard.UniqueId);
				healthReductionScore += (agentBefore.CurrentHp - agentAfter.CurrentHp) * _weights[EBW.A_HEALTH_REDUCED];
			}
			else
			{
				// If the agent is lost, add a kill penalty
				killScore += _weights[EBW.A_KILLED];
			}
		}

		// double ownCountScore = after.Count * _weights[EBW.A_OWN_AMOUNT];
		// double enemyCountScore = before.Count * _weights[EBW.A_ENEMY_AMOUNT];

		return healthReductionScore + killScore + after.Count - before.Count;
	}


	/// <summary>
	/// Computes the combined value for a set of cards
	/// Curses are adjusted: for our decks they add a penalty, while for enemy decks they are beneficial
	/// A bonus is added for combo potential (cards from the same deck), ignoring TREASURY
	/// </summary>
	private double ComputeCardPoolValue(IEnumerable<UniqueCard> cards, bool isEnemy)
	{
		double value = 0.0;

		foreach (var card in cards)
		{
			double tierValue = (int)CardTierList.GetCardTier(card.Name);
			double baseValue = tierValue * _weights[EBW.C_TIER_POOL] + card.Cost * _weights[EBW.C_GOLD_COST];

			if (card.Type == CardType.CURSE)
			{
				if (isEnemy)
				{
					// Enemy curse removed is bad
					baseValue -= _weights[EBW.CURSE_REMOVED];
				}
				else
				{
					// Our curse removed is good
					baseValue += _weights[EBW.CURSE_REMOVED];
				}
			}

			value += baseValue;
		}

		// Group cards by their deck to check combo potential
		var groups = cards.Where(c => c.Deck != PatronId.TREASURY).GroupBy(c => c.Deck);

		foreach (var group in groups)
		{
			int count = group.Count();

			if (count >= 2)
			{
				// Bonus is applied per additional card in the same deck
				double bonus = (count - 1) * (isEnemy ? _weights[EBW.C_ENEMY_COMBO] : _weights[EBW.C_OWN_COMBO]);
				value += bonus;
			}
		}

		return value;
	}

	/// <summary>
	/// Calculates the change in our card pool value (after - before) across all piles
	/// </summary>
	private double CalculateMyScoreCards(FairSerializedPlayer before, SerializedPlayer after)
	{
		var allBefore = before.Hand
			.Concat(before.Played)
			.Concat(before.CooldownPile)
			.Concat(before.DrawPile)
			.Concat(before.KnownUpcomingDraws);

		var allAfter = after.Hand
			.Concat(after.Played)
			.Concat(after.CooldownPile)
			.Concat(after.DrawPile)
			.Concat(after.KnownUpcomingDraws);

		return ComputeCardPoolValue(allAfter, isEnemy: false) - ComputeCardPoolValue(allBefore, isEnemy: false);
	}

	/// <summary>
	/// Calculates the change in the enemy card pool value (after - before)
	/// </summary>
	private double CalculateEnemyScoreCards(FairSerializedEnemyPlayer before, SerializedPlayer after)
	{
		var allBefore = before.HandAndDraw
			.Concat(before.Played)
			.Concat(before.CooldownPile);

		var allAfter = after.Hand
			.Concat(after.Played)
			.Concat(after.CooldownPile)
			.Concat(after.DrawPile);

		return ComputeCardPoolValue(allAfter, isEnemy: true) - ComputeCardPoolValue(allBefore, isEnemy: true);
	}

	private double NormalizeTier(TierEnum tier)
	{
		return tier switch
		{
			TierEnum.S => 1.0,
			TierEnum.A => 0.75,
			TierEnum.B => 0.5,
			TierEnum.C => 0.25,
			TierEnum.D => 0.1,
			_ => 0.0,
		};
	}

	/// <summary>
	/// Calculates a heuristic score for the Tavern cards based on their strategic value
	/// Special cards like Tithe, Black Sacrament, Ambush, Blackmail, and Imprisonment are scored specifically
	/// Other cards incur a penalty based on their tier
	/// </summary>
	private double CalculateTavernScore(GameState before, SeededGameState after)
	{
		double score = 0.0;
		// Move constants to class level
		const double MaxAgentMultiplier = 2.0;
		const int WinPrestigeThreshold = 40;

		int currentPatronCalls = (int)before.CurrentPlayer.PatronCalls;
		int enemyAgentCount = before.EnemyPlayer.Agents.Count;
		int myAgentCount = before.CurrentPlayer.Agents.Count;
		int myPrestige = before.CurrentPlayer.Prestige;
		int enemyPrestige = before.EnemyPlayer.Prestige;

		foreach (var card in before.TavernAvailableCards)
		{
			bool isRemoved = !after.TavernAvailableCards.Any(c => c.Name == card.Name);

			switch (card.Name)
			{
				case "Tithe":
					// Offensive bonus: If we have less than 2 patron calls and remove the card
					if (currentPatronCalls < 2 && isRemoved)
					{
						score += _weights[EBW.T_TITHE] * (2 - currentPatronCalls);
					}

					// Defensive bonus: If opponent could use Tithe effectively against us
					if (isRemoved && enemyPrestige >= WinPrestigeThreshold - 10) // Enemy close to winning
					{
						// Enemy getting an extra patron call would be bad for us
						score += _weights[EBW.T_TITHE] * _weights[EBW.H_DRAFT];
					}
					break;

				case "Black Sacrament":
					// Offensive bonus: If enemy has agents and we remove the card
					if (enemyAgentCount > 0 && isRemoved)
					{
						score += _weights[EBW.T_BLACK_SACRAMENT];
					}

					// Defensive bonus: If we have agents (enemy could use against us)
					if (isRemoved && myAgentCount > 0)
					{
						score += _weights[EBW.T_BLACK_SACRAMENT] * _weights[EBW.H_DRAFT];
					}
					break;

				case "Ambush":
					// Offensive bonus: Based on enemy agent count
					if (enemyAgentCount >= 2 && isRemoved)
					{
						score += Math.Pow(_weights[EBW.T_AMBUSH] * MaxAgentMultiplier, 2);
					}
					else if (enemyAgentCount == 1 && isRemoved)
					{
						score += Math.Pow(_weights[EBW.T_AMBUSH], 2);
					}

					// Defensive bonus: Based on our agent count (what enemy could kill)
					if (isRemoved && myAgentCount >= 2)
					{
						score += Math.Pow(_weights[EBW.T_AMBUSH] * _weights[EBW.H_DRAFT] * MaxAgentMultiplier, 2);
					}
					else if (isRemoved && myAgentCount == 1)
					{
						score += Math.Pow(_weights[EBW.T_AMBUSH] * _weights[EBW.H_DRAFT], 2);
					}
					break;

				case "Blackmail":
					// Offensive bonus: If our prestige is below threshold
					if (isRemoved && myPrestige < WinPrestigeThreshold)
					{
						int prestigeGap = Math.Max(0, WinPrestigeThreshold - myPrestige);
						score += _weights[EBW.T_BLACKMAIL] * (prestigeGap / (double)WinPrestigeThreshold);
					}

					// Defensive bonus: If enemy is close to winning by prestige
					if (isRemoved && enemyPrestige >= WinPrestigeThreshold - 15)
					{
						// Enemy getting more power/prestige would be bad for us
						score += _weights[EBW.T_BLACKMAIL] * _weights[EBW.H_DRAFT];
					}
					break;

				case "Imprisonment":
					// Offensive bonus: If our prestige is below threshold
					if (isRemoved && myPrestige < WinPrestigeThreshold)
					{
						int prestigeGap = Math.Max(0, WinPrestigeThreshold - myPrestige);
						score += _weights[EBW.T_IMPRISONMENT] * (prestigeGap / (double)WinPrestigeThreshold);
					}

					// Defensive bonus: If enemy is close to winning by prestige
					if (isRemoved && enemyPrestige >= WinPrestigeThreshold - 15)
					{
						// Enemy getting more power/prestige would be bad for us
						score += _weights[EBW.T_IMPRISONMENT] * _weights[EBW.H_DRAFT];
					}
					break;

				default:
					// For any other Tavern card, subtract a small penalty based on its tier
					// score -= (int)CardTierList.GetCardTier(card.Name) * _weights[EBW.C_TIER_TAVERN];
					score -= NormalizeTier(CardTierList.GetCardTier(card.Name)) * _weights[EBW.C_TIER_TAVERN];
					break;
			}
		}

		return score;
	}

	#endregion

	#region Utility Methods

	/// <summary>
	/// Formats a move into a human-readable string
	/// </summary>
	private string FormatMove(Move move) =>
		move switch
		{
			SimpleCardMove scm => $"[CardMove: {scm.Command} {scm.Card}]",
			SimplePatronMove spm => $"[PatronMove: {spm}]",
			MakeChoiceMove<UniqueCard> mcc => $"[ChoiceMove (Card): {mcc.Command} {mcc}]",
			MakeChoiceMove<UniqueEffect> mce => $"[ChoiceMove (Effect): {mce.Command} {mce}]",
			_ => $"[Move: {move}]"
		};

	/// <summary>
	/// Sets the weights for the evolutionary scoring function from an array.
	/// The array length must match the number of EBW values.
	/// </summary>
	public void SetAgentWeights(double[] w)
	{
		if (w.Length != Enum.GetNames(typeof(EBW)).Length)
		{
			throw new Exception("Incorrect number of weights provided.");
		}

		_weights[EBW.A_HEALTH_REDUCED] = w[0];
		_weights[EBW.A_KILLED] = w[1];
		_weights[EBW.A_OWN_AMOUNT] = w[2];
		_weights[EBW.A_ENEMY_AMOUNT] = w[3];
		_weights[EBW.CURSE_REMOVED] = w[4];
		_weights[EBW.C_TIER_POOL] = w[5];
		_weights[EBW.C_TIER_TAVERN] = w[6];
		_weights[EBW.C_GOLD_COST] = w[7];
		_weights[EBW.C_OWN_COMBO] = w[8];
		_weights[EBW.C_ENEMY_COMBO] = w[9];
		_weights[EBW.COIN_AMOUNT] = w[10];
		_weights[EBW.POWER_AMOUNT] = w[11];
		_weights[EBW.PRESTIGE_AMOUNT] = w[12];
		_weights[EBW.H_DRAFT] = w[13];
		_weights[EBW.T_TITHE] = w[14];
		_weights[EBW.T_BLACK_SACRAMENT] = w[15];
		_weights[EBW.T_AMBUSH] = w[16];
		_weights[EBW.T_BLACKMAIL] = w[17];
		_weights[EBW.T_IMPRISONMENT] = w[18];
		_weights[EBW.P_AMOUNT] = w[19];
	}

	#endregion

	#region AI Interface Methods

	/// <summary>
	/// Called once at the start of the game to select a patron.
	/// Currently selects a random patron (excluding TREASURY).
	/// </summary>
	public override PatronId SelectPatron(List<PatronId> availablePatrons, int round)
	{
		// TODO: improve patron selection based on the best winrates using the winning patron combinations log
		return availablePatrons.PickRandom(_rng);
	}

	/// <summary>
	/// Called repeatedly to choose a move until an END_TURN move is returned.
	/// </summary>
	public override Move Play(GameState gameState, List<Move> possibleMoves, TimeSpan remainingTime)
	{
		if (_startOfGame)
		{
			_myPlayerID = gameState.CurrentPlayer.PlayerID;
			_patrons = string.Join(",", gameState.Patrons.Where(x => x != PatronId.TREASURY).Select(n => n.ToString()));

			string? p1WeightsEnv = Environment.GetEnvironmentVariable("EVO_BOT_P1_WEIGHTS");
			string? p2WeightsEnv = Environment.GetEnvironmentVariable("EVO_BOT_P2_WEIGHTS");
			string? weightStringToParseForPlay = null;
			string determinedSourceForPlay = "None";

			bool p1VarSet = !string.IsNullOrEmpty(p1WeightsEnv);
			bool p2VarSet = !string.IsNullOrEmpty(p2WeightsEnv);

			// Load P1 weights if the variable is set and the bot is Player 1
			if (p1VarSet && _myPlayerID == PlayerEnum.PLAYER1)
			{
				weightStringToParseForPlay = p1WeightsEnv;
				determinedSourceForPlay = "EVO_BOT_P1_WEIGHTS (Play)";
			}
			// Load P2 weights if the variable is set and the bot is Player 2
			else if (p2VarSet && _myPlayerID == PlayerEnum.PLAYER2)
			{
				weightStringToParseForPlay = p2WeightsEnv;
				determinedSourceForPlay = "EVO_BOT_P2_WEIGHTS (Play)";
			}

			if (!string.IsNullOrEmpty(weightStringToParseForPlay))
			{
				Log($"Play(): PlayerID {_myPlayerID} attempting to load weights from {determinedSourceForPlay}.");
				string[] parts = weightStringToParseForPlay.Split(',');
				if (parts.Length == Enum.GetNames(typeof(EBW)).Length)
				{
					try
					{
						double[] weights = parts.Select(s => double.Parse(s.Trim(), CultureInfo.InvariantCulture)).ToArray();
						SetAgentWeights(weights);
						Log($"Play(): Weights finalized for PlayerID {_myPlayerID} from {determinedSourceForPlay}.");
					}
					catch (Exception ex)
					{
						Log($"Play(): Error parsing from {determinedSourceForPlay} for PlayerID {_myPlayerID}. Using InitBot/Fallback weights: {ex.Message}.");
					}
				}
			}

			_startOfGame = false;
		}

		Log($"Current Turn: {_turnCounter}");
		var bestMovePair = GetBestMove(gameState, possibleMoves);

		if (bestMovePair.Key.Command == CommandEnum.END_TURN)
		{
			_moveCounter = 0;
			_turnCounter++;
			Log("------------------------------------------------------");
		}
		else
		{
			Log($"Selected move: {FormatMove(bestMovePair.Key)} with score {bestMovePair.Value:F2}");
			_moveCounter++;
		}

		return bestMovePair.Key;
	}

	/// <summary>
	/// Called at game end for cleanup. Also logs the patron sequence if the bot wins.
	/// </summary>
	public override void GameEnd(EndGameState state, FullGameState? finalBoardState)
	{
		// Report any excessive values collected during the game
		if (_excessiveValues.Count > 0)
		{
			Log("======= EXCESSIVE SCORE COMPONENT VALUES REPORT =======");
			Log($"Total excessive values detected: {_excessiveValues.Count}");

			// Group by turn for better readability
			var byTurn = _excessiveValues
				.GroupBy(x => x.turnNumber)
				.OrderBy(g => g.Key);

			foreach (var turn in byTurn)
			{
				Log($"Turn {turn.Key}:");
				// Console.WriteLine($"Turn {turn.Key}:");

				foreach (var (turnNum, moveNum, component, value) in turn.OrderBy(x => x.moveNumber))
				{
					Log($"  Move {moveNum}: {component} = {value:F2}");
					// Console.WriteLine($"  Move {moveNum}: {component} = {value:F2}");
				}
			}

			Log("=====================================================");
		}

		//? Commented out for performance reasons
		// if (state.Winner == _myID)
		// {
		// 	string logEntry = $"{DateTime.Now:yyyy-MM-dd,HH:mm:ss},{_patrons}{Environment.NewLine}";
		// 	File.AppendAllText(_logPath, logEntry);
		// 	Log($"Winning patron combination logged to: {_logPath}");
		// }

		InitBot();
	}

	#endregion
}
