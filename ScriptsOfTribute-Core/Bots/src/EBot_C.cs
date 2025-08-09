/*
 * MIT License
 * 
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
/// Enums for the various weights in the evolutionary scoring function.
/// These weights are tuned by an evolutionary algorithm.
/// </summary>
public enum EBW_C
{
	// Agent related weights
	A_HEALTH_REDUCED,               // Bonus for reducing enemy agent health
	A_KILLED,                       // Bonus for killing an enemy agent
	A_OWN_AMOUNT,                   // Bonus for the number of our own agents
	A_ENEMY_AMOUNT,                 // Penalty for the number of enemy agents

	// Card and resource related weights
	CURSE_REMOVED,                  // Penalty (or bonus for enemy) for curses in the deck
	C_TIER_POOL,                    // Weight for card tier pool value
	C_TIER_TAVERN,                  // Weight for card tier tavern value
	C_GOLD_COST,                    // Weight for gold cost of a card
	C_OWN_COMBO,                    // Bonus for our own combo potential (cards from same deck)
	C_ENEMY_COMBO,                  // Bonus for enemy combo potential (to burden the enemy deck)
	COIN_AMOUNT,                    // Weight for coin differences
	POWER_AMOUNT,                   // Weight for power differences
	PRESTIGE_AMOUNT,                // Weight for prestige differences
	H_DRAFT,                        // Weight for denying the enemy a card
	UNSPENT_COIN,                   // Penalty for wasting resources at the end of a turn

	// Tavern specific weights
	T_TITHE,                        // Extra patron activation
	T_BLACK_SACRAMENT,              // Knockout enemy agent
	T_AMBUSH,                       // x2 Knockout enemy agents
	T_BLACKMAIL,                    // gaining power
	T_IMPRISONMENT,                 // x2 gaining power

	// Patron related weight
	P_AMOUNT,                       // Weight for an action that changes a patron favor
	P_COHESION,                     // Bonus for having cards from the same patron
	P_SWING,                        // Bonus for taking an enemy patron, penalty for reusing our own

	// Conditional weights
	W_PRESTIGE_ENDGAME_MULT,        // Multiplier to prestige/power when the game is ending
	W_PATRON_ENDGAME_MULT,          // Multiplier to patron interactions when the game is ending
	W_POWER_PRESTIGE_INTERACTION,   // Relationship between gaining power and the opponent's prestige
}

/// <summary>
/// EBot_C (short of Evolutionary Bot with Coevolution weights) is an AI agent for Scripts of Tribute that uses a parametric, greedy selection approach.
/// For each available move, it simulates the resulting game state and computes a score. Then selects the move with the highest score.
/// </summary>
public sealed class EBot_C : AI
{
	#region Declarations and initialization

	private const int SIM_SEED = 123;
	private const int PREST_ENDGAME_THR = 30;
	private SeededRandom _rng = null!; // Using null forgiving operator to avoid compiler warnings
	private bool _startOfGame;
	private PlayerEnum _myPlayerID;
	private string _patrons = null!;
	private readonly string _logPath = "patronsEBot.csv";
	private int _turnCounter;
	private int _moveCounter;
	private Dictionary<EBW_C, double> _weights = null!;
	private List<Dictionary<EBW_C, double>> _ensembleWeights = new();
	private HashSet<PatronId> _patronCommitment = new();
	private List<(int turnNumber, int moveNumber, string componentName, double value)> _excessiveValues = new();

	/// <summary>
	/// Initializes a new instance of the EBot_C class.
	/// </summary>
	public EBot_C()
	{
		InitBot();
	}

	/// <summary>
	/// Resets the bot's variables.
	/// </summary>
	private void InitBot()
	{
		_rng = new SeededRandom(SIM_SEED);
		_startOfGame = true;
		_turnCounter = 0;
		_moveCounter = 0;
		_weights = new Dictionary<EBW_C, double>();
		_ensembleWeights.Clear();
		_excessiveValues.Clear();
		_patronCommitment.Clear();
	}

	#endregion

	#region Move Simulation and Scoring

	/// <summary>
	/// For each available move, simulate its effect on the game state and select the move with the highest score.
	/// </summary>
	private KeyValuePair<Move, double> GetBestMove(GameState gameState, List<Move> possibleMoves)
	{
		// Determine if we should use the ensemble or a single weight set
		bool useEnsemble = _ensembleWeights.Count > 1;
		var weightsToUse = useEnsemble ? _ensembleWeights : new List<Dictionary<EBW_C, double>> { _weights };

		if (useEnsemble) Log($"Evaluating with ensemble of {weightsToUse.Count} members using Borda Count.");

		// Force playing cards from hand first
		var cardPlayMoves = possibleMoves.Where(m => m.Command == CommandEnum.PLAY_CARD).ToList();

		if (cardPlayMoves.Any())
		{
			possibleMoves = cardPlayMoves;
		}

		var originalWeights = _weights;

		// Precalculate initial state values
		double myInitialCardValue = ComputeCardPoolValue(GetAllCardsForPlayer(gameState.CurrentPlayer), isEnemy: false);
		double enemyInitialCardValue = ComputeCardPoolValue(GetAllCardsForPlayer(gameState.EnemyPlayer), isEnemy: true);
		int myInitialPatronFavors = GetPatronFavorStates(gameState.PatronStates.All).GetValueOrDefault(_myPlayerID);

		// Dictionary for the Borda points for each move
		var moveBordaPoints = new Dictionary<Move, double>();

		foreach (var weightSet in weightsToUse)
		{
			// Use the current weight set for this evaluation round
			_weights = weightSet;
			var scoresForThisSet = new List<KeyValuePair<Move, double>>();

			// Get the score for every possible move using the current weight set
			foreach (var move in possibleMoves)
			{
				double score;

				if (move.Command == CommandEnum.END_TURN)
				{
					score = -(gameState.CurrentPlayer.Coins * _weights[EBW_C.UNSPENT_COIN]);
				}
				else
				{
					try
					{
						var (newState, _) = gameState.ApplyMove(move, SIM_SEED);
						score = ScoreMove(gameState, newState, myInitialCardValue, enemyInitialCardValue, myInitialPatronFavors, move);
					}
					catch (Exception ex)
					{
						Log($"Simulation failed for move {FormatMove(move)}: {ex.Message}");
						score = -10000.0;
					}
				}
				scoresForThisSet.Add(new KeyValuePair<Move, double>(move, score));
			}

			var rankedMoves = scoresForThisSet.OrderByDescending(kv => kv.Value).ToList();

			// Accumulate Borda points based on rank
			int numMoves = rankedMoves.Count;

			for (int i = 0; i < numMoves; i++)
			{
				Move move = rankedMoves[i].Key;
				double points = numMoves - i;

				moveBordaPoints.TryGetValue(move, out double currentPoints);
				moveBordaPoints[move] = currentPoints + points;
			}
		}

		_weights = originalWeights;

		// Fallback
		if (!moveBordaPoints.Any())
		{
			return new KeyValuePair<Move, double>(possibleMoves.First(), 0);
		}

		// Find the move with the highest total Borda points
		var winningMovePair = moveBordaPoints.OrderByDescending(kv => kv.Value).First();

		if (useEnsemble)
		{
			Log($"Ensemble decision: {FormatMove(winningMovePair.Key)} with {winningMovePair.Value} Borda points.");
		}

		return winningMovePair;
	}

	/// <summary>
	/// Computes the overall score of a move by comparing key aspects of the game state before and after the move.
	/// </summary>
	private double ScoreMove(
		GameState before,
		SeededGameState after,
		double myInitialCardValue,
		double enemyInitialCardValue,
		int myInitialPatronFavors,
		Move move
	)
	{
		if (after is null)
			return -100;

		// Condition detection
		var beforePatronStates = GetPatronFavorStates(before.PatronStates.All);
		int enemyInitialPatronFavors = beforePatronStates[before.EnemyPlayer.PlayerID];
		int myPrestige = before.CurrentPlayer.Prestige;
		int enemyPrestige = before.EnemyPlayer.Prestige;
		bool isPlayerInCheck = enemyPrestige >= 40;
		bool isPrestigeEndgame = myPrestige > PREST_ENDGAME_THR || enemyPrestige > PREST_ENDGAME_THR;
		bool isPatronEndgame = myInitialPatronFavors == 3 || enemyInitialPatronFavors == 3;

		// If game is close to end, apply multipliers
		double prestigeMultiplier = 1.0;
		double patronMultiplier = 1.0;

		if (isPrestigeEndgame || isPlayerInCheck)
		{
			prestigeMultiplier = 1.0 + _weights[EBW_C.W_PRESTIGE_ENDGAME_MULT];
		}

		if (isPatronEndgame)
		{
			patronMultiplier = 1.0 + _weights[EBW_C.W_PATRON_ENDGAME_MULT];
		}

		double prestigeWeight = _weights[EBW_C.PRESTIGE_AMOUNT] * prestigeMultiplier;
		double patronWeight = _weights[EBW_C.P_AMOUNT] * patronMultiplier;
		double powerWeight = _weights[EBW_C.POWER_AMOUNT] * prestigeMultiplier;

		// Check for immediate win by patrons
		var afterPatronStates = GetPatronFavorStates(after.PatronStates.All);
		if (afterPatronStates.GetValueOrDefault(_myPlayerID) == 4) return double.MaxValue;
		if (afterPatronStates.GetValueOrDefault(before.EnemyPlayer.PlayerID) == 4) return double.MinValue;

		// Standard score differences
		double patronDiff = (afterPatronStates[_myPlayerID] - myInitialPatronFavors) * patronWeight;
		double powerChange = after.CurrentPlayer.Power - before.CurrentPlayer.Power;
		double myPrestigeDiff = (after.CurrentPlayer.Prestige - myPrestige) * prestigeWeight;
		double enemyPrestigeDiff = (after.EnemyPlayer.Prestige - enemyPrestige) * prestigeWeight;
		double powerScore = (after.CurrentPlayer.Power - before.CurrentPlayer.Power) * powerWeight;

		double interactionScore = 0;
		if (powerChange > 0)
		{
			// Normalize enemyPrestige to prevent score explosion
			double urgencyFactor = enemyPrestige / 40.0;
			interactionScore = powerChange * urgencyFactor * _weights[EBW_C.W_POWER_PRESTIGE_INTERACTION];
		}

		// Remaining scores
		double myCardScore = ComputeCardPoolValue(GetAllCardsForPlayer(after.CurrentPlayer), isEnemy: false) - myInitialCardValue;
		double enemyCardScore = ComputeCardPoolValue(GetAllCardsForPlayer(after.EnemyPlayer), isEnemy: true) - enemyInitialCardValue;
		double coinScore = (after.CurrentPlayer.Coins - before.CurrentPlayer.Coins) * _weights[EBW_C.COIN_AMOUNT];
		double enemyAgentScore = CalculateScoreAgents(before.EnemyPlayer.Agents, after.EnemyPlayer.Agents) * _weights[EBW_C.A_ENEMY_AMOUNT];
		double myAgentScore = CalculateScoreAgents(before.CurrentPlayer.Agents, after.CurrentPlayer.Agents) * _weights[EBW_C.A_OWN_AMOUNT];
		double tavernScore = CalculateTavernScore(before, after);

		double cohesionScore = 0;
		double patronSwingScore = 0;

		if (move is SimpleCardMove scm && scm.Command == CommandEnum.BUY_CARD && scm.Card.Deck != PatronId.TREASURY)
		{
			// Deck Building Logic: Applies only when buying cards
			if (_patronCommitment.Contains(scm.Card.Deck))
			{
				cohesionScore += _weights[EBW_C.P_COHESION];
			}
			else if (_patronCommitment.Count >= 2)
			{
				cohesionScore -= _weights[EBW_C.P_COHESION];
			}
		}
		else if (move is SimplePatronMove spm && spm.PatronId != PatronId.TREASURY)
		{
			// Patron Control Logic: Applies only when using a patron
			var originalOwner = before.PatronStates.All[spm.PatronId];
			if (originalOwner == before.EnemyPlayer.PlayerID)
			{
				// High bonus for stealing a patron from the enemy
				patronSwingScore += _weights[EBW_C.P_SWING];
			}
			else if (originalOwner == _myPlayerID)
			{
				// Small penalty for reusing a patron
				patronSwingScore -= _weights[EBW_C.P_SWING] * 0.5;
			}
		}

		double finalScore =
			+(myPrestigeDiff - enemyPrestigeDiff)
			+ (myCardScore - enemyCardScore)
			+ (myAgentScore - enemyAgentScore)
			+ patronDiff
			+ coinScore
			+ powerScore
			+ tavernScore
			+ interactionScore
			+ cohesionScore
			+ patronSwingScore;

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
			("tavernScore", tavernScore),
			("interactionScore", interactionScore),
			("cohesionScore", cohesionScore),
			("patronSwingScore", patronSwingScore)
		);

		if (hasExcessiveValues)
		{
			Log("[ERROR]: Detected excessive score component values!\n\n\n");
		}

		Log($"Complete move score: Patrons({patronDiff:F2}) Prestige({myPrestigeDiff:F2} - {enemyPrestigeDiff:F2}), " +
						$"Agents({myAgentScore:F2} - {enemyAgentScore:F2}), Coins({coinScore:F2}), Power({powerScore:F2}), " +
						$"Cards({myCardScore:F2} - {enemyCardScore:F2}), Tavern({tavernScore:F2}) + Interaction({interactionScore:F2}) " +
						$"+ Cohesion({cohesionScore:F2}) + PatronSwing({patronSwingScore:F2}) = {finalScore:F2}");

		return finalScore;
	}

	#endregion

	#region Scoring Helpers

	/// <summary>
	/// Returns a list of all cards for the given player.
	/// </summary>
	private List<UniqueCard> GetAllCardsForPlayer(FairSerializedPlayer player)
	{
		var allCards = new List<UniqueCard>();
		allCards.AddRange(player.Hand);
		allCards.AddRange(player.Played);
		allCards.AddRange(player.CooldownPile);
		allCards.AddRange(player.DrawPile);
		allCards.AddRange(player.KnownUpcomingDraws);
		return allCards;
	}

	/// <summary>
	/// Returns a list of all cards for the given player.
	/// </summary>
	private List<UniqueCard> GetAllCardsForPlayer(FairSerializedEnemyPlayer player)
	{
		var allCards = new List<UniqueCard>();
		allCards.AddRange(player.HandAndDraw);
		allCards.AddRange(player.Played);
		allCards.AddRange(player.CooldownPile);
		return allCards;
	}

	/// <summary>
	/// Returns a list of all cards for the given player.
	/// </summary>
	private List<UniqueCard> GetAllCardsForPlayer(SerializedPlayer player)
	{
		var allCards = new List<UniqueCard>();
		allCards.AddRange(player.Hand);
		allCards.AddRange(player.Played);
		allCards.AddRange(player.CooldownPile);
		allCards.AddRange(player.DrawPile);
		allCards.AddRange(player.KnownUpcomingDraws);
		return allCards;
	}

	/// <summary>
	/// Calculates agent related score differences based on health reduction and losses.
	/// </summary>
	private double CalculateScoreAgents(List<SerializedAgent> before, List<SerializedAgent> after)
	{
		double healthReductionScore = 0;
		double killScore = 0;

		// Convert the "after" list to a dictionary for O(1) lookups
		var afterAgentsDict = after.ToDictionary(agent => agent.RepresentingCard.UniqueId);

		foreach (var agentBefore in before)
		{
			if (afterAgentsDict.TryGetValue(agentBefore.RepresentingCard.UniqueId, out var agentAfter))
			{
				// Agent survived
				healthReductionScore += (agentBefore.CurrentHp - agentAfter.CurrentHp) * _weights[EBW_C.A_HEALTH_REDUCED];
			}
			else
			{
				// Agent was killed
				killScore += _weights[EBW_C.A_KILLED];
			}
		}

		return healthReductionScore + killScore + (after.Count - before.Count);
	}


	/// <summary>
	/// Computes the combined value for a set of cards.
	/// Curses are adjusted: for our decks they add a penalty, while for enemy decks they are beneficial.
	/// A bonus is also added for combo potential.
	/// </summary>
	private double ComputeCardPoolValue(IEnumerable<UniqueCard> cards, bool isEnemy)
	{
		double value = 0.0;

		foreach (var card in cards)
		{
			double tierScore = NormalizeTier(CardTierList.GetCardTier(card.Name));
			double baseValue = tierScore * _weights[EBW_C.C_TIER_POOL] - card.Cost * _weights[EBW_C.C_GOLD_COST];


			if (card.Type == CardType.CURSE)
			{
				if (isEnemy)
				{
					// Enemy curse removed is bad
					baseValue -= _weights[EBW_C.CURSE_REMOVED];
				}
				else
				{
					// Our curse removed is good
					baseValue += _weights[EBW_C.CURSE_REMOVED];
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
				double bonus = (count - 1) * (isEnemy ? _weights[EBW_C.C_ENEMY_COMBO] : _weights[EBW_C.C_OWN_COMBO]);
				value += bonus;
			}
		}

		return value;
	}

	/// <summary>
	/// Normalizes the tier value to a range between 0 and 1.
	/// </summary>
	private double NormalizeTier(TierEnum tier)
	{   // TODO: and improvement would be to use the staged card tiers from BestMCTS instead of CardTierList.cs
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

	private double CalculateTavernScore(GameState before, SeededGameState after)
	{
		// Find the card that was actually removed from the tavern, if any
		var removedCard = before.TavernAvailableCards
			.FirstOrDefault(cardBefore => !after.TavernAvailableCards.Any(cardAfter => cardAfter.UniqueId == cardBefore.UniqueId));

		// If no card was removed from the tavern, there is no score change
		if (removedCard is null)
		{
			return 0.0;
		}

		double score = 0.0;
		const double MaxAgentMultiplier = 2.0;
		const int WinPrestigeThreshold = 40;

		// Only need to check the name of the single removedCard
		switch (removedCard.Name)
		{
			case "Tithe":
				int currentPatronCalls = (int)before.CurrentPlayer.PatronCalls;
				if (currentPatronCalls < 2)
				{
					score += _weights[EBW_C.T_TITHE] * (2 - currentPatronCalls);
				}
				if (before.EnemyPlayer.Prestige >= WinPrestigeThreshold - 10)
				{
					score += _weights[EBW_C.T_TITHE] * _weights[EBW_C.H_DRAFT];
				}
				break;

			case "Black Sacrament":
				if (before.EnemyPlayer.Agents.Count > 0)
				{
					score += _weights[EBW_C.T_BLACK_SACRAMENT];
				}
				if (before.CurrentPlayer.Agents.Count > 0)
				{
					score += _weights[EBW_C.T_BLACK_SACRAMENT] * _weights[EBW_C.H_DRAFT];
				}
				break;

			case "Ambush":
				if (before.EnemyPlayer.Agents.Count >= 2)
				{
					score += Math.Pow(_weights[EBW_C.T_AMBUSH] * MaxAgentMultiplier, 2);
				}
				else if (before.EnemyPlayer.Agents.Count == 1)
				{
					score += Math.Pow(_weights[EBW_C.T_AMBUSH], 2);
				}
				if (before.CurrentPlayer.Agents.Count >= 2)
				{
					score += Math.Pow(_weights[EBW_C.T_AMBUSH] * _weights[EBW_C.H_DRAFT] * MaxAgentMultiplier, 2);
				}
				else if (before.CurrentPlayer.Agents.Count == 1)
				{
					score += Math.Pow(_weights[EBW_C.T_AMBUSH] * _weights[EBW_C.H_DRAFT], 2);
				}
				break;

			case "Blackmail":
				if (before.CurrentPlayer.Prestige < WinPrestigeThreshold)
				{
					int prestigeGap = Math.Max(0, WinPrestigeThreshold - before.CurrentPlayer.Prestige);
					score += _weights[EBW_C.T_BLACKMAIL] * (prestigeGap / (double)WinPrestigeThreshold);
				}
				if (before.EnemyPlayer.Prestige >= WinPrestigeThreshold - 15)
				{
					score += _weights[EBW_C.T_BLACKMAIL] * _weights[EBW_C.H_DRAFT];
				}
				break;

			case "Imprisonment":
				if (before.CurrentPlayer.Prestige < WinPrestigeThreshold)
				{
					int prestigeGap = Math.Max(0, WinPrestigeThreshold - before.CurrentPlayer.Prestige);
					score += _weights[EBW_C.T_IMPRISONMENT] * (prestigeGap / (double)WinPrestigeThreshold);
				}
				if (before.EnemyPlayer.Prestige >= WinPrestigeThreshold - 15)
				{
					score += _weights[EBW_C.T_IMPRISONMENT] * _weights[EBW_C.H_DRAFT];
				}
				break;


			default:
				// For any other card taken, it is a small gain because we denied the opponent
				score += NormalizeTier(CardTierList.GetCardTier(removedCard.Name)) * _weights[EBW_C.H_DRAFT];
				break;
		}

		return score;
	}

	#endregion

	#region Utility Methods

	/// <summary>
	/// Returns a dictionary with the number of patron favors for each player (ignoring TREASURY).
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
	/// Checks if any of the scoring components exceeds the specified threshold.
	/// </summary>
	private bool HasExcessiveComponent(double threshold = 50, params (string name, double value)[] components)
	{
		bool foundExcessive = false;

		foreach (var (name, value) in components)
		{
			if (Math.Abs(value) > threshold)
			{
				// Store excessive value for end of game reporting
				_excessiveValues.Add((_turnCounter, _moveCounter, name, value));
				foundExcessive = true;
			}
		}

		return foundExcessive;
	}

	/// <summary>
	/// Formats a move into a human-readable string.
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
	/// The array length must match the number of EBW_C values.
	/// </summary>
	public void SetAgentWeights(double[] w)
	{
		var ebwValues = Enum.GetValues(typeof(EBW_C)).Cast<EBW_C>().ToArray();

		if (w.Length != ebwValues.Length)
		{
			throw new Exception("Incorrect number of weights provided.");
		}

		_weights = ebwValues.Zip(w, (key, value) => new { key, value }).ToDictionary(x => x.key, x => x.value);
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
	/// Sets the bot's weights from different sources.
	/// </summary>
	public void GetEvoWeights(GameState gameState)
	{
		_myPlayerID = gameState.CurrentPlayer.PlayerID;
		_patrons = string.Join(",", gameState.Patrons.Where(x => x != PatronId.TREASURY).Select(n => n.ToString()));

		string weightsEnvVarName = _myPlayerID == PlayerEnum.PLAYER1 ? "EVO_BOT_P1_WEIGHTS" : "EVO_BOT_P2_WEIGHTS";
		string? weightString = Environment.GetEnvironmentVariable(weightsEnvVarName);
		string weightsSource = "None";
		bool weightsLoaded = false;

		if (!string.IsNullOrEmpty(weightString))
		{
			// TRAINING MODE: Load a single weight set from the environment variable
			Log($"PlayerID {_myPlayerID} attempting to load weights from {weightsEnvVarName}.");
			string[] parts = weightString.Split(',');

			if (parts.Length == Enum.GetNames(typeof(EBW_C)).Length)
			{
				try
				{
					double[] weights = parts.Select(s => double.Parse(s.Trim(), CultureInfo.InvariantCulture)).ToArray();
					SetAgentWeights(weights);
					weightsSource = weightsEnvVarName;
					weightsLoaded = true;
				}
				catch (Exception ex)
				{
					Log($"Error parsing weights from {weightsEnvVarName}: {ex.Message}. Falling back.");
				}
			}
			else
			{
				Log($"Expected {Enum.GetNames(typeof(EBW_C)).Length} weights but got {parts.Length} from {weightsEnvVarName}. Falling back.");
			}
		}

		if (!weightsLoaded)
		{
			// COMPETITION MODE: Load the entire ensemble from the file
			string weightsFilePath = GetWeightsFilePath();
			if (File.Exists(weightsFilePath))
			{
				try
				{
					var lines = File.ReadAllLines(weightsFilePath)
									.Where(l => !string.IsNullOrWhiteSpace(l) && !l.StartsWith("#"));

					foreach (var line in lines)
					{
						var parts = line.Trim().Split(',');
						if (parts.Length == Enum.GetNames(typeof(EBW_C)).Length)
						{
							var weights = parts.Select(s => double.Parse(s.Trim(), CultureInfo.InvariantCulture)).ToArray();
							var weightDict = Enum.GetValues(typeof(EBW_C)).Cast<EBW_C>()
												 .Zip(weights, (k, v) => new { k, v })
												 .ToDictionary(x => x.k, x => x.v);
							_ensembleWeights.Add(weightDict);
						}
					}

					if (_ensembleWeights.Any())
					{
						// Set the primary weights to the first one in the file
						_weights = _ensembleWeights.First();
						Log($"Competition mode: Loaded {_ensembleWeights.Count} weight sets into ensemble.");
					}
					else
					{
						// Fallback if file is empty or malformed
						LoadDefaultWeights();
					}
				}
				catch (Exception ex)
				{
					Log($"Error loading ensemble weights file: {ex.Message}");
					LoadDefaultWeights();
				}
			}
			else
			{
				LoadDefaultWeights();
			}
		}
	}


	/// <summary>
	/// Hardcoded set of weights to use when no valid weights are found.
	/// </summary>
	private void LoadDefaultWeights()
	{
		Log("No valid weights found from environment or file. Using hardcoded default weights.");
		double[] defaultWeights = {
			0.00608954,0.00333326,0.00000000,0.99999997,0.00000000,
			0.61937121,0.00000000,0.00000000,0.85716977,0.01766012,
			0.28604759,0.18548796,0.08984469,0.98377612,0.96114306,
			0.74400253,0.56901472,0.49667712,0.00000000,0.64391429,
			0.82871507,0.00097986,0.99703272,0.71650534,0.34196094,
			0.99980960
		};
		SetAgentWeights(defaultWeights);
	}

	#endregion

	#region Interface Methods

	/// <summary>
	/// Called once at the start of the game to select a patron.
	/// Currently selects a random patron.
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
			GetEvoWeights(gameState);
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

		// Patron commitment tracking
		var chosenMove = bestMovePair.Key;
		if (chosenMove is SimpleCardMove scm && scm.Command == CommandEnum.BUY_CARD && scm.Card.Deck != PatronId.TREASURY)
		{
			_patronCommitment.Add(scm.Card.Deck);
		}
		else if (chosenMove is SimplePatronMove spm && spm.PatronId != PatronId.TREASURY)
		{
			_patronCommitment.Add(spm.PatronId);
		}

		return bestMovePair.Key;
	}

	/// <summary>
	/// Called at end of the game for cleanup.
	/// </summary>
	public override void GameEnd(EndGameState state, FullGameState? finalBoardState)
	{
		// Report any excessive values collected during the game
		if (_excessiveValues.Count > 0)
		{
			Log("======= EXCESSIVE SCORE COMPONENT VALUES REPORT =======");
			// Log($"Total excessive values detected: {_excessiveValues.Count}");
			Console.WriteLine($"Total excessive values detected: {_excessiveValues.Count}");

			// Group by turn for better readability
			var byTurn = _excessiveValues
				.GroupBy(x => x.turnNumber)
				.OrderBy(g => g.Key);

			foreach (var turn in byTurn)
			{
				// Log($"Turn {turn.Key}:");
				Console.WriteLine($"Turn {turn.Key}:");

				foreach (var (turnNum, moveNum, component, value) in turn.OrderBy(x => x.moveNumber))
				{
					// Log($"  Move {moveNum}: {component} = {value:F2}");
					Console.WriteLine($"  Move {moveNum}: {component} = {value:F2}");
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
