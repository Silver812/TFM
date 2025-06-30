# TFM

Repository for my Master's Thesis at the University of Granada, Spain.

Check this for how to install .NET on Ubuntu 22.04: <https://learn.microsoft.com/en-us/dotnet/core/install/linux-ubuntu-install?pivots=os-linux-ubuntu-2204&tabs=dotnet8>
In our case, use the following commands:

Remove the current version of .NET as using the snap package manager is not recommended:

```bash
sudo snap remove dotnet-sdk
```

Install the new SDK .NET 8.0:

```bash
sudo apt update && sudo apt install -y dotnet-sdk-8.0
```

Now the runtime, choose the one that matches the SDK version:

```bash
sudo apt install -y aspnetcore-runtime-8.0
```

## Changelog

- Engine logging system now properly works
- The winning patron combination now is properly stored on a csv format inside GameRunner\bin\Debug\net8.0
- Added a new weight to separate the tavern and card pool calculations
- Simplified Blackmail and Imprisonment scoring logic
- Implemented hate-drafting, if we are in a position where the enemy can hurt us by picking a card from the tavern, we will pick it first
- Got rid of many magic numbers
- Fixed bug related to the number of games in the trainer
- Now the new best weights are automatically saved by the EB_Trainer and loaded by the EvolutionaryBot
- The bot can now be trained against different opponents at random
- The bot can start first or second at random
- Added a logging system to the trainer
- Added a file observer to make datasets of the training process
- Added a new python script to plot the training process
- Added back the posibility to have seeded matches
- Fixed culture weights bug on linux
- Added logging for too large calculations
- Temporal fix for the new deck problems (revisit later)
- Added "train_run" column for managing multiple training runs on the same training session
- Added "avg_weights" column with the average weights of the population for that generation
- Added the parameters to the csv dataset
- Changed all of the plots
- T_AMBUSH exponential 2
- P_AMOUNT exponential 2 from division of 1-x
- Revamped calculation for A_OWN_AMOUNT and A_ENEMY_AMOUNT
- Tentative fix for C_TIER_TAVERN 0 weight due to bad CardTierList
- Changed the arguments for the trainer
- Changed P_AMOUNT to be a multiplication like the rest of the weights

- Implemented training and logs against multiple bots (fixed mode)
  - Bots: PatronFavorsBot, MaxAgentsBot and MaxPrestigeBot
  - num_games is distributed evenly among the bots
  - inspyred.ec.evaluators.parallel_evaluation_mp expects a list of numbers, so to correctly log the matchups per bot a multiprocessing.Manager().dict() was introduced. This dictionary is shared between all the processes
  - A new column (best_indiv_win_breakdown) uses the info in the dictionary

- Implemented coevolution training and logging
  - Custom evaluator that uses round robin for the matchups (every bot plays against every other bot)
  - The coevolution orquestrator uses a new parallelism API called concurrent.futures.ProcessPoolExecutor that is higher level than multiprocessing.Pool. This orquestrator combines the results of the matchups to calculate the final fitness of the bots
  - The new dictionary is also used to create the csv dataset in coevolution mode
  - Adapted the weight sharing on the EB bot to work with multiples environment variables (1 for each bot)

- Implemented hybrid training that uses fixed oponents and coevolution
  - Example (out of 100 generations): fixed:40 --> coevolution:20 --> fixed:40
  - Number of generations are automatically calculated based on the number of generations and the input percentages
  - Uses inspyred evaluator in fixed mode and the custom orquestrator in coevolution mode

- Implemented a hall of fame system: the best individuals from previous train runs are matched against the current best individual of the current run

- Implemented a champion clash for the final weights to be saved into a file: the best individuals of each train run are matched against each other to determine the final champion

- Complete code refactor (from a single file to multiple files)
  - run_evolution.py: main script to manage the evolution at a high level
  - config.py: defaults values and parsing functions for the script
  - simulation.py: connection between the python and the C# gamerunner. Launches all the matches and passes the weights
  - evaluators.py: contains the fixed inspyred evaluator and the custom coevolution orquestrator
  - evolution_logic.py: logic for the training process, launching the inspyred EA algorithm. Manages the hybrid loops and the hall of fame system
  - observers.py: file observer and the logging system
  - benchmark.py: runs champion clashes and saves the final weights
  - utils.py: utility functions

- Changed argparse to Pydantic to use its validation system
- Changed old os.path to pathlib
- Changed prints to logging
- Adjusted the plotting script to work with the new dataset
- Added a bash nohup script to automate the experiment process
- Added an intra hall of fame system to the trainer
- Added powershell scripts to run the experiment on Windows
- Deleted old code and changed old scripts names
- Cleaned up the trainer code
- Added a new script to evaluate all the weights on a single experiment
