# Set Env Vars
export PYTHONPATH="."

# Q-Table
# python ./egd/main.py --player0 QLearningAgent --player1 QLearningAgent --player2 QLearningAgent --player3 QLearningAgent --games 25000 --verbose 0 --loadmodel 0 --savemodel 1 --inference 0
# python ./egd/main.py --player0 QLearningAgent --player1 Simple --player2 Random --player3 Simple --games 100000 --verbose 0 --loadmodel 0 --savemodel 1 --inference 0

# Deep Q-Agent
# python ./egd/main.py --player0 DeepQAgent --player1 Simple --player2 Random --player3 Simple --games 100000 --verbose 0 --loadmodel 0 --savemodel 1 --inference 0
# python -u ./egd/main.py --player0 DeepQAgent --player1 Simple --player2 Random --player3 Random --games 100000 --verbose 0 --loadmodel 1 --savemodel 1 --inference 0 | Tee-Object -file train_log.txt
python ./egd/main.py --player0 DeepQAgent --player1 Random --player2 Random --player3 Random --games 100000 --verbose 0 --loadmodel 0 --savemodel 1 --inference 0

# Profiling
# python -m memory_profiler ./egd/main.py --player0 DeepQAgent --player1 Simple --player2 Random --player3 Simple --games 50 --verbose 0 --loadmodel 0 --savemodel 1 --inference 0
# mprof run ./egd/main.py --player0 DeepQAgent --player1 Simple --player2 Random --player3 Simple --games 200 --verbose 0 --loadmodel 0 --savemodel 1 --inference 0
# python -m cProfile -s time -o "cprof.profile" ./egd/main.py --player0 DeepQAgent --player1 Random --player2 Random --player3 Random --games 100 --verbose 0 --loadmodel 0 --savemodel 1 --inference 0

# Record games
python ./egd/main.py --player0 Simple --player1 Simple --player2 Random --player3 Random --verbose 0 --loadmodel 0 --savemodel 0 --inference 0 --savehistories 1 --games 250000 --process_idx_offset 0
python ./egd/main.py --player0 Simple --player1 Simple --player2 Random --player3 Random --verbose 0 --loadmodel 0 --savemodel 0 --inference 0 --savehistories 1 --games 250000 --process_idx_offset 25
python ./egd/main.py --player0 Simple --player1 Simple --player2 Random --player3 Random --verbose 0 --loadmodel 0 --savemodel 0 --inference 0 --savehistories 1 --games 250000 --process_idx_offset 50
python ./egd/main.py --player0 Simple --player1 Simple --player2 Random --player3 Random --verbose 0 --loadmodel 0 --savemodel 0 --inference 0 --savehistories 1 --games 250000 --process_idx_offset 75
# python ./egd/main.py --player0 Simple --player1 Random --player2 Simple --player3 Simple --games 1 --verbose 1 --loadmodel 0 --savemodel 0 --inference 0 --savehistories 1
