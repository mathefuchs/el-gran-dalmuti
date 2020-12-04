$env:PYTHONPATH="."
# python ./egd/simulation.py --player0 QLearningAgent --player1 QLearningAgent --player2 QLearningAgent --player3 QLearningAgent --games 25000 --verbose 0 --loadmodel 0 --savemodel 1 --inference 0
# python ./egd/simulation.py --player0 QLearningAgent --player1 Simple --player2 Random --player3 Simple --games 100000 --verbose 0 --loadmodel 0 --savemodel 1 --inference 0
python ./egd/simulation.py --player0 DeepQAgent --player1 Simple --player2 Random --player3 Simple --games 100000 --verbose 0 --loadmodel 0 --savemodel 1 --inference 0
# python -m memory_profiler ./egd/simulation.py --player0 DeepQAgent --player1 Simple --player2 Random --player3 Simple --games 50 --verbose 0 --loadmodel 0 --savemodel 1 --inference 0
# mprof run ./egd/simulation.py --player0 DeepQAgent --player1 Simple --player2 Random --player3 Simple --games 200 --verbose 0 --loadmodel 0 --savemodel 1 --inference 0
