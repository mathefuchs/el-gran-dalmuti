$env:PYTHONPATH="."
python ./egd/simulation.py --player0 QLearningAgent --player1 QLearningAgent --player2 QLearningAgent --player3 QLearningAgent --games 25000 --verbose 0 --loadmodel 0 --savemodel 1 --inference 0
