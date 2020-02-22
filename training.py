from lstm_model import *

# Just a big number since the weights are saved and the training can be interrupted without losing much progress.
TRAIN_STEPS = 10000000


test = AirPredictor(load_weights=True)
test.train(TRAIN_STEPS)

















