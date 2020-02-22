import tensorflow as tf
tf.compat.v1.disable_eager_execution() # Improves performance by at least an order of magnitude

import numpy as np
from data_utils import *

#Total number of inputs to the network (sensor data of each attribute from each station + nan flags)
NUM_INPUTS = 300
NUM_HID = 1024

# Number of inputs that are actual data and not nan flags.
NON_NAN_FLAG_INPUTS = 156
NAN_FLAG_INPUTS = NUM_INPUTS - NON_NAN_FLAG_INPUTS

MODEL_WEIGHTS_PATH = 'lstm_model_weights.h5'

BATCH_SIZE = 8

# Length to be used when training and input length is not random.
DEFAULT_LENGTH = 384

# First number is the minimum size for random length, second is the maximum. Used for training with different sized inputs.
RAND_LENGTH_RANGE = (128,256)


data,_ = load_all_preprocessed()
train_data = data[:TRAIN_EXAMPLES]
def get_batch_seqs(rand_length=True, batch_size=BATCH_SIZE, default_length=DEFAULT_LENGTH):
    if rand_length:
        length = np.random.randint(RAND_LENGTH_RANGE[0], RAND_LENGTH_RANGE[1])
    else:
        length = default_length

    batch_seqs = []
    starts = np.random.randint(0,TRAIN_EXAMPLES - length - 1, size=batch_size)
    for start in starts:
        batch_seqs.append(train_data[start:start + length + 1])
    return np.asarray(batch_seqs)


'''
    Applies two dense layers to the input. The first outputs "num_act_1" values with "act1" as the activation function,
    the second has a similar output but for "num_act_2" and "act2". The results are then fed into a concatenation layer
    whose output is returned.

    This function is mainly for predicting the nan flags since they are in [0,1] whereas most other attributes are normalized to mean 0 and std 1.

'''
def split_act_func(x,num_act_1,num_act_2,act1,act2):
    outs_1,outs_2 = tf.keras.layers.Dense(num_act_1, activation=act1)(x),\
                    tf.keras.layers.Dense(num_act_2, activation=act2)(x)
    return tf.keras.layers.Concatenate(axis=-1)([outs_1,outs_2])


'''
    Builds the lstm model. If you want to try messing with the model this is the place to do it.
'''
def build_model(num_inputs,num_hid,drop_rate=0.2):
    x = tf.keras.layers.Input([None,num_inputs])

    embedding = tf.keras.layers.Dense(num_hid, activation='relu')(x)

    lstm1 = tf.keras.layers.LSTM(num_hid, return_sequences=True, activation='tanh')(embedding)
    lstm1 = tf.keras.layers.Dropout(drop_rate)(lstm1)

    lstm2 = tf.keras.layers.LSTM(num_hid,activation='tanh')(lstm1)
    lstm2 = tf.keras.layers.Dropout(drop_rate)(lstm2)


    h1 = tf.keras.layers.Dense(num_hid,activation='relu')(lstm2)


    y = split_act_func(h1, num_act_1=NON_NAN_FLAG_INPUTS,
                           num_act_2=NAN_FLAG_INPUTS,
                           act1=None, act2='sigmoid')

    model = tf.keras.models.Model(inputs=x,outputs=y)

    return model


class AirPredictor:
    def __init__(self,num_inputs=NUM_INPUTS,num_hid=NUM_HID,load_weights=True):

        self.num_inputs = num_inputs
        self.num_hid = num_hid

        self.model = build_model(num_inputs,num_hid)
        self.model.compile(tf.keras.optimizers.Adam(),loss='mse')

        if load_weights and os.path.exists(MODEL_WEIGHTS_PATH):
            self.model.load_weights(MODEL_WEIGHTS_PATH)


    def train_step(self,sequences):
        outs = sequences[:,-1,:]
        in_seq = sequences[:,:-1,:]

        self.model.train_on_batch(in_seq,outs)


    def train(self, train_steps, steps_p_save=50, rand_seq_length=True):
        for i in range(train_steps):
            batch_seqs = get_batch_seqs(rand_length=rand_seq_length)

            print("Step: %i of %i (%.3f pct)  length: %i" % (i, train_steps, i / train_steps * 100,batch_seqs.shape[1]))

            self.train_step(batch_seqs)
            if i % steps_p_save == 0:
                self.model.save_weights(MODEL_WEIGHTS_PATH)


    def predict_seq(self,in_seq,length):
        out_seq = []

        for i in range(length):
            next_out = self.model.predict(in_seq)[:,np.newaxis,:]

            in_seq = np.concatenate([in_seq,next_out],axis=1)

            out_seq.append(next_out)

        return np.concatenate(out_seq,axis=1)




if __name__=="__main__":
    test = AirPredictor(load_weights=True)
    test.train(TRAIN_STEPS)



























