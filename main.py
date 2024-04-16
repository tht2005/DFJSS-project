from keras.layers import Input, Dense, Activation
from keras.models import Sequential, load_model, clone_model
from keras.optimizers import Adam
import numpy as np

import random
from collections import deque

from Env import env

NEPISODE = 10

REPLAY_MEM = 2000
MINIBATCH_SIZE = 32
GAMMA = 0.95
ALPHA = 0.00025
MU = 1.8
C = 100


def build_model():
    model = Sequential([
        Input(shape=(4,)),
        Dense(30, activation="relu"),
        Dense(30, activation="relu"),
        Dense(30, activation="relu"),
        Dense(4, activation="linear")
    ])
    model.compile(
        optimizer=Adam(learning_rate=ALPHA),
        loss='mse',
        metrics=['accuracy']
    )
    # show off
    model.summary()
    return model

def clone_(model):
    clone_one = clone_model(model)
    clone_one.build((None, 4))
    clone_one.compile(
        optimizer=Adam(learning_rate=ALPHA),
        loss='mse',
        metrics=['accuracy']
    )
    clone_one.set_weights(model.get_weights())
    return clone_one

def chooseAction(e_soft, theta, state):
    if np.random.uniform(low=0, high=1) < e_soft:
        return np.random.randint(low=0, high=4)
    values = theta(np.array([state]))
    return np.argmax(values)

D = deque(maxlen=REPLAY_MEM)
theta = build_model()
theta2 = clone_(theta)

e = env()

for cur_ep in range(NEPISODE):
    e.initEpisode()

    NUM_OP = sum(e.h)
    DECAY_VALUE = np.power(NUM_OP, MU)

    total_reward = 0
    done = False
    step = 0

    while not done:
        step += 1

        state = e.getState()
        action = chooseAction(max(0.1, 1 - step / DECAY_VALUE), theta, state)
        done = e.makeAction(action)

        next_state = e.getState()
        reward = state[3] - next_state[3] # P_a(t) - P_a(t + 1)

        total_reward += reward
        print(step, '/', NUM_OP, '=>', total_reward)

        D.append((state, action, reward, next_state))

        minibatch = random.sample(D, min(len(D), MINIBATCH_SIZE))

        inputs = []
        predicts = []

        for state, action, reward, next_state in minibatch:
            Y = reward
            input_ = np.array([state])
            predict = theta(input_)

            if not done:
                opt_action = np.argmax(predict)
                Y += GAMMA * theta2(input_)[0][opt_action]

            target = np.array(predict[0])
            target[action] = Y

            inputs.append(state)
            predicts.append(target)

        theta.fit(np.array(inputs), np.array(predicts), epochs=1, verbose=1)

        if step % C == 0:
            theta2 = clone_(theta)

