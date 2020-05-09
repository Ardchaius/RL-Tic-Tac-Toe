import numpy as np
import environment as env
import tensorflow as tf
from tensorflow import keras
import copy


def softmax(x):
    max_x = np.max(x)
    exp_x = np.exp(x - max_x)
    return exp_x / np.sum(exp_x)


class ttt_agent:
    model = 0
    opponent_model = 0
    num_episodes = 0
    environment = env.ttt_environment()
    previous_action = 0
    previous_state = 0
    previous_action_values = 0
    reward = 0
    terminal = 0
    discount = 0
    learning_rate = 0

    def __init__(self, hidden_layer_neurons, num_episodes, discount, learning_rate):
        self.num_episodes = num_episodes
        self.discount = discount
        self.learning_rate = learning_rate
        self.model = keras.models.Sequential([
            keras.layers.Dense(hidden_layer_neurons, activation=tf.nn.relu, input_shape=(9,)),
            keras.layers.Dense(100, activation=tf.nn.relu),
            keras.layers.Dense(9)
        ])
        self.model.compile(loss=tf.losses.mse, optimizer=tf.optimizers.SGD(0.000001))

    def action_value_policy(self, state):
        action_values = self.model.predict(state.T).T
        return action_values

    def update_weights(self, actual):
        self.model.fit(self.previous_state.T, actual.T, verbose=0, epochs=1)

    def choose_action(self, action_values):
        action_distribution = softmax(action_values).squeeze()
        action = np.random.choice(np.arange(9), p=action_distribution)

        return action

    def opponent_action(self, state):
        op_av = self.opponent_model(-state.T)
        return op_av

    def agent_start(self):
        # Save previous action, action value, and state for policy update. Currently the assumption is that the agent
        # always makes the first move.
        self.previous_state = self.environment.state
        self.previous_action_values = self.action_value_policy(self.previous_state)
        av_distribution = softmax(self.previous_action_values).squeeze()
        actions = np.random.choice(np.arange(9), 9, p=av_distribution, replace=False)
        for action in actions:
            valid_move, self.terminal, self.reward = self.environment.make_move(1, action)
            if valid_move:
                self.previous_action = action
                break

        # If the agent did not win the game, or have it end in a draw, the opponent performs a single action.
        # Choose and perform an action for the opponent. To avoid illegal moves, all action values are extracted and
        # chosen in descending order according to action value. If an action is invalid, that action value is replaced
        # with minus twice the absolute value of the worst possible original alternative to avoid it being chosen again.
        # This is repeated until a valid move is chosen.
        op_av = self.opponent_action(self.environment.state)
        op_actions = np.random.choice(np.arange(9), 9, p=softmax(op_av).squeeze(), replace=False)
        for action in op_actions:
            valid_move, self.terminal, self.reward = self.environment.make_move(-1, action)
            if valid_move:
                break

    def agent_step(self):
        # Calculate the action value set given the current set. Do not choose action yet, that will be done at a later
        # point. Once the action values have been calculated, gradient of the loss will be calculated. All entries are
        # 0, except for the action taken.
        action_values = self.action_value_policy(self.environment.state)
        actual = self.previous_action_values
        actual[self.previous_action] = self.reward
        if self.terminal == 0:
            actual[self.previous_action] += self.discount * np.max(action_values)

        # loss is now a 9 by 1 vector with all 0 entries, apart from the entry corresponding to the action taken during
        # self.previous_state. This is passed to the update function as the gradient of the loss function.
        self.update_weights(actual)

        # Once the weights have been updated, an additional agent move is only performed if the previous state was not
        # terminal. Generate a list of the actions randomly chosen according to their likelihood indicated by the
        # softmax of the action values of each action. This is done without replacement, so each action appears once in
        # the actions vector.
        if self.terminal == 0:
            self.previous_state = self.environment.state
            av_distribution = softmax(action_values).squeeze()
            actions = np.random.choice(np.arange(9), 9, p=av_distribution, replace=False)
            for action in actions:
                valid_move, self.terminal, self.reward = self.environment.make_move(1, action)
                if valid_move:
                    self.previous_action = action
                    self.previous_action_values = action_values
                    break

        # If the agent did not win the game, or have it end in a draw, the opponent performs a single action.
        # Choose and perform an action for the opponent. To avoid illegal moves, all action values are extracted and
        # chosen in descending order according to action value. If an action is invalid, that action value is replaced
        # with minus twice the absolute value of the worst possible original alternative to avoid it being chosen again.
        # This is repeated until a valid move is chosen.
        if self.terminal == 0:
            op_av = self.opponent_action(self.environment.state)
            op_actions = np.random.choice(np.arange(9), 9, p=softmax(op_av).squeeze(), replace=False)
            for action in op_actions:
                valid_move, self.terminal, self.reward = self.environment.make_move(-1, action)
                if valid_move:
                    self.reward *= -1
                    break

    def learn_to_play(self):
        # Iterate over the number of games to train for, this is saved as self.num_episodes, each episode being 1 game.
        # Each game starts with the parameters of the neural network being copied and set to the opponents parameters.
        # The idea is that the agent will constantly play against the previous version of itself.
        for game in range(self.num_episodes):
            print(game)
            self.opponent_model = keras.models.clone_model(self.model)
            self.environment.reset()
            starter = np.random.choice([1, -1])
            if starter == -1:
                op_av = self.opponent_action(self.environment.state)
                op_action = np.random.choice(np.arange(9), p=softmax(op_av).squeeze())
                self.environment.make_move(-1, op_action)
            self.agent_start()
            while True:
                if self.terminal:
                    self.agent_step()
                    break
                else:
                    self.agent_step()
