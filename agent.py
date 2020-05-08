import numpy as np
import environment as env
import copy


def relu(x):
    return (x >= 0) * x


def relu_grad(x):
    return (x >= 0) * 1


def mse(actual, prediction):
    return np.sum((actual - prediction) ** 2) / 9


def mse_grad(actual, prediction):
    return -2 * (actual - prediction)


def softmax(x):
    max_x = np.max(x)
    exp_x = np.exp(x - max_x)
    return exp_x / np.sum(exp_x)


class ttt_agent:
    layer_sizes = []
    nn_parameters = {}
    activations = {}
    layer_z_values = {}
    opponent_parameters = {}
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
        self.layer_sizes = np.array([9, hidden_layer_neurons, 9])
        self.num_episodes = num_episodes
        self.discount = discount
        self.learning_rate = learning_rate

        # Initialization with He-et-al random initialization
        self.activations[0] = 0
        for i in range(1, len(self.layer_sizes)):
            self.nn_parameters[i] = {}
            self.nn_parameters[i]["w"] = np.random.randn(self.layer_sizes[i], self.layer_sizes[i - 1])\
                                         * np.sqrt(2 / self.layer_sizes[i - 1])
            self.nn_parameters[i]["b"] = np.zeros((self.layer_sizes[i], 1))
            self.activations[i] = 0
            self.layer_z_values[i] = 0

    def action_value_policy(self, state):
        action_values = state
        self.activations[0] = state
        for layer in range(1, len(self.layer_sizes)):
            action_values = np.dot(self.nn_parameters[layer]["w"], action_values) + self.nn_parameters[layer]["b"]
            self.layer_z_values[layer] = action_values
            if layer != (len(self.layer_sizes) - 1):
                action_values = relu(action_values)
            self.activations[layer] = action_values
        return action_values

    def update_weights(self, loss_grad):
        gradients = {}
        delta_z = loss_grad * np.ones([9, 1])
        gradients[2] = {}
        gradients[2]["w"] = np.dot(delta_z, self.activations[1].T)
        gradients[2]["b"] = delta_z
        delta_z = np.dot(self.nn_parameters[2]["w"].T, delta_z) * relu_grad(self.layer_z_values[1])
        gradients[1] = {}
        gradients[1]["w"] = np.dot(delta_z, self.activations[0].T)
        gradients[1]["b"] = delta_z

        for i in range(1, len(self.layer_sizes)):
            self.nn_parameters[i]["w"] -= self.learning_rate * gradients[i]["w"]
            self.nn_parameters[i]["b"] -= self.learning_rate * gradients[i]["b"]

    def choose_action(self, state):
        action_values = self.action_value_policy(state)
        action_distribution = softmax(action_values).squeeze()

        action = np.random.choice(np.arange(9), p=action_distribution)

        return action, action_values

    def opponent_action(self, state):
        actions = np.random.choice(np.arange(9), 9, replace=False)

        return actions

    def agent_start(self):
        # Save previous action, action value, and state for policy update. Currently the assumption is that the agent
        # always makes the first move.
        self.previous_state = self.environment.state
        self.previous_action, action_values = self.choose_action(self.previous_state)
        self.previous_action_values = action_values

        # Perform chosen action
        self.environment.make_move(1, self.previous_action)

        # If the agent did not win the game, or have it end in a draw, the opponent performs a single action.
        # Choose and perform an action for the opponent. To avoid illegal moves, all action values are extracted and
        # chosen in descending order according to action value. If an action is invalid, that action value is replaced
        # with minus twice the absolute value of the worst possible original alternative to avoid it being chosen again.
        # This is repeated until a valid move is chosen.
        opponent_actions = self.opponent_action(self.environment.state)
        for action in opponent_actions:
            valid_move, self.terminal, self.reward = self.environment.make_move(-1, action)
            if valid_move:
                break

    def agent_step(self):
        # Calculate the action value set given the current set. Do not choose action yet, that will be done at a later
        # point. Once the action values have been calculated, gradient of the loss will be calculated. All entries are
        # 0, except for the action taken.
        valid_move = 0
        _, action_values = self.choose_action(self.environment.state)
        actual = action_values
        actual[self.previous_action] = self.reward
        if self.terminal == 0:
            actual[self.previous_action] += self.discount * np.max(action_values)
        prediction = self.previous_action_values
        loss_grad = mse_grad(actual, prediction)

        # loss is now a 9 by 1 vector with all 0 entries, apart from the entry corresponding to the action taken during
        # self.previous_state. This is passed to the update function as the gradient of the loss function.
        self.update_weights(loss_grad)

        # Once the weights have been updated, an additional agent move is only performed if the previous state was not
        # terminal. Generate a list of the actions randomly chosen according to their likelihood indicated by the
        # softmax of the action values of each action. This is done without replacement, so each action appears once in
        # the actions vector.
        if self.terminal == 0:
            self.previous_state = self.environment.state
            actions = np.random.choice(np.arange(9), 9, p=softmax(action_values).squeeze(), replace=False)
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
            opponent_actions = self.opponent_action(self.environment.state)
            for action in opponent_actions:
                valid_move, self.terminal, self.reward = self.environment.make_move(-1, action)
                if valid_move:
                    self.reward *= -100
                    break

    def learn_to_play(self):
        # Iterate over the number of games to train for, this is saved as self.num_episodes, each episode being 1 game.
        # Each game starts with the parameters of the neural network being copied and set to the opponents parameters.
        # The idea is that the agent will constantly play against the previous version of itself.
        for game in range(self.num_episodes):
            print(game)
            self.environment.reset()
            self.opponent_parameters = copy.deepcopy(self.nn_parameters)
            self.agent_start()
            while True:
                if self.terminal:
                    self.agent_step()
                    break
                else:
                    self.agent_step()
