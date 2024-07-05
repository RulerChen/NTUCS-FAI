# Ref : https://medium.com/@mycorino/building-a-deep-q-network-powered-poker-bot-1a48e296805d

from game.players import BasePokerPlayer
from game.players import BasePokerPlayer
from game.engine.card import Card
from game.game import setup_config, start_poker
from .utils import encode_card, estimate_hole_card_win_rate

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torch.nn.functional import one_hot

import random
import numpy as np
import os
from collections import deque

from baseline4 import setup_ai as baseline4_ai
from baseline5 import setup_ai as baseline5_ai
from baseline6 import setup_ai as baseline6_ai
from baseline7 import setup_ai as baseline7_ai


def load_model(input_size, action_size, bet_size=30, training_mode=False):
    model = DQN(input_size, action_size, bet_size)

    if not training_mode:
        model_path = './src/model.pth'
        if os.path.exists(model_path):
            model.load_state_dict(torch.load(model_path))
            print('Model loaded from', model_path)
        else:
            print('New model created')
    else:
        print('Training mode activated')

    return model


def choose_action(state, model, valid_actions, epsilon=0.01, bet_size=30):
    best_action = valid_actions[0]['action']
    bet_amount_category = 0

    if np.random.rand() <= epsilon:
        best_action = np.random.choice([action['action'] for action in valid_actions])
        bet_amount_category = np.random.randint(0, bet_size)
    else:
        predictions = model.predict(state.reshape(1, -1))
        q_values = predictions[0][0]
        bet_amounts = predictions[1][0]

        action_indices = {action['action']: idx for idx, action in enumerate(valid_actions)}

        valid_action_q_values = [q_values[action_indices[action['action']]] for action in valid_actions]
        best_action = valid_actions[np.argmax(valid_action_q_values)]['action']
        bet_amount_category = np.argmax(bet_amounts)

    return best_action, bet_amount_category


def compute_reward(action, is_winner, stack_size_before, stack_size_after, hole_card, round_state):
    reward = 0

    if is_winner:
        reward += 20
    else:
        reward -= 20

    if action == 'fold':
        reward -= 25

    hole_card = [Card.from_str(c) for c in hole_card]
    community_card = [Card.from_str(c) for c in round_state['community_card']]
    win_rate = estimate_hole_card_win_rate(nb_simulation=3000, nb_player=2,
                                           hole_card=hole_card, community_card=community_card)

    if win_rate > 0.8:
        reward += 5
    elif win_rate < 0.2:
        reward -= 5

    stack_change = stack_size_after - stack_size_before
    reward += stack_change / 50

    pot_size = round_state['pot']['main']['amount']
    reward -= pot_size / 40

    return reward


def replay(memory, model, model_target, batch_size, gamma, action_loss, bet_loss, optimizer):

    states, actions, bet_amount_categories, rewards, next_states, dones = memory.sample(batch_size)

    states = np.array(states)
    next_states = np.array(next_states)
    bet_amount_categories = np.array(bet_amount_categories)
    rewards = np.array(rewards)
    dones = np.array(dones)

    states = torch.tensor(states, dtype=torch.float)
    next_states = torch.tensor(next_states, dtype=torch.float)
    bet_amount_categories = torch.tensor(
        bet_amount_categories, dtype=torch.long)
    rewards = torch.tensor(rewards, dtype=torch.float)
    dones = torch.tensor(dones, dtype=torch.float)

    with torch.no_grad():
        next_q_values, next_bets = model_target(next_states)
    target_action = rewards + gamma * torch.max(next_q_values, dim=1)[0] * (1 - dones)

    q_value_action, q_values_bet_amount = model(states)

    for i, action in enumerate(actions):
        q_value_action[i][action] = target_action[i]

    bet_amount_categories = np.array(
        [cat if cat is not None else 0 for cat in bet_amount_categories], dtype='int64')

    bet_amount_categories_one_hot = one_hot(torch.tensor(
        bet_amount_categories), num_classes=q_values_bet_amount.shape[1]).float()

    dataset = TensorDataset(states, q_value_action,
                            bet_amount_categories_one_hot)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    model.train()
    for state, q_value_action, bet_amount_category in dataloader:
        optimizer.zero_grad()
        action, bet = model(state)
        loss = action_loss(action, q_value_action) + bet_loss(bet, torch.argmax(bet_amount_category, dim=1))

        loss.backward()
        optimizer.step()


class DQN(nn.Module):
    def __init__(self, input_size, action_size, bet_size):
        super(DQN, self).__init__()
        self.action_network = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.Linear(32, action_size)
        )
        self.bet_network = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.Linear(32, bet_size)
        )

    def forward(self, state):

        action = self.action_network(state)
        bet = F.softmax(self.bet_network(state), dim=1)
        return action, bet


class ReplayBuffer:
    def __init__(self, buffer_size):
        self.buffer = deque(maxlen=buffer_size)

    def add(self, state, action, bet_amount_category, reward, next_state, done):
        self.buffer.append(
            (state, action, bet_amount_category, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, min(len(self.buffer), batch_size))
        state, action, bet_amount_category, reward, next_state, done = zip(
            *batch)
        return state, action, bet_amount_category, reward, next_state, done


class DQNPokerAgent:
    def __init__(self, state_size, action_size, model=None, replay_buffer_size=50000, batch_size=32, gamma=0.95, epsilon=1, epsilon_min=0.01, epsilon_decay=0.9995):
        self.state_size = state_size
        self.action_size = action_size
        self.replay_buffer = ReplayBuffer(replay_buffer_size)
        self.batch_size = batch_size
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.model = model

        self.q_network = model if model else load_model(state_size, action_size)
        self.target_q_network = DQN(state_size, action_size, 30)
        self.update_target_network()

        self.optimizer = optim.Adam(self.q_network.parameters())
        self.action_loss = nn.MSELoss()
        self.bet_loss = nn.CrossEntropyLoss()

    def update_target_network(self):
        """Updates the target Q-network's weights."""
        self.target_q_network.load_state_dict(self.q_network.state_dict())

    def act(self, state, valid_actions):
        """Choose an action and a bet amount category based on the current state."""
        action, bet_amount_category = choose_action(np.reshape(
            state, [1, self.state_size]), self.q_network, valid_actions, self.epsilon)
        return action, bet_amount_category

    def learn(self):
        """Trains the model using a batch of experiences from the replay buffer."""
        # Sample a batch of experiences from the replay buffer
        states, actions, bet_amount_categories, rewards, next_states, dones = self.replay_buffer.sample(
            self.batch_size)

        replay(self.replay_buffer, self.q_network,
               self.target_q_network, self.batch_size, self.gamma, self.action_loss, self.bet_loss, self.optimizer)

        # Epsilon decay
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay


class DQNPokerPlayer(BasePokerPlayer):
    def __init__(self, dqn_agent=None, state_size=120, bet_size=30, training_mode=False):
        super().__init__()
        self.dqn_agent = dqn_agent if dqn_agent else DQNPokerAgent(state_size, 3)
        self.bet_size = bet_size
        self.current_state = np.zeros(state_size)
        self.prev_state = np.zeros(state_size)
        self.hole_card = None
        self.last_action = None
        self.action_index = None
        self.bet_amount_category = None
        self.done = False

        self.last_action_details = {'action': None, 'amount': 0}

        self.training_mode = training_mode
        self.games_since_last_learn = 0

        self.total_round = 20
        self.round = 0

    def declare_action(self, valid_actions, hole_card, round_state):
        self.round = round_state['round_count']

        if self.training_mode == False and self.my_stack - 1000 > (self.total_round - self.round + 1) * 10:
            return "fold", 0

        if self.current_state is None:
            self.current_state = self._extract_state(round_state, hole_card)

        action, bet_amount_category = self.dqn_agent.act(self.current_state, valid_actions)
        self.last_action = action
        self.bet_amount_category = bet_amount_category

        action_index = self.map_action_to_index(action, valid_actions)
        self.action_index = action_index

        action_dict = next((item for item in valid_actions if item["action"] == action), None)

        if action == 'raise' and action_dict['amount']['min'] != -1:
            min_raise = action_dict['amount']['min']
            max_raise = action_dict['amount']['max']
            fraction = (self.bet_amount_category + 1) / self.bet_size
            # print(f"min_raise: {min_raise}, max_raise: {max_raise}, fraction: {fraction}")
            amount = int(min_raise + fraction * (max_raise - min_raise))
        elif action == 'raise' and action_dict['amount']['min'] == -1:
            action = 'call'
            amount = valid_actions[1]['amount']
        else:
            amount = action_dict['amount'] if action_dict else 0

        self.last_action_details = {'action': action, 'amount': amount}

        return action, amount

    def receive_game_start_message(self, game_info):
        self.my_stack = game_info['rule']['initial_stack']

    def receive_round_start_message(self, round_count, hole_card, seats):
        # print(f"hold card: {hole_card}")
        self.hole_card = hole_card

        self.total_round = 20
        self.round = 0

        self.done = False
        self.prev_state = None
        self.action_index = None
        self.bet_amount_category = None
        self.last_action_details = {'action': None, 'amount': 0}

    def receive_game_update_message(self, action, round_state):
        self.prev_state = self.current_state

        if self.current_state is None:
            print('None detected in receive_game_update_message', self.current_state)

        if self.hole_card:
            self.current_state = self._extract_state(round_state, self.hole_card)

    def _get_stack_size(self, player_uuid, round_state):
        for player_info in round_state['seats']:
            if player_info['uuid'] == player_uuid:
                return player_info['stack']
        return None

    def receive_round_result_message(self, winners, hand_info, round_state):
        # print('\n')
        is_winner = any(winner['uuid'] == self.uuid for winner in winners)

        stack_size_before = self.my_stack
        stack_size_after = self._get_stack_size(self.uuid, round_state)
        self.my_stack = stack_size_after

        self.done = True
        if self.bet_amount_category is None:
            self.bet_amount_category = 0

        if self.training_mode:
            reward = compute_reward(self.last_action, is_winner, stack_size_before,
                                    stack_size_after, self.hole_card, round_state)

            self.dqn_agent.replay_buffer.add(
                self.prev_state, self.action_index, self.bet_amount_category, reward, self.current_state, self.done)

            self.games_since_last_learn += 1
            if self.games_since_last_learn >= 5:
                self.dqn_agent.learn()
                self.games_since_last_learn = 0

    def _get_stack_size(self, uuid, round_state):
        for player in round_state['seats']:
            if player['uuid'] == uuid:
                return player['stack']
        return None

    def _extract_state(self, round_state, hole_card):
        hole_cards_encoded = [encode_card(card) for card in hole_card]
        community_cards_encoded = [encode_card(
            card) for card in round_state['community_card']]

        N = len(encode_card(hole_card[0])) if hole_card else 26

        hole_cards_vector = np.concatenate(hole_cards_encoded) if hole_cards_encoded else np.zeros(N * 2)
        community_cards_vector = np.concatenate(
            community_cards_encoded) if community_cards_encoded else np.zeros(N * len(community_cards_encoded))

        zero_vector = np.zeros(N)
        padded_community_cards = np.concatenate([community_cards_vector] + [
            zero_vector for _ in range(5 - len(community_cards_encoded))])

        pot_size = np.array([round_state['pot']['main']['amount'] / 2000])

        state = np.concatenate([hole_cards_vector, padded_community_cards, pot_size])
        return state

    def map_action_to_index(self, action, valid_actions):
        action_dict = {act['action']: idx for idx, act in enumerate(valid_actions)}
        return action_dict.get(action, -1)

    def receive_street_start_message(self, street, round_state):
        pass


def setup_ai():
    return DQNPokerPlayer()


def train():
    dqn_model = load_model(120, 3, 30, training_mode=True)
    dqn_agent = DQNPokerAgent(120, 3, dqn_model)
    agent = DQNPokerPlayer(dqn_agent, 120, 30, training_mode=True)

    for _ in range(100):
        print(f"epoch: {_ + 1}")

        opponent = random.randint(1, 100)
        max_round = 20

        config = setup_config(
            max_round=max_round, initial_stack=1000, small_blind_amount=5)
        config.register_player(name="p1", algorithm=agent)

        if opponent <= 25:
            config.register_player(name="p2", algorithm=baseline4_ai())
            opponent = 4
        elif opponent <= 50:
            config.register_player(name="p2", algorithm=baseline5_ai())
            opponent = 5
        elif opponent <= 75:
            config.register_player(name="p2", algorithm=baseline6_ai())
            opponent = 6
        elif opponent <= 100:
            config.register_player(name="p2", algorithm=baseline7_ai())
            opponent = 7

        game_result = start_poker(config, verbose=0)
        if game_result['players'][0]['stack'] > game_result['players'][1]['stack']:
            print(
                f"Player1, {opponent}, {game_result['players'][0]['stack']}, {game_result['players'][1]['stack']}")
        else:
            print(
                f"Player2, {opponent}, {game_result['players'][0]['stack']}, {game_result['players'][1]['stack']}")

    torch.save(dqn_model.state_dict(), './src/model.pth')
