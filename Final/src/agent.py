# Ref : https://github.com/ishikota/PyPokerEngine/blob/master/pypokerengine/utils/card_utils.py

from game.players import BasePokerPlayer
from game.engine.card import Card
from .utils import estimate_hole_card_win_rate
import random
# import json


# implement monte carlo tree search here
class CallPlayer(
    BasePokerPlayer
):  # Do not forget to make parent class as "BasePokerPlayer"

    def __init__(self):
        super(BasePokerPlayer, self)
        self.hole_card = []
        self.my_stack = 1000
        self.opponent_stack = 1000
        self.my_bet = 0
        self.upper_bet = 0
        self.total_round = 20
        self.round = 0
        self.is_berserk = False

    def declare_action(self, valid_actions, hole_card, round_state):
        self.hole_card = hole_card
        self.round = round_state['round_count']
        self.upper_bet = (self.total_round - self.round + 1) * 4

        if self.my_stack - 1000 > (self.total_round - self.round + 1) * 10:
            return "fold", 0
        elif self.my_stack < 1000 - (self.total_round - self.round + 1) * 8:
            self.is_berserk = True
        elif self.round >= 16 and self.my_stack < self.opponent_stack:
            self.is_berserk = True
        else:
            self.is_berserk = False

        call_amount = valid_actions[1]['amount']
        min_raise = valid_actions[2]['amount']['min']
        max_raise = valid_actions[2]['amount']['max']

        hole_card = [Card.from_str(c) for c in hole_card]
        community_card = [Card.from_str(c)
                          for c in round_state['community_card']]

        win_rate = estimate_hole_card_win_rate(
            nb_simulation=50000, nb_player=2, hole_card=hole_card, community_card=community_card)

        if self.is_berserk:
            if round_state['street'] == 'preflop':
                if win_rate >= 0.9:
                    action, amount = "raise", min_raise + \
                        (max_raise - min_raise) // 2
                elif win_rate >= 0.7:
                    action, amount = "raise", min_raise + \
                        (max_raise - min_raise) // 20
                elif win_rate >= 0.3:
                    action, amount = "call", call_amount
                elif call_amount == 0:
                    action, amount = "call", call_amount
                else:
                    action, amount = "fold", 0

            elif round_state['street'] == 'flop':
                if win_rate >= 0.9:
                    action, amount = "raise", min_raise + \
                        (max_raise - min_raise) // 2
                elif win_rate >= 0.7:
                    action, amount = "raise", min_raise + \
                        (max_raise - min_raise) // 5
                elif win_rate >= 0.5:
                    action, amount = "raise", min_raise + \
                        (max_raise - min_raise) // 20
                elif win_rate <= 0.3:
                    action = random.choice(["raise", "call", "fold"])
                    if action == "raise":
                        amount = min_raise
                    elif action == "call":
                        amount = call_amount
                    elif action == "fold":
                        amount = 0
                elif call_amount == 0:
                    action, amount = "call", call_amount
                elif self.my_bet >= self.upper_bet:
                    action, amount = "call", call_amount
                else:
                    action, amount = "fold", 0

            elif round_state['street'] == 'turn':
                if win_rate >= 0.9:
                    action, amount = "raise", min_raise + \
                        (max_raise - min_raise) // 2
                elif win_rate >= 0.7:
                    action, amount = "raise", min_raise + \
                        (max_raise - min_raise) // 5
                elif win_rate >= 0.5:
                    action, amount = "raise", min_raise + \
                        (max_raise - min_raise) // 20
                elif win_rate <= 0.3:
                    action = random.choice(["raise", "call", "fold"])
                    if action == "raise":
                        amount = min_raise
                    elif action == "call":
                        amount = call_amount
                    elif action == "fold":
                        amount = 0
                elif self.my_bet >= self.upper_bet:
                    action, amount = "call", call_amount
                elif call_amount == 0:
                    action, amount = "call", call_amount
                else:
                    action, amount = "fold", 0

            elif round_state['street'] == 'river':
                if win_rate >= 0.9:
                    action, amount = "raise", max_raise
                elif win_rate >= 0.7:
                    action, amount = "raise", min_raise + \
                        (max_raise - min_raise) // 5
                elif win_rate >= 0.5:
                    action, amount = "call", call_amount
                elif win_rate <= 0.2:
                    action = random.choice(["raise", "call", "fold"])
                    if action == "raise":
                        amount = min_raise
                    elif action == "call":
                        amount = call_amount
                    elif action == "fold":
                        amount = 0
                elif self.my_bet >= self.upper_bet:
                    action, amount = "call", call_amount
                elif call_amount == 0:
                    action, amount = "call", call_amount
                else:
                    action, amount = "fold", 0

        elif self.is_berserk == False:
            if round_state['street'] == 'preflop':
                if win_rate >= 0.7:
                    action, amount = "raise", min_raise
                elif win_rate >= 0.3 and call_amount <= 50:
                    action, amount = "call", call_amount
                elif call_amount == 0:
                    action, amount = "call", call_amount
                else:
                    action, amount = "fold", 0

            elif round_state['street'] == 'flop':
                if win_rate >= 0.9:
                    action, amount = "raise", min_raise + \
                        (max_raise - min_raise) // 4
                elif win_rate >= 0.7:
                    if call_amount <= 50:
                        action, amount = "raise", min_raise + \
                            (max_raise - min_raise) // 30
                    else:
                        action, amount = "raise", min_raise
                elif call_amount == 0:
                    action, amount = "call", call_amount
                elif self.my_bet >= self.upper_bet:
                    action, amount = "call", call_amount
                elif win_rate >= 0.6:
                    if call_amount <= 50:
                        action, amount = "call", call_amount
                    else:
                        action, amount = "fold", 0
                elif win_rate <= 0.2:
                    action = random.choice(["raise", "call", "fold"])
                    if action == "raise":
                        amount = min_raise
                    elif action == "call":
                        amount = call_amount
                    elif action == "fold":
                        amount = 0
                else:
                    action, amount = "fold", 0

            elif round_state['street'] == 'turn':
                if win_rate >= 0.9:
                    action, amount = "raise", min_raise + \
                        (max_raise - min_raise) // 4
                elif win_rate >= 0.7:
                    if call_amount <= 100:
                        action, amount = "raise", min_raise + \
                            (max_raise - min_raise) // 20
                    else:
                        action, amount = "call", call_amount
                elif call_amount == 0:
                    action, amount = "call", call_amount
                elif self.my_bet >= self.upper_bet:
                    action, amount = "call", call_amount
                elif win_rate >= 0.5:
                    if call_amount <= 50:
                        action, amount = "call", call_amount
                    else:
                        action, amount = "fold", 0
                else:
                    action, amount = "fold", 0

            elif round_state['street'] == 'river':
                if win_rate >= 0.9:
                    action, amount = "raise", max_raise
                elif win_rate >= 0.7:
                    if call_amount <= 50:
                        action, amount = "raise", min_raise + \
                            (max_raise - min_raise) // 15
                    else:
                        action, amount = "call", call_amount
                elif call_amount == 0:
                    action, amount = "call", call_amount
                elif win_rate >= 0.5:
                    action, amount = "call", call_amount
                elif self.my_bet >= self.upper_bet:
                    action, amount = "call", call_amount
                else:
                    action, amount = "fold", 0

        self.my_bet += amount
        return action, amount

    def receive_game_start_message(self, game_info):
        pass

    def receive_round_start_message(self, round_count, hole_card, seats):
        self.my_bet = 0
        pass

    def receive_street_start_message(self, street, round_state):
        pass

    def receive_game_update_message(self, action, round_state):
        pass

    def receive_round_result_message(self, winners, hand_info, round_state):
        # print('\n')
        if round_state['seats'][0]['uuid'] == self.uuid:
            self.my_stack = round_state['seats'][0]['stack']
            self.opponent_stack = round_state['seats'][1]['stack']
        else:
            self.opponent_stack = round_state['seats'][0]['stack']
            self.my_stack = round_state['seats'][1]['stack']

        # round = round_state['round_count']
        # round_state['hole_card'] = self.hole_card
        # with open(f'./log/{round}.json', 'w') as f:
        #     json.dump(round_state, f, indent=4)


def setup_ai():
    return CallPlayer()
