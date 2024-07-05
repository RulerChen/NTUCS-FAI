from game.players import BasePokerPlayer
from .cards import get_best_hand, is_flush, most_flush, longest_straight
import json


class CallPlayer(
    BasePokerPlayer
):

    def __init__(self):
        super(BasePokerPlayer, self)
        self.hole_card = []
        self.my_stack = 1000
        self.opponent_stack = 1000
        self.total_round = 20
        self.round = 0

    def declare_action(self, valid_actions, hole_card, round_state):
        """
        玩家行動時
        """
        self.round = round_state['round_count']

        if self.my_stack - 1000 > (self.total_round - self.round + 1) * 10:
            return "fold", 0

        action = "fold"
        amount = 0
        self.hole_card = hole_card

        call_amount = valid_actions[1]['amount']
        min_raise = valid_actions[2]['amount']['min']
        max_raise = valid_actions[2]['amount']['max']

        if round_state['street'] == 'preflop':
            rank, num = get_best_hand(hole_card)
            if rank == 2:
                action, amount = "raise", min_raise
            elif call_amount >= 100:
                action, amount = "fold", 0
            else:
                action, amount = "call", call_amount

        elif round_state['street'] == 'flop':
            rank, num = get_best_hand(
                hole_card + round_state['community_card'])
            agent_rank, agent_num = get_best_hand(
                round_state['community_card'])

            if rank == 9:
                action, amount = "raise", max_raise
            elif rank == 8:
                action, amount = "raise", max_raise
            elif rank == 7:
                if agent_rank == 4:
                    action, amount = "raise", min_raise
                elif agent_rank == 2:
                    action, amount = "raise", min_raise + \
                        (max_raise - min_raise) // 4
                else:
                    action, amount = "call", call_amount
            elif rank == 6:
                if most_flush(round_state['community_card']) == 3:
                    action, amount = "raise", min_raise + \
                        (max_raise - min_raise) // 4
                else:
                    action, amount = "call", call_amount
            elif rank == 5:
                if longest_straight(round_state['community_card']) == 3:
                    action, amount = "raise", min_raise + \
                        (max_raise - min_raise) // 4
                else:
                    action, amount = "call", call_amount
            elif rank == 4:
                if agent_rank == 4:
                    if call_amount >= 300:
                        action, amount = "fold", 0
                    else:
                        action, amount = "call", call_amount
                elif agent_rank == 2:
                    action, amount = "raise", min_raise
                else:
                    action, amount = "raise", min_raise + \
                        (max_raise - min_raise) // 4
            elif rank == 3:
                if agent_rank == 2:
                    if call_amount >= 200:
                        action, amount = "fold", 0
                    else:
                        action, amount = "call", call_amount
                else:
                    action, amount = "call", call_amount
            elif most_flush(hole_card + round_state['community_card']) >= 4:
                if call_amount >= 50:
                    action, amount = "fold", 0
                else:
                    action, amount = "call", call_amount
            elif longest_straight(hole_card + round_state['community_card']) >= 4:
                if call_amount >= 50:
                    action, amount = "fold", 0
                else:
                    action, amount = "call", call_amount
            elif rank == 2:
                if agent_rank == 2:
                    if call_amount >= 100:
                        action, amount = "fold", 0
                    else:
                        action, amount = "call", call_amount
                action, amount = "call", call_amount
            elif rank == 1:
                if call_amount >= 50:
                    action, amount = "fold", 0
                else:
                    action, amount = "call", call_amount
            else:
                action, amount = "fold", 0

        elif round_state['street'] == 'turn':
            rank, num = get_best_hand(
                hole_card + round_state['community_card'])
            pot_size = round_state['pot']['main']['amount']
            agent_rank, agent_num = get_best_hand(
                round_state['community_card'])

            if rank == 9:
                action, amount = "raise", max_raise
            elif rank == 8:
                if agent_rank == 8:
                    action, amount = "call", call_amount
                else:
                    action, amount = "raise", max_raise
            elif rank == 7:
                if agent_rank == 4:
                    action, amount = "raise", min_raise
                elif agent_rank == 3:
                    action, amount = "call", call_amount
                elif agent_rank == 2:
                    action, amount = "raise", min_raise + \
                        (max_raise - min_raise) // 4
                else:
                    action, amount = "call", call_amount
            elif rank == 6:
                if most_flush(round_state['community_card']) == 4:
                    action, amount = "raise", min_raise
                elif most_flush(round_state['community_card']) == 3:
                    action, amount = "raise", min_raise + (max_raise - min_raise) // 4
                else:
                    action, amount = "call", call_amount
            elif rank == 5:
                if longest_straight(round_state['community_card']) == 4:
                    action, amount = "raise", min_raise
                elif longest_straight(round_state['community_card']) == 3:
                    action, amount = "raise", min_raise + \
                        (max_raise - min_raise) // 4
                else:
                    action, amount = "call", call_amount
            elif rank == 4:
                if agent_rank == 2:
                    action, amount = "raise", min_raise
                else:
                    action, amount = "raise", min_raise + \
                        (max_raise - min_raise) // 4
            elif rank == 3:
                if agent_rank == 3 or agent_rank == 2:
                    if call_amount <= 30:
                        action, amount = "call", call_amount
                    else:
                        action, amount = "fold", 0
                else:
                    action, amount = "raise", min_raise
            elif most_flush(hole_card + round_state['community_card']) >= 4:
                if call_amount >= 50:
                    action, amount = "fold", 0
                else:
                    action, amount = "call", call_amount
            elif longest_straight(hole_card + round_state['community_card']) >= 4:
                if call_amount >= 50:
                    action, amount = "fold", 0
                else:
                    action, amount = "call", call_amount
            elif rank == 2:
                if agent_rank == 2:
                    if call_amount <= 20:
                        action, amount = "call", call_amount
                    else:
                        action, amount = "fold", 0
                else:
                    action, amount = "call", call_amount
            elif rank == 1:
                if call_amount <= 20:
                    action, amount = "call", call_amount
                else:
                    action, amount = "fold", 0
            else:
                action, amount = "fold", 0

        elif round_state['street'] == 'river':
            rank, num = get_best_hand(
                hole_card + round_state['community_card'])
            pot_size = round_state['pot']['main']['amount']
            agent_rank, agent_num = get_best_hand(
                round_state['community_card'])

            if rank == 9:
                if agent_rank == 9:
                    action, amount = "call", call_amount
                else:
                    action, amount = "raise", max_raise
            elif rank == 8:
                if agent_rank == 8:
                    action, amount = "call", call_amount
                else:
                    action, amount = "raise", max_raise
            elif rank == 7:
                if agent_rank == 7:
                    action, amount = "call", call_amount
                elif agent_rank == 4:
                    action, amount = "raise", min_raise
                elif agent_rank == 3:
                    action, amount = "call", call_amount
                elif agent_rank == 2:
                    action, amount = "raise", min_raise + \
                        (max_raise - min_raise) // 2
                else:
                    action, amount = "call", call_amount
            elif rank == 6:
                action, amount = "call", call_amount
            elif rank == 5:
                action, amount = "call", call_amount
            elif rank == 4:
                if agent_rank == 2:
                    action, amount = "call", call_amount
                else:
                    action, amount = "raise", min_raise
            elif rank == 3:
                if agent_rank == 3 or agent_rank == 2:
                    if call_amount <= 30:
                        action, amount = "call", call_amount
                    else:
                        action, amount = "fold", 0
                else:
                    action, amount = "raise", min_raise
            elif rank == 2:
                if agent_rank == 2:
                    action, amount = "fold", 0
                elif call_amount > 20 and num < 10:
                    action, amount = "fold", 0
                elif call_amount > 30:
                    action, amount = "fold", 0
                else:
                    action, amount = "call", call_amount
            elif rank == 1:
                if call_amount <= 10:
                    action, amount = "call", call_amount
                else:
                    action, amount = "fold", 0
            else:
                action, amount = "fold", 0

        return action, amount

    def receive_game_start_message(self, game_info):
        """
        遊戲開始時
        """
        pass

    def receive_round_start_message(self, round_count, hole_card, seats):
        """
        回合開始時
        """
        pass

    def receive_street_start_message(self, street, round_state):
        """
        下注開始時
        """
        pass

    def receive_game_update_message(self, action, round_state):
        """
        玩家進行動作時
        """
        pass

    def receive_round_result_message(self, winners, hand_info, round_state):
        """
        回合結束時
        """
        self.opponent_stack = round_state['seats'][0]['stack']
        self.my_stack = round_state['seats'][1]['stack']

        round = round_state['round_count']
        round_state['hole_card'] = self.hole_card
        with open(f'./log/{round}.json', 'w') as f:
            json.dump(round_state, f, indent=4)

        pass


def setup_ai():
    return CallPlayer()
