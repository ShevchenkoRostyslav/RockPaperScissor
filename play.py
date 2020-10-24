import argparse
from typing import Dict

from src.game import RockPaperScissorGame, User, Bot
import tensorflow.keras as K
import logging

LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.DEBUG)

def parse_args() -> Dict:
    """Input argument parser

    :return:
    """
    parser = argparse.ArgumentParser(description='Play the rock-paper-scissor game using the web-camera')
    parser.add_argument('--rounds', type=int, help='Number of rounds to be played', required=True)
    args = parser.parse_args()
    return vars(args)


def initialize_game(rounds: int) -> RockPaperScissorGame:
    # load the trained model weights
    trained_model = K.models.load_model('model_weights/model.h5')
    # initialize the players: Bot and User
    user = User(trained_model)
    bot = Bot()
    game = RockPaperScissorGame(user, bot, rounds)
    return game


if __name__ == '__main__':
    in_args = parse_args()
    game = initialize_game(in_args['rounds'])
    game.play()