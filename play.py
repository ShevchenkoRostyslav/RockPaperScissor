from src.game import RockPaperScissorGame, User, Bot
import tensorflow.keras as K
import logging

LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.DEBUG)


if __name__ == '__main__':
    trained_model = K.models.load_model('model_weights/model.h5')
    user = User(trained_model)
    bot = Bot()
    game = RockPaperScissorGame(user, bot, 3)
    game.play()