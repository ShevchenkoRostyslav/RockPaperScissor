import time
from unittest import TestCase
from play import RockPaperScissorGame, Bot, User, LOGGER
import cv2
import logging
import sys
import tensorflow.keras as K

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)


class TestRockPaperScissorGame(TestCase):
    def test_play(self):
        self.fail()

    def test_round_bots(self):
        # case 1: two bots
        cap = cv2.VideoCapture(0)
        user = Bot()
        bot = Bot()
        game = RockPaperScissorGame(user, bot, 3)
        game.round(cap)

        # close the camera
        cap.release()

        # close all the opened windows
        cv2.destroyAllWindows()

    def test_round_bot_user(self):
        # case 1: two bots
        cap = cv2.VideoCapture(0)
        trained_model = K.models.load_model('../model_weights/model.h5')
        user = User(trained_model)
        bot = Bot()
        game = RockPaperScissorGame(user, bot, 3)
        game.round(cap)

        # close the camera
        cap.release()

        # close all the opened windows
        cv2.destroyAllWindows()

    def test_visualize_choices(self):
        cap = cv2.VideoCapture(0)
        bot1 = Bot()
        bot2 = Bot()
        game = RockPaperScissorGame(bot1, bot2, 3)
        while True:
            ret, frame = cap.read()
            frame = game.visualize_choices(frame, 'rock', 'paper')
            frame = game.visualize_camera_frame(frame)
            if cv2.waitKey(1) & 0xff == ord('q'):
                break

    def test_update_scores(self):
        user = Bot()
        bot = Bot()
        game = RockPaperScissorGame(user, bot, 3)
        # paper bits the rock
        game.update_scores(choice1='paper', choice2='rock')
        self.assertEqual(user.score, 1)
        self.assertEqual(bot.score, 0)
        # rock bits the scissor
        game.update_scores(choice1='scissor', choice2='rock')
        self.assertEqual(user.score, 1)
        self.assertEqual(bot.score, 1)
        # scissor = scissor
        game.update_scores(choice1='scissor', choice2='scissor')
        self.assertEqual(user.score, 1)
        self.assertEqual(bot.score, 1)
        # unexpected word should throw the ValueError
        with self.assertRaises(ValueError):
            game.update_scores(choice1='scissor', choice2='blabla')

    def test_check_winner(self):
        player1 = Bot()
        player2 = Bot()
        game = RockPaperScissorGame(player1, player2, 3)
        player1.score = 5
        player2.score = 1
        winner = game.check_winner()
        self.assertEqual(winner, player1)
        player1.score = 0
        player2.score = 1
        winner = game.check_winner()
        self.assertEqual(winner, player2)
        player1.score = 1
        player2.score = 1
        winner = game.check_winner()
        self.assertEqual(winner, None)

    def test_declare_winner(self):
        player1 = Bot()
        game = RockPaperScissorGame(player1, player1, 1)
        cap = cv2.VideoCapture(0)
        game.declare_winner(player1, cap)