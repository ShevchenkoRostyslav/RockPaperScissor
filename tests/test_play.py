import time
from unittest import TestCase
from play import RockPaperScissorGame, Bot, User
import cv2

import tensorflow.keras as K


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
        # while True:
        #     ret, frame = cap.read()
        #     cv2.rectangle(frame, (100, 150), (300, 350), (255, 255, 255), 2)
        #     # cv2.imshow('Rock Paper Scissor', frame)
        #     cv2.imshow('Rock Paper Scissor', frame[50:350, 100:400])
        #     if cv2.waitKey(1) & 0xff == ord('q'):
        #         break
        # load the model
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
        self.fail()

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