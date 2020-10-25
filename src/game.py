import time
import cv2
from typing import Union
import os

import random
import numpy as np
from abc import abstractmethod
from src.trainTestSplit import NUMBER_LABEL_MATCH
from src.utils import prepImg
import logging

LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.DEBUG)

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


class Player:

    def __init__(self, name):
        self.score = 0
        self.name = name

    @abstractmethod
    def choose(self):
        pass

    def scored(self):
        self.score += 1

    def won(self):
        return f'{self.name} WON the match!'


class Bot(Player):

    def __init__(self):
        super().__init__(name='Bot')

    def choose(self):
        rndm_choice = random.randrange(0, 3)
        choice_name = RockPaperScissorGame.OPTIONS[rndm_choice]
        return choice_name


class User(Player):

    def __init__(self, trained_model, name: str = 'you'):
        """AI empowered user.

        """
        super().__init__(name=name)
        self.trained_model = trained_model
        self.video_cap = cv2.VideoCapture(0)

    def choose(self):
        prediction = self._get_model_prediction()
        choice = NUMBER_LABEL_MATCH[prediction]
        LOGGER.debug(f'User "{self.name}" choose: {choice}')
        return choice

    def _get_model_prediction(self):
        _, frame = self.video_cap.read()
        prediction_probs = self.trained_model.predict(prepImg(frame[100:420, 100:420]))
        LOGGER.debug(f'User "{self.name}" predicted probabilities: {prediction_probs}')
        max_prediction = np.argmax(prediction_probs)
        return max_prediction


class RockPaperScissorGame:
    OPTIONS = ['rock', 'paper', 'scissor']
    RULES = {'rock': 'scissor', 'scissor': 'paper', 'paper': 'rock'}

    def __init__(self, player1: Player, player2: Player, num_rounds: int):
        self.player1 = player1
        self.player2 = player2
        self.rounds = num_rounds
        self.current_round = 0

    def play(self):
        cap = cv2.VideoCapture(0)
        for round in range(1, self.rounds+1):
            LOGGER.info('ROUND:', round)
            self.current_round = round
            self.round(cap)
        winner = self.check_winner()
        # close all the opened windows
        cv2.destroyAllWindows()
        self.declare_winner(winner, cap)

        # close the camera
        cap.release()
        # close all the opened windows
        cv2.destroyAllWindows()

    def declare_winner(self, player: Player, cap) -> None:
        """Winner declaration

        :param player: winner
        :param cap: VideoCapture
        :return:
        """
        start = time.time()
        while True:
            ret, frame = cap.read()
            cnt = int(time.time() - start)
            winner_message = 'Equal scores'
            if player:
                winner_message = player.won()
            frame = cv2.putText(frame, winner_message, (500, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (250, 250, 0), 2, cv2.LINE_AA)
            self.visualize_camera_frame(frame, with_ractangle=False)
            if cv2.waitKey(1) & 0xff == ord('q'):
                break
            if cnt >=5:
                break

    def round(self, cap):
        """One round of the game.

        :param cap: VideoCapture
        :return:
        """
        start = time.time()
        played = False
        player1_choice, player2_choice = '', ''
        while True:
            ret, frame = cap.read()
            cnt = int(time.time() - start)
            # display the countdown
            frame = cv2.putText(frame, str(cnt), (550, 100), cv2.FONT_HERSHEY_SIMPLEX, 3, (250, 250, 0), 2, cv2.LINE_AA)
            # display the round
            frame = cv2.putText(frame, f'Round: {self.current_round}', (550, 150), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                (250, 250, 0), 2, cv2.LINE_AA)
            if cnt >= 5:
                # players make a move
                if not played:
                    player1_choice = self.player1.choose()
                    player2_choice = self.player2.choose()
                    played = self.update_scores(player1_choice, player2_choice)
            frame = self.visualize_choices(frame, player1_choice, player2_choice)
            self.visualize_camera_frame(frame)
            if cv2.waitKey(1) & 0xff == ord('q'):
                break
            if cnt > 12 and played:
                break
        return 0

    def visualize_camera_frame(self, frame, with_ractangle: bool=True) -> None:
        """Show the video-frame image together with the current player scores and a rectangle for the user-hand.

        :param frame:
        :return:
        """
        # show current scores
        frame = cv2.putText(frame,
                            f"{self.player2.name} : {self.player2.score}", (950, 90), cv2.FONT_HERSHEY_SIMPLEX, 1,
                            (250, 250, 0), 2, cv2.LINE_AA)
        frame = cv2.putText(frame,
                            f"{self.player1.name} : {self.player1.score}", (100, 90), cv2.FONT_HERSHEY_SIMPLEX, 1,
                            (250, 250, 0), 2, cv2.LINE_AA)
        # rectangle for the user-hand
        if with_ractangle: cv2.rectangle(frame, (100, 100), (420, 420), (255, 255, 255), 2)
        # display the web-camera frame
        cv2.imshow('Rock Paper Scissor', frame)

    def visualize_choices(self, frame, choice1: str, choice2: str):
        """Update the video-frame with the user choices.

        :param frame:
        :param choice1: choice of the first player
        :param choice2: choice of the second player
        :return: updated video-frame
        """
        frame = cv2.putText(frame, f"{self.player1.name} played : {choice1}", (100, 450), cv2.FONT_HERSHEY_SIMPLEX, 1,
                            (250, 250, 0), 2, cv2.LINE_AA)
        frame = cv2.putText(frame, f"{self.player2.name} played : {choice2}", (800, 450), cv2.FONT_HERSHEY_SIMPLEX, 1,
                            (250, 250, 0), 2, cv2.LINE_AA)
        return frame

    def update_scores(self, choice1: str, choice2: str) -> bool:
        """Update the players scores according to the rules of the game.

        :param choice1: choice of the player1.
        :param choice2: choice of the player2.
        :return:
        """
        try:
            if choice1 == RockPaperScissorGame.RULES[choice2]:
                self.player2.scored()
        except KeyError:
            raise ValueError(f'"{choice2}" is not a valid option {RockPaperScissorGame.OPTIONS}.')
        try:
            if choice2 == RockPaperScissorGame.RULES[choice1]:
                self.player1.scored()
        except KeyError:
            raise ValueError(f'"{choice1}" is not a valid option {RockPaperScissorGame.OPTIONS}.')
        return True

    def check_winner(self) -> Union[Player, None]:
        """Determine the game winner after playing all the rounds.

        :return: either a player object or None if both players have same scores
        """
        if self.player1.score > self.player2.score:
            return self.player1
        elif self.player1.score == self.player2.score:
            return None
        else:
            return self.player2