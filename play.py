import random
import time

import tensorflow.keras as K
import cv2
import numpy as np
import os
from abc import abstractmethod
from trainTestSplit import NUMBER_LABEL_MATCH
import logging
LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.DEBUG)

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

def prepImg(pth):
    return cv2.resize(pth, (300, 300)).reshape(1, 300, 300, 3)/255.


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
        LOGGER.info(f'{self.name} WON the match!')


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
        for round in range(self.rounds):
            LOGGER.info('ROUND:', round)
            self.current_round = round
            self.round(cap)
        winner = self.check_winner()

        # close the camera
        cap.release()
        # close all the opened windows
        cv2.destroyAllWindows()

    def round(self, cap):
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
            self.update_camera_frame(frame)
            print(cnt)
            if cv2.waitKey(1) & 0xff == ord('q'):
                break
            if cnt > 12 and played:
                break
        return 0  # self.update_scores(player1_choice, player2_choice)

    def update_camera_frame(self, frame):
        # show current scores
        frame = cv2.putText(frame,
                            f"{self.player2.name} : {self.player2.score}", (950, 90), cv2.FONT_HERSHEY_SIMPLEX, 1,
                            (250, 250, 0), 2, cv2.LINE_AA)
        frame = cv2.putText(frame,
                            f"{self.player1.name} : {self.player1.score}", (100, 90), cv2.FONT_HERSHEY_SIMPLEX, 1,
                            (250, 250, 0), 2, cv2.LINE_AA)
        # ractangle for the user-hand
        cv2.rectangle(frame, (100, 100), (420, 420), (255, 255, 255), 2)
        # display the web-camera frame
        cv2.imshow('Rock Paper Scissor', frame)

    def visualize_choices(self, frame, choice1: str, choice2: str):
        frame = cv2.putText(frame, f"{self.player1.name} played : {choice1}", (100, 450), cv2.FONT_HERSHEY_SIMPLEX, 1,
                            (250, 250, 0), 2, cv2.LINE_AA)
        frame = cv2.putText(frame, f"{self.player2.name} played : {choice2}", (800, 450), cv2.FONT_HERSHEY_SIMPLEX, 1,
                            (250, 250, 0), 2, cv2.LINE_AA)
        return frame

    def update_scores(self, choice1: str, choice2: str):
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

    def check_winner(self):
        """Determine the game winner after playing all the rounds.

        :return: either a player object or None if both players have same scores
        """
        if self.player1.score > self.player2.score:
            return self.player1
        elif self.player1.score == self.player2.score:
            return None
        else:
            return self.player2

if __name__ == '__main__':
    trained_model = K.models.load_model('model_weights/model.h5')
    user = User(trained_model)
    bot = Bot()
    game = RockPaperScissorGame(user, bot, 3)
    game.play()

#
# prediction = trained_model.predict(prepImg(frame[50:350, 100:400]))
# print(prediction)
# max_prediction = np.argmax(trained_model.predict(prepImg(frame[50:350, 100:400])))
# print(max_prediction)
# pred = arr_to_shape[np.argmax(trained_model.predict(prepImg(frame[50:350, 100:400])))]
# print(pred)
#
#
#
#
#
# trained_model = K.models.load_model('model_weights/model.h5')
#
# def prepImg(pth):
#     # image = K.preprocessing.image.load_img(pth)
#     # input_arr = K.preprocessing.image.img_to_array(image)
#     # input_arr = np.array([input_arr])  # Convert single image to a batch.
#     # return input_arr
#     # # predictions = model.predict(input_arr)
#     return cv2.resize(pth, (300, 300)).reshape(1, 300, 300, 3)/255.
#     # return pth.reshape(1, 300, 300, 3)
#
# def updateScore(play,bplay,p,b):
#     winRule = {'rock':'scissor','scissor':'paper','paper':'rock'}
#     if play == bplay:
#         return p,b
#     elif bplay == winRule[play]:
#         return p+1,b
#     else:
#         return p,b+1
#
# if __name__ == '__main__':
#
#     # print(trained_model.weights)
#
#     cap = cv2.VideoCapture(0)
#     options = ['rock', 'paper', 'scissor']
#     winRule = {'rock': 'scissor', 'scissor': 'paper', 'paper': 'rock'}
#     rounds = 0
#     botScore = 0
#     playerScore = 0
#     NUM_ROUNDS = 2
#     bplay = ""
#     shape_to_label = {'rock': np.array([1., 0., 0.]), 'paper': np.array([0., 1., 0.]),
#                       'scissor': np.array([0., 0., 1.])}
#     arr_to_shape = {np.argmax(shape_to_label[x]): x for x in shape_to_label.keys()}
#
#     for rounds in range(NUM_ROUNDS):
#         pred = ""
#         for i in range(90):
#             ret, frame = cap.read()
#             print(i, i // 20)
#
#             # Countdown
#             if i // 20 < 3:
#                 frame = cv2.putText(frame, str(i // 20 + 1), (320, 100), cv2.FONT_HERSHEY_SIMPLEX, 3, (250, 250, 0), 2,
#                                     cv2.LINE_AA)
#
#             # Prediction
#             elif i / 20 < 3.5:
#                 # pred = [np.argmax(trained_model.predict(frame[50:350, 100:450]))]
#                 # pred = [np.argmax(trained_model.predict(prepImg(frame[50:350,100:450])))]
#                 prediction = trained_model.predict(prepImg(frame[50:350,100:400]))
#                 print(prediction)
#                 max_prediction = np.argmax(trained_model.predict(prepImg(frame[50:350,100:400])))
#                 print(max_prediction)
#                 pred = arr_to_shape[np.argmax(trained_model.predict(prepImg(frame[50:350,100:400])))]
#                 print(pred)
#
#             # Get Bots Move
#             elif i / 20 == 3.5:
#                 bplay = random.choice(options)
#                 print(pred, bplay)
#
#             # Update Score
#             elif i // 20 == 4:
#                 playerScore, botScore = updateScore(pred, bplay, playerScore, botScore)
#                 break
#
#             cv2.rectangle(frame, (100, 150), (300, 350), (255, 255, 255), 2)
#             frame = cv2.putText(frame, "Player : {}      Bot : {}".format(playerScore, botScore), (120, 400),
#                                 cv2.FONT_HERSHEY_SIMPLEX, 1, (250, 250, 0), 2, cv2.LINE_AA)
#             frame = cv2.putText(frame, pred, (150, 140), cv2.FONT_HERSHEY_SIMPLEX, 1, (250, 250, 0), 2, cv2.LINE_AA)
#             frame = cv2.putText(frame, "Bot Played : {}".format(bplay), (300, 140), cv2.FONT_HERSHEY_SIMPLEX, 1,
#                                 (250, 250, 0), 2, cv2.LINE_AA)
#             cv2.imshow('Rock Paper Scissor', frame)
#             if cv2.waitKey(1) & 0xff == ord('q'):
#                 break
#
