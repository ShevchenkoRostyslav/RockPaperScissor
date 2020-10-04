from unittest import TestCase
import logging
import tensorflow.keras as K
from play import User
import logging
import sys
from mock import patch


logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)


class TestUser(TestCase):
    @patch('play.User._get_model_prediction', return_value=0)
    def test_choose(self, _):
        # model is not needed as prediction has been mocked out
        user = User(None)
        result = user.choose()
        self.assertEqual(result, 'rock')
