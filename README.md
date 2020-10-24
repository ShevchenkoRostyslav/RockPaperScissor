# Building an interactive Rock-Paper-Scissor game using OpenCV + Tensorflow 2.0

To play 3 rounds of the game:

```
python play.py --rounds 3
```
Details about how to generate the training data and train the DNN model can be found below. 
## Generate the training data

To generate the training data for any of labels: rock/scissor/paper, run:
```
python genLabeledData.py --label rock 
```
To quite the generator press and hold ``q`` or provide the ``--max_images`` argument.

## Model training

