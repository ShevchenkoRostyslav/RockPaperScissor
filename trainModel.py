import glob

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import MaxPool2D, Flatten, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

from trainTestSplit import LABELS, train_test_split
import matplotlib.pyplot as plt

# This function will plot images in the form of a grid with 1 row and 5 columns where images are placed in each column.
def plotImages(images_arr):
    fig, axes = plt.subplots(1, len(images_arr), figsize=(20,20))
    if len(images_arr) > 1:
      axes = axes.flatten()
    else: axes = [axes]
    for img, ax in zip( images_arr, axes):
        ax.imshow(img)
        ax.axis('off')
    plt.tight_layout()
    plt.show()



def genericModel(base):
    model = Sequential()
    model.add(base)
    model.add(MaxPool2D())
    model.add(Flatten())
    model.add(Dense(4, activation='softmax'))
    model.compile(optimizer=Adam(),loss='categorical_crossentropy',metrics=['acc'])
    return model


if __name__ == "__main__":

    batch_size = 16
    IMG_HEIGHT = 300
    IMG_WIDTH = 300
    train_path, test_path = train_test_split(0.10, 1)
    total_train = sum([len(glob.glob(f'{train_path}/{label}/{label}*')) for label in LABELS])

    image_gen_train = ImageDataGenerator(
        rescale=1. / 255,
        rotation_range=45,
        width_shift_range=.15,
        height_shift_range=.15,
        horizontal_flip=True,
        zoom_range=0.25,
        channel_shift_range=100,
        brightness_range=[0.2, 1.0],
        shear_range=40,
        validation_split=0.2
    )

    train_data_gen = image_gen_train.flow_from_directory(batch_size=batch_size,
                                                         directory='train',
                                                         shuffle=True,
                                                         target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                         classes=LABELS,
                                                         class_mode='categorical')
    #
    # augmented_images = [train_data_gen[0][0][0] for i in range(10)]
    # plotImages(augmented_images)
    densenet = DenseNet121(include_top=False, weights='imagenet', classes=4, input_shape=(300, 300, 3))
    densenet.trainable = False
    dnet = genericModel(densenet)
    print(dnet.summary())

    checkpoint = ModelCheckpoint(
        'model.h5',
        monitor='val_acc',
        verbose=1,
        save_best_only=True,
        # save_weights_only=True,
        mode='auto'
    )
    es = EarlyStopping(patience=3)

    history= dnet.fit(
        train_data_gen,
        steps_per_epoch=total_train // batch_size,
        epochs=8,
        callbacks=[checkpoint, es]
    )
