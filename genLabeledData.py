import argparse
from typing import Dict
import cv2
import os


def parse_args() -> Dict:
    """Input argument parser

    :return:
    """
    parser = argparse.ArgumentParser(description='Generate data by showing rock/paper/scissor/noise to the web-cam')
    parser.add_argument('--label', type=str, choices=['rock', 'paper', 'scissor', 'noise'],
                        help='Label of the pictures to be shown', required=True)
    parser.add_argument('--max_images', type=int, default=None,
                        help='Maximal number of images to be saved')
    # Execute parse_args()
    args = parser.parse_args()
    return vars(args)


def prepare_image_path(label) -> str:
    """If does not exist - create the directory by the label name.

    :param label: name of the label
    :return:
    """
    PATH = os.getcwd()
    save_path = os.path.join(PATH, label)
    try:
        os.mkdir(save_path)
    except FileExistsError:
        pass
    return save_path


def count_existing_images(save_path: str, label: str) -> int:
    """Count existing images according to the label in save_path

    :param save_path: path where the data is stored
    :param label: image label
    :return:
    """
    return len(
        [name for name in os.listdir(save_path) if os.path.isfile(os.path.join(save_path, name)) and label in name])


def capture_labeled_images(label: str, max_imgs: int) -> None:
    """Make a photo according to the label using a primary device.

    :param label: label to be shown to the camera
    :param max_imgs: maximum allowed number of images
    :return:
    """
    save_path = prepare_image_path(label)
    cap = cv2.VideoCapture(0)
    # counter for the number of images
    n_images = count_existing_images(save_path, label)
    print('EXIST images', n_images)
    while True:
        ret, frame = cap.read()
        cv2.imshow('Get Data : ' + label, frame[50:350, 100:450])
        if cv2.waitKey(1) & 0xFF == ord(' '):
            img_path = os.path.join(save_path, f'{label}{n_images}.jpg')
            cv2.imwrite(img_path, frame[50:350, 100:450])
            print(f'{img_path} is captured')
            n_images += 1
        elif (max_imgs and n_images >= max_imgs) or cv2.waitKey(1) & 0xFF == ord('q'):
            print('Stop recording...')
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    inputs = parse_args()
    label, max_imgs = inputs['label'], inputs['max_images']
    capture_labeled_images(label, max_imgs)
    print('All the data is stored.')
