import glob
import os
from pathlib import Path
import shutil
import random
from typing import Tuple

LABELS = ['rock', 'paper', 'scissor', 'noise']
NUMBER_LABEL_MATCH = { idx: name for idx, name in enumerate(LABELS)}
PATH = os.getcwd()


def create_folder(dataset_path: str) -> str:
    """Create an empty clear folder according to the path.

    :param dataset_path: absolute path to the folder
    :return: dataset_path
    """
    path = Path(dataset_path)
    try:
        # clean the old training dir
        shutil.rmtree(path)
    except FileNotFoundError:
        pass
    path.mkdir()
    return dataset_path


def move_label(label: str, src_dir: str, dst_dir: str, number: int, random_seed: int = None) -> None:
    """Move number of images according to label from src_dir to dst_dir.

    :param label: name of the label
    :param src_dir: absolute path to the source directory with images
    :param dst_dir: absolute path to the destination directory
    :param number: number of images to transfer
    :param random_seed: random seed to fix or None if not
    :return:
    """
    if random_seed:
        random.seed(random_seed)
    to_be_moved = random.sample(glob.glob(f"{src_dir}/{label}*"), int(number))
    for img in to_be_moved:
        dst = os.path.join(dst_dir, os.path.basename(img))
        shutil.move(img, dst)


def copy_label(label: str, src_dir: str, dst_dir: str) -> int:
    """Copy images according to label from src_dir to dst_dir.

    :param label: name of the label
    :param src_dir: absolute path to the source directory with images
    :param dst_dir: absolute path to the destination directory
    :return: number of images that has been transferred
    """
    imgs = glob.glob(f'{src_dir}/{label}*')
    for img in imgs:
        dst = os.path.join(dst_dir, os.path.basename(img))
        shutil.copy(img, dst)
    return len(imgs)


def train_test_split(test_frac: float, with_noise: bool=True, random_seed: int = None) -> Tuple[str, str]:
    """Split the images into training and test according to the test_frac.

    :param test_frac: fraction of the images to use in the test dataset [0:1]
    :param with_noise: whether add noise data to the train/test folders
    :param random_seed: random seed to fix or None if not
    :return: path to the training and test datasets
    """
    # clean directories for test and train images
    train_path = create_folder(os.path.join(PATH, 'train'))
    test_path = create_folder(os.path.join(PATH, 'test'))
    for label in LABELS:
        # remove noise if specified
        if not with_noise and label == 'noise': continue
        src = os.path.join(PATH, label)
        train_dst = create_folder(os.path.join(train_path, label))
        # copy images to training folder
        total_imgs = copy_label(label, src, train_dst)
        # randomly choose test_frac of images from train folder into the test folder
        test_dst = create_folder(os.path.join(test_path, label))
        test_imgs = int(total_imgs * test_frac)
        move_label(label, train_dst, test_dst, test_imgs, random_seed)
        print(f'For label **{label}** there are {total_imgs - test_imgs} train images and {test_imgs} test images.')
    return train_path, test_path


if __name__ == '__main__':
    train_test_split(0.10, 1)
