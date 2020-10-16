import cv2

def prepImg(pth):
    """Prepare an image for recognition.

    :param pth:
    :return:
    """
    return cv2.resize(pth, (300, 300)).reshape(1, 300, 300, 3) / 255.