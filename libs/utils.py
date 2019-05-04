import numpy as np
from keras.applications import vgg16
from keras.preprocessing.image import load_img, img_to_array


def get_dimensions(path, target_height):
    """
    Calculate dimensions to resize all three pictures to the same dimensions,
    based on the input height and keeping the aspect ratio
    """
    width, height = load_img(path).size
    target_width = int(width * target_height / height)
    return (target_height, target_width)


def preprocess_image(path, target_dims):
    img = load_img(path, target_size=target_dims)
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = vgg16.preprocess_input(img)
    return img


def deprocess_image(img):
    # Center to zero by removing the mean pixel value (required for VGG models)
    img[:, :, 0] += 103.939
    img[:, :, 1] += 116.779
    img[:, :, 2] += 123.68
    # Reverse channel order from BGR to RGB
    img = img[:, :, ::-1]
    img = np.clip(img, 0, 256).astype('uint8')
    return img
