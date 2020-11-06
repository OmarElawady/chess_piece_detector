import cv2
import cairosvg
import numpy as np


def convert_svg_text_to_png(text, width, height):
    """
        Constructs png image from its svg representation
    """
    png = cairosvg.svg2png(text, output_width=width, output_height=height)
    encoded = np.frombuffer(png, dtype=np.uint8)
    png_image = cv2.imdecode(encoded, cv2.IMREAD_UNCHANGED)
    return png_image


def read_svg(path, width=1024, height=1024):
    """
        Reads an svg image and returns it as png with the given width and height
    """
    f = open(path, "r")
    contents = f.read()
    f.close()
    image = cairosvg.svg2png(contents, output_width=width, output_height=height)
    np_arr = np.frombuffer(image, dtype=np.uint8)
    png_image = cv2.imdecode(np_arr, cv2.IMREAD_UNCHANGED)
    return pngify(png_image)


def gray_to_rgb(img):
    """
        receive (height, width) numpy array representing a gray image and duplicate it 3 times along a new axis
    """
    assert img.ndim == 2
    expanded = np.expand_dims(img, 2)
    repeated = np.repeat(expanded, 3, 2)
    return repeated


def image_to_gray(img):
    """
        Returns the 2 channel gray version of the passed image
        img: numpy array of ndim 2 or 3 with the 3rd dimension containing 3 or 4 channels
    """
    if img.ndim == 2:
        return img
    elif img.ndim == 3 and img.shape[2] == 3:
        return np.mean(img, axis=2)
    elif img.ndim == 3 and img.shape[2] == 4:
        return np.mean(img[:, :, :3], axis=2) * img[:, :, 3]
    else:
        assert False and "Trying to gray an image of suspicious size " + str(img.shape)


def flip_image(img):
    """
        Returns a vertically flipped version of img
    """
    return cv2.flip(img, 0)


def add_alpha_channel(img):
    """
        Adds an alpha channel to the given image to be completely opaque.
        img: numpy array of shape (height, width, 3)
    """
    assert img.ndim == 3 and img.shape[2] == 3
    alpha_channel = np.zeros_like(img[:, :, 0:1])
    return np.concatenate((img, alpha_channel), axis=2)


def remove_alpha_channel(img):
    """
        Adds an alpha channel to the given image to be completely opaque.
        img: numpy array of shape (height, width, 3)
    """
    assert img.ndim == 3 and img.shape[2] == 4
    return img[:, :, :3]


def pngify(img):
    """
        Return a png image version of the passed image
        Convert to RGB if it's in gray
        Add an alpha channel if it doesn't exist
    """
    assert img.ndim == 2 or img.ndim == 3
    colored = img
    if img.ndim == 2:
        colored = gray_to_rgb(img)
    alphad = colored
    if colored.shape[2] == 3:
        alphad = add_alpha_channel(colored)
    return alphad


def jpgify(img):
    """
        Return a jpg image version of the passed image
        Convert to RGB if it's in gray
        Remove the alpha channel if it exists
    """
    assert img.ndim == 2 or img.ndim == 3
    colored = img
    if img.ndim == 2:
        colored = gray_to_rgb(img)
    dealphad = colored
    if colored.shape[2] == 4:
        dealphad = remove_alpha_channel(colored)
    return dealphad


def read_image_as_png(path):
    """
        Read the image at the given path and converts it to png
    """
    if path.endswith(".svg"):
        return read_svg(path)
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    return pngify(img)


def read_image_as_jpg(path):
    """
        Returns a 3 channel colored image at the given path
    """
    if path.endswith(".svg"):
        return jpgify(read_svg(path))
    img = cv2.imread(path)
    return img


def read_image_as_gray(path):
    """
        Returns a 2 channel gray image at the given path
    """
    if path.endswith(".svg"):
        return image_to_gray(read_svg(path))
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    return img


def write_image(path, img):
    """
        Save the given image at the given path
    """
    cv2.imwrite(path, img)


def resize_image(img, shape, interpolation=cv2.INTER_NEAREST):
    """
        Resize img to be of size shape
        shape: (width, height)
    """
    return cv2.resize(img, shape, interpolation=interpolation)


def scale_image(img, factor, interpolation=cv2.INTER_NEAREST):
    """
        Scale the img by the given factor
    """
    new_shape = (round(img.shape[1] * factor), round(img.shape[0] * factor))
    return resize_image(img, new_shape, interpolation=interpolation)


def overlay_image(jpg_image, png_image, x, y):
    """
        MODIFIES jpg_image
        overlays a png image onto a jpg image
    """
    assert jpg_image.ndim == 3 and jpg_image.shape[2] == 3 and png_image.ndim == 3 and png_image.shape[2] == 4
    if x < 0:
        png_image = png_image[-x:]
        x = 0
    if y < 0:
        png_image = png_image[:, -y:]
        y = 0
    if x + png_image.shape[0] > jpg_image.shape[0]:
        png_image = png_image[: jpg_image.shape[0] - x]
    if y + png_image.shape[1] > jpg_image.shape[1]:
        png_image = png_image[:, : jpg_image.shape[1] - y]
    alpha_channel = np.repeat(png_image[:, :, 3:4] / 255, 3, axis=2)
    height, width = png_image.shape[0], png_image.shape[1]
    jpg_image[x : x + height, y : y + width] = (
        alpha_channel * png_image[:, :, :3] + (1 - alpha_channel) * jpg_image[x : x + height, y : y + width]
    )
    return jpg_image