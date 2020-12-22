import random
from fenify.helpers.image import scale_image
import cv2
import numpy as np
from .config import CIRCLE
from fenify.helpers.image import convert_svg_text_to_png

CIRCLE_STROKE_WIDTH_400 = 3.120833396911621

class ImageOverlay:
    def __init__(self, x, y, image):
        self.x = x
        self.y = y
        self.image = image
        self.original_length = image.shape[0]
        self.absolute_images  = []

class PieceDistorter:
    DENSITY = 0
    def __init__(self):
        self.type = "identity"

    def distort(self, image_overlay):
        return image_overlay

class RandomZoomPieceDistorter(PieceDistorter):
    DENSITY = .5
    def __init__(self, mn=95, mx=105):
        super().__init__()
        self.type = "zoom"
        self.zoom = random.uniform(mn, mx) / 100

    def distort(self, image_overlay):
        image_overlay.image = scale_image(image_overlay.image, self.zoom, cv2.INTER_LINEAR)
        return image_overlay

class RandomTransiterPieceDistorter(PieceDistorter):
    DENSITY = .5
    def __init__(self, percentage=.05):
        super().__init__()
        self.type = "transition"
        self.shiftx = random.uniform(-percentage, percentage)
        self.shifty = random.uniform(-percentage, percentage)

    def distort(self, image_overlay):
        image_overlay.x += self.shiftx * image_overlay.image.shape[0]
        image_overlay.y += self.shifty * image_overlay.image.shape[1]
        return image_overlay

class LichessOverlayDistorter(PieceDistorter):
    DENSITY = .05
    def __init__(self):
        self.type = "lichess_overlay"

    def distort(self, image_overlay):
        l = image_overlay.original_length
        lichess_png_image = np.zeros((l, l, 4))
        lichess_png_image[:l, :l] = [0, 199, 155, 102]
        image_overlay.absolute_images.append(lichess_png_image)
        return image_overlay

class YellowOverlayDistorter(PieceDistorter):
    DENSITY = .05
    def __init__(self):
        self.type = "yellow_overlay"

    def distort(self, image_overlay):
        l = image_overlay.original_length
        lichess_png_image = np.zeros((l, l, 4))
        lichess_png_image[:l, :l] = [51, 255, 255, 128]
        image_overlay.absolute_images.append(lichess_png_image)
        return image_overlay

class RedOverlayDistorter(PieceDistorter):
    DENSITY = .05
    def __init__(self):
        self.type = "red_overlay"

    def distort(self, image_overlay):
        l = image_overlay.original_length
        lichess_png_image = np.zeros((l, l, 4))
        lichess_png_image[:l, :l] = [50, 42, 244, 230]
        image_overlay.absolute_images.append(lichess_png_image)
        return image_overlay

class CircleDistorter(PieceDistorter):
    DENSITY = .05
    def __init__(self):
        self.type = "circle"

    def distort(self, image_overlay):
        l = image_overlay.original_length
        circle_radius = l * 4 / 9
        png_image = convert_svg_text_to_png(CIRCLE % (CIRCLE_STROKE_WIDTH_400 / 400 * l, 0, 0, circle_radius), l, l)
        image_overlay.absolute_images.append(png_image)
        return image_overlay
