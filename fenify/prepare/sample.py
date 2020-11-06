from fenify.helpers.image import scale_image, read_image_as_png, overlay_image
import random
import cv2
import numpy as np
from .config import MOUSE_POINTER_IMG


class BoardSample:
    def __init__(self, fen, image):
        """
        Input output pair
        
        Args:
            fen (str): The fen of the board
            image (nparr): Image of dimensions (length, length, 3) representing the image.
        """
        self.fen = fen
        self.image = image

class DetailedBoardSample(BoardSample):
    def __init__(self, fen, image, pieces_distortions, board_distortions):
        super().__init__(fen, image)
        self.pieces_distortions = pieces_distortions
        self.board_distorions = board_distortions
