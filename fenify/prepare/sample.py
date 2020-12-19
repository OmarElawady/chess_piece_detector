from fenify.helpers.image import scale_image, read_image_as_png, overlay_image
import random
import cv2
import numpy as np
from .config import MOUSE_POINTER_IMG


class BoardSample:
    def __init__(self, image, fen, pieces_type=None, board_type=None, pieces_distortions=None, board_distortions=None):
        """
        Input output pair
        
        Args:
            fen (str): The fen of the board
            image (nparr): Image of dimensions (length, length, 3) representing the image.
        """
        self.fen = fen
        self.image = image
        self.pieces_type = pieces_type
        self.board_type = board_type
        self.pieces_distortions = pieces_distortions
        self.board_distortions = board_distortions

class PieceSample:
    def __init__(self, image, piece_type, pieces_type=None, board_type=None, square_color=None, distortions=None):
        self.image = image
        self.piece_type = piece_type
        self.pieces_type = pieces_type
        self.board_type = board_type
        self.square_color = square_color
        self.distortions = None
