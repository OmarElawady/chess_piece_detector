from os import lseek
from prepare.piece_distortion import PieceDistorter
from fenify.helpers.image import scale_image, read_image_as_png, overlay_image, convert_svg_text_to_png
import random
from .config import MOUSE_POINTER_IMG, ARROW_LINE, ARROW_CHESS_LINE
import cv2
import numpy as np

LINE_STROKE_WIDTH_400 = 7.802083492279053

def decision(prob):
    return random.random() < prob

class BoardDistorter:
    DENSITY = 0
    def __init__(self):
        pass

    def distort(self, board_image):
        board_image = board_image
        return board_image
class BoardShiftDistorter(BoardDistorter):
    DENSITY = 1
    def __init__(self):
        pass

    def distort(self, board_image):
        l = board_image.shape[0]
        board_image = np.hstack((board_image, board_image, board_image))
        board_image = np.vstack((board_image, board_image, board_image))
        x_base, y_base = l, l
        offx, offy = (random.random() - .5) * 0.025 * l, (random.random() - .5) * 0.05 * l
        x, y = int(x_base - offx), int(y_base - offy)
        return board_image[x: x + l, y: y + l]
class MousePointerBoardDistorter(BoardDistorter):
    DENSITY = 15
    def __init__(self):
        self.posx = random.random()
        self.posy = random.random()

    def distort(self, board_image):
        mouse_image = read_image_as_png(MOUSE_POINTER_IMG)
        x = self.posx * board_image.shape[0]
        y = self.posy * board_image.shape[1]
        x = int(x)
        y = int(y)
        board_image = overlay_image(board_image, mouse_image, x, y)
        return board_image

class ArrowDistorter(BoardDistorter):
    DENSITY = 2
    def __init__(self):
        self.x1, self.y1 = (0.5 + random.randint(0, 7)) / 8, (0.5 + random.randint(0, 7)) / 8
        self.x2, self.y2 = (0.5 + random.randint(0, 7)) / 8, (0.5 + random.randint(0, 7)) / 8
        self.x1 -= .5 # Origin at the center it appears
        self.y1 -= .5
        self.x2 -= .5
        self.y2 -= .5
    
    def distort(self, board_image):
        l = board_image.shape[0]
        line_image = convert_svg_text_to_png(ARROW_LINE % (LINE_STROKE_WIDTH_400 * l / 400, self.x1 * l, self.y1 * l, self.x2 * l, self.y2 * l), l, l)
        board_image = overlay_image(board_image, line_image, 0, 0)
        return board_image

class BishopArrowDistorter(BoardDistorter):
    DENSITY = 2
    def __init__(self):
        self.x1, self.y1 = random.randint(0, 7), random.randint(0, 7)
        cnt = 0
        for i in range(8):
            for j in range(8):
                if (i + j == self.x1 + self.y1 or i - j == self.x1 - self.y1) and (i != self.x1 or j != self.y1):
                    cnt += 1
        self.x2, self.y2 = -1, -1
        rem = cnt
        for i in range(8):
            for j in range(8):
                if (i + j == self.x1 + self.y1 or i - j == self.x1 - self.y1) and (i != self.x1 or j != self.y1):
                    if decision(1.0 / rem):
                        self.x2, self.y2 = i, j
                        break
                    else:
                        rem -= 1
            if self.x2 != -1:
                break
        assert self.x2 != -1
        self.x1 = (self.x1 + .5) / 8 - .5
        self.y1 = (self.y1 + .5) / 8 - .5
        self.x2 = (self.x2 + .5) / 8 - .5
        self.y2 = (self.y2 + .5) / 8 - .5

    
    def distort(self, board_image):
        l = board_image.shape[0]
        line_image = convert_svg_text_to_png(ARROW_CHESS_LINE % (LINE_STROKE_WIDTH_400 * l / 400, self.x1 * l, self.y1 * l, self.x2 * l, self.y2 * l), l, l)
        board_image = overlay_image(board_image, line_image, 0, 0)
        return board_image

class RookArrowDistorter(BoardDistorter):
    DENSITY = 2
    def __init__(self):
        self.x1, self.y1 = random.randint(0, 7), random.randint(0, 7)
        cnt = 0
        for i in range(8):
            for j in range(8):
                if (i == self.x1 or j == self.y1) and (i != self.x1 or j != self.y1):
                    cnt += 1
        self.x2, self.y2 = -1, -1
        rem = cnt
        for i in range(8):
            for j in range(8):
                if (i == self.x1 or j == self.y1) and (i != self.x1 or j != self.y1):
                    if decision(1.0 / rem):
                        self.x2, self.y2 = i, j
                        break
                    else:
                        rem -= 1
            if self.x2 != -1:
                break
        assert self.x2 != -1
        self.x1 = (self.x1 + .5) / 8 - .5
        self.y1 = (self.y1 + .5) / 8 - .5
        self.x2 = (self.x2 + .5) / 8 - .5
        self.y2 = (self.y2 + .5) / 8 - .5

    
    def distort(self, board_image):
        l = board_image.shape[0]
        line_image = convert_svg_text_to_png(ARROW_CHESS_LINE % (LINE_STROKE_WIDTH_400 * l / 400, self.x1 * l, self.y1 * l, self.x2 * l, self.y2 * l), l, l)
        board_image = overlay_image(board_image, line_image, 0, 0)
        return board_image

class KnightArrowDistorter(BoardDistorter):
    DENSITY = 2
    def __init__(self):
        self.x1, self.y1 = random.randint(0, 7), random.randint(0, 7)
        cnt = 0
        for i in range(8):
            for j in range(8):
                if abs(self.x1 - i) == 2 and abs(self.y1 - j) == 1 or abs(self.x1 - i) == 1 and abs(self.y1 - j) == 2:
                    cnt += 1
        self.x2, self.y2 = -1, -1
        rem = cnt
        for i in range(8):
            for j in range(8):
                if abs(self.x1 - i) == 2 and abs(self.y1 - j) == 1 or abs(self.x1 - i) == 1 and abs(self.y1 - j) == 2:
                    if decision(1.0 / rem):
                        self.x2, self.y2 = i, j
                        break
                    else:
                        rem -= 1
            if self.x2 != -1:
                break
        assert self.x2 != -1
        self.linex, self.liney = self.x2, self.y2
        if abs(self.x2 - self.x1) == 2:
            self.liney = self.y1
        else:
            self.linex = self.x1
        self.x1 = (self.x1 + .5) / 8 - .5
        self.y1 = (self.y1 + .5) / 8 - .5
        self.x2 = (self.x2 + .5) / 8 - .5
        self.y2 = (self.y2 + .5) / 8 - .5
        self.linex = (self.linex + .5) / 8 - .5
        self.liney = (self.liney + .5) / 8 - .5
    
    def distort(self, board_image):

        l = board_image.shape[0]
        cv2.line(board_image, (int(l * self.x1 + l / 2), int(l * self.y1 + l / 2)), (int(l * self.linex + l / 2), int(l * self.liney + l / 2)), (0, 168, 255), int(LINE_STROKE_WIDTH_400 * l // 400))

        line_image = convert_svg_text_to_png(ARROW_CHESS_LINE % (LINE_STROKE_WIDTH_400 * l / 400, self.linex * l, self.liney * l, self.x2 * l, self.y2 * l), l, l)
        board_image = overlay_image(board_image, line_image, 0, 0)
        return board_image
