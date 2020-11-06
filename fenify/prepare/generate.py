from .sample import BoardSample
from .config import BOARDS_DIR, PIECES_DIR
from helpers.image import read_image_as_png, read_image_as_jpg, overlay_image
import os
import random
from glob import glob
from .piece_distortion import RandomTransiterPieceDistorter, RandomZoomPieceDistorter, ImageOverlay, LichessOverlayDistorter, RedOverlayDistorter, YellowOverlayDistorter, CircleDistorter
from .baord_distortion import MousePointerBoardDistorter, ArrowDistorter, BishopArrowDistorter, RookArrowDistorter, KnightArrowDistorter


def random_piece():
    pieces_kinds = ["pawn", "rook", "knight", "bishop", "king", "queen"]
    pieces_colors = ["black", "white"]
    pieces = []
    for kind in pieces_kinds:
        for color in pieces_colors:
            pieces.append((kind, color))
    return random.choice(pieces)


def generate_random_board():
    piece_list = []
    for i in range(64):
        if random.randint(0, 12) == 0:
            piece_list.append(None)
        else:
            piece_list.append(random_piece())
    return piece_list

def piece_to_char(piece):
    if piece.lower() == "knight":
        return "n"
    else:
        return piece[0]


def piece_pairs_to_list(piece_pairs):
    res = []
    for p in piece_pairs:
        if p is None:
            res.append("")
        else:
            piece_char = piece_to_char(p[0])
            if p[1] == "white":
                piece_char = piece_char.upper()
            res.append(piece_char)
    return res

def fen_to_piece_list(fen):
    piece_list = []
    cleaned = fen.replace("/", "")
    for e in cleaned:
        if e.isdigit():
            x = int(e)
            for q in range(x):
                piece_list.append("")
        else:
            piece_list.append(e)
    assert len(piece_list) == 64
    return piece_list


def piece_list_to_fen(piece_list):
    fen = ""
    last_empty_run = 0
    for i, e in enumerate(piece_list):
        if e == "":
            last_empty_run += 1
        else:
            if last_empty_run != 0:
                fen += str(last_empty_run)
                last_empty_run = 0
            fen += e
        if i % 8 == 7:
            if last_empty_run:
                fen += str(last_empty_run)
                last_empty_run = 0
            fen += "/"
    return fen[:-1]

def get_piece_image(pieces_type, piece, color):
    file_name_prefix = f"{piece}.{color}"
    pieces_dir = os.path.join("data/pieces", pieces_type)
    file_name = glob(os.path.join(pieces_dir, file_name_prefix) + '*')
    if not file_name:
        raise ValueError(f"Can't find {piece} of color {color} inside the pieces type {pieces_type}")
    file_name = file_name[0]
    return read_image_as_png(file_name)

def get_board_image(board_type):
    board_dir = os.path.join("data/boards", board_type)
    return read_image_as_jpg(board_dir)

def list_pieces_type():
    types = glob(os.path.join(PIECES_DIR, "*"))
    prefix_length = len(PIECES_DIR)
    if PIECES_DIR[-1] != '/':
        prefix_length += 1
    for i in range(len(types)):
        types[i] = types[i][prefix_length:]
    return types

def list_board_types():
    types = glob(os.path.join(BOARDS_DIR, "*"))
    prefix_length = len(BOARDS_DIR)
    if BOARDS_DIR[-1] != '/':
        prefix_length += 1
    for i in range(len(types)):
        types[i] = types[i][prefix_length:]
    return types

def list_pieces_distortions():
    return [RandomZoomPieceDistorter, RandomTransiterPieceDistorter, LichessOverlayDistorter, RedOverlayDistorter, YellowOverlayDistorter, CircleDistorter]

def list_boards_distortions():
    return [MousePointerBoardDistorter, ArrowDistorter, BishopArrowDistorter, RookArrowDistorter, KnightArrowDistorter]

def decision(prob):
    return random.random() < prob

def apply_pieces_distortions(pieces_overlay_images, pieces_distortions):
    for i in range(len(pieces_overlay_images)):
        for dist in pieces_distortions[i]:
            pieces_overlay_images[i] = dist.distort(pieces_overlay_images[i])
    return pieces_overlay_images

def apply_board_distortions(board_image, board_distortions):
    for dist in board_distortions:
        board_image = dist.distort(board_image)
    return board_image

def overlay_pieces(board_image, overlays):
    length = board_image.shape[0]
    for i, ov in enumerate(overlays):
        if ov is None:
            continue
        x = i // 8
        y = i % 8
        absx = x * length // 8
        absy = y * length // 8
        posx = absx + ov.x
        posy = absy + ov.y
        posx = int(posx)
        posy = int(posy)
        for absolute_image in ov.absolute_images:
            board_image = overlay_image(board_image, absolute_image, absx, absy)
            print(absolute_image)
        board_image = overlay_image(board_image, ov.image, posx, posy)
    return board_image




def generate_board_sample():
    availabale_pieces_distortions = list_pieces_distortions()
    availabale_boards_distortions = list_boards_distortions()
    pieces_distortions = [[] for _ in range(64)]
    board_distortions = []
    for dist in availabale_boards_distortions:
        for _ in range(dist.DENSITY):
            board_distortions.append(dist())
    board_pieces = generate_random_board()
    for i in range(64):
        for dist in availabale_pieces_distortions:
            if board_pieces[i] and decision(dist.DENSITY):
                pieces_distortions[i].append(dist())
    pieces_overlay_images = []
    pieces_type = random.choice(list_pieces_type())
    board_image =  get_board_image(random.choice(list_board_types()))
    for piece in board_pieces:
        if piece is None:
            pieces_overlay_images.append(piece)
        else:
            piece_image = get_piece_image(pieces_type, piece[0], piece[1])
            ov = ImageOverlay(0, 0, piece_image)
            pieces_overlay_images.append(ov)
    print(board_pieces)
    distorted_pieces_overlay_images = apply_pieces_distortions(pieces_overlay_images, pieces_distortions)
    overlay_pieces(board_image, distorted_pieces_overlay_images)
    board_image = apply_board_distortions(board_image, board_distortions)
    return board_image