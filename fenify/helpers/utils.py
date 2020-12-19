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
