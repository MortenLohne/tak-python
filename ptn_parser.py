import copy
import sys

from tak import GameState


def is_result(word):
    return word == 'R-0' or word == 'F-0' or word == "0-R" or word == "0-F" or word == "1-0" or word == "0-1" or word == "1/2-1/2"


def add_ptn(ptn, max_plies=sys.maxsize):

    print('Reading ptn')

    spl = ptn.split("\n\n")
    headers = spl[0]
    moves = spl[1]

    # parse headers
    spl = headers.split("\n")
    size = 6
    result = '*'

    for line in spl:
        if line.startswith('[Size'):
            size = int(line[7])
        elif line.startswith('[Result'):
            result = line[9:12]

    if size != 6:
        return

    # parse moves
    all_moves = []

    for move_string in moves.split():
        if len(move_string) > 0 and not move_string.__contains__('.') and not is_result(move_string):
            all_moves.append(move_string)

    # apply upper bound of ply depth
    all_moves = all_moves[0:min(len(all_moves), max_plies)]

    # create board
    tak = GameState(size)

    positions = []

    # make all moves
    for i in range(0, len(all_moves)):
        last_tps = tak.get_tps()
        tak.move(all_moves[i])
        last_move = all_moves[i]
        positions.append(copy.deepcopy(tak))
    return positions


def main(ptn_file):

    print('Reading ptns')

    max_plies = 200

    ptn = ''

    f = open(ptn_file)
    count = f.read().count('[Site')
    print(f'Counted {count} games')
    f.close()

    ptns = 0

    positions = [];

    with open(ptn_file) as f:
        line = f.readline()
        ptn += line
        line = f.readline()
        while line:
            if line.startswith("[Event"):
                positions.extend(add_ptn(ptn, max_plies))
                ptn = ''
                ptns += 1
                print(f'Read {ptns} ptns')
            ptn += line
            line = f.readline()

    for position in positions[:10]:
        print(position.print_state())
        print(position.get_tps())
