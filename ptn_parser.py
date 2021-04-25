import sys

from tak import GameState


def is_result(word):
    return word == 'R-0' or word == 'F-0' or word == "0-R" or word == "0-F"


def add_ptn(ptn, max_plies=sys.maxsize):

    print('Reading ptn')

    spl = ptn.split("\n\n")
    headers = spl[0]
    moves = spl[1]

    # parse headers
    spl = headers.split("\n")
    size = 0
    result = '*'

    for line in spl:
        if line.startswith('[Size'):
            size = int(line[7])
        elif line.startswith('[Result'):
            result = line[9:12]

    if size != 6:
        return

    # parse moves
    spl = moves.split("\n")
    all_moves = []

    for row in spl:
        if len(row) == 0:
            continue
        two_ply = row.split(" ")
        if not is_result(two_ply[0]):
            all_moves.append(two_ply[1])

        if len(two_ply) > 2:
            all_moves.append(two_ply[2])

    # apply upper bound of ply depth
    all_moves = all_moves[0:min(len(all_moves), max_plies)]

    # create board
    tak = GameState(size)

    # make all moves
    for i in range(0, len(all_moves)):
        last_tps = tak.get_tps()
        tak.move(all_moves[i])
        last_move = all_moves[i]
        print(f'Last_move {last_move}, result {result}, tps {tak.get_tps()}')


def main(ptn_file):

    print('Reading ptns')

    max_plies = 200

    ptn = ''

    f = open(ptn_file)
    count = f.read().count('[Site')
    print(f'Counted {count} games')
    f.close()

    with open(ptn_file) as f:
        line = f.readline()
        ptn += line
        line = f.readline()
        while line:
            if line.startswith("[Site"):
                add_ptn(ptn, max_plies)
                ptn = ''
            ptn += line
            line = f.readline()
