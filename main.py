from ppadb.client import Client
from PIL import Image, ImageEnhance, ImageFilter, ImageChops, ImageDraw, ImageFont, ImageOps
from io import BytesIO
import networkx as nx
from itertools import product
import os
import dawg
import subprocess


LETTER_VALUES = {'A': 1, 'B': 4, 'C': 4, 'D': 2, 'E': 1, 'F': 4, 'G': 3, 'H': 3, 'I': 1, 'J': 10, 'K': 5, 'L': 2,
                 'M': 4, 'N': 2, 'O': 1, 'P': 4, 'Qu': 10, 'R': 1, 'S': 1, 'T': 1, 'U': 2, 'V': 5, 'W': 4, 'X': 8,
                 'Y': 3, 'Z': 10}
MULT_VALUES = {'': (1, 1), 'DL': (2, 1), 'TL': (3, 1), 'DW': (1, 2), 'TW': (1, 3)}
LENGTH_BONUS = [0, 0, 0, 0, 0, 3, 6, 10, 15, 20, 25, 30, 35, 40, 45, 50]

# load DAWG dictionary (extracted from APK)
WORDS_DAWG = dawg.CompletionDAWG()
WORDS_DAWG.load('dictionary_en.dawg')


class Board:
    def __init__(self, image):
        """  Takes a gameboard image and constructs a board object """
        self.image = image
        self.tiles = self.get_tiles()
        self.board = [self.read_tile(tile) for tile in self.tiles]
        # construct a 4x4 King's graph and label the nodes with the tile info
        self.graph = build_kings(4)
        nx.relabel_nodes(
            self.graph,
            {n: (n, *b) for n, b in zip(self.graph, self.board)},
            copy=False
        )
        self.words = sorted(self.gen_words(), key=lambda w: w.score/(len(w.string) + 1), reverse=True)

    def get_tiles(self):
        """ Chops the board image into 16 tile images and returns them as a list """
        assert self.image.size == (1080, 1092), 'Improper image dimensions'
        # hardcoded values for the boxes and spacing as image is normalised to a known size
        s = 247  # spacing between boxes
        x1, y1, x2, y2 = 43, 54, 294, 305  # initial box location
        boxes = [(x1 + s * dx, y1 + s * dy, x2 + s * dx, y2 + s * dy) for dy in range(4) for dx in range(4)]
        return [self.image.crop(box) for box in boxes]

    def read_tile(self, tile):
        """ Rudimentary OCR (optical character recognition) algorithm for reading info from a tile """
        # First detect the letter:
        # crop the tile to focus on the letter
        letter_target = tile.crop((59, 59, 192, 182))
        # apply some basic contrast and sharpness enhancements and convert to binary black / white image
        letter_target = ImageEnhance.Contrast(letter_target).enhance(3.0).filter(ImageFilter.SHARPEN)
        letter_target = letter_target.convert('1')
        # load the font of the gameboard letters (extracted from the APK)
        font = ImageFont.truetype('Eurostile-Hea.ttf', 145)
        letter_weights = self.gen_weights(letter_target, LETTER_VALUES, font)
        # find the letter with minimum weight (ie. the fewest differing pixels)
        letter = min(letter_weights)[1]

        # Next detect the multiplier:
        # crop the tile to focus on the multiplier
        mult_target = tile.crop((1, 22, 72, 53))
        # darken the image and convert to greyscale
        mult_target = ImageEnhance.Brightness(mult_target.filter(ImageFilter.SHARPEN)).enhance(0.1)
        mult_target = mult_target.convert('L')
        # enhance contrast, invert the image (so black text on white background) and convert to black / white
        mult_target = ImageEnhance.Contrast(mult_target).enhance(50)
        mult_target = ImageOps.invert(mult_target)
        mult_target = mult_target.convert('1')
        # change the font size
        font = ImageFont.truetype('Eurostile-Hea.ttf', 41)
        mult_weights = self.gen_weights(mult_target, MULT_VALUES, font)
        min_weight, min_mult = min(mult_weights)
        # if there is a close enough match, use that - otherwise use no multiplier
        mult = min_mult if min_weight < 100000 else ''

        return letter, mult

    @staticmethod
    def gen_weights(target, candidates, font):
        """ Generates weights for how much candidate strings differ from a target image when rendered with a given
        font (where smaller weights correspond to closer matches) """
        width, height = target.size
        # create a black / white image for trial and instantiate a drawing object
        trial = Image.new('1', target.size, 1)
        draw = ImageDraw.Draw(trial)
        for candidate in candidates:
            # fill the trial image with white
            trial.paste(255, (0, 0, *trial.size))
            # find the size of the candidate string and draw it centred on the trial image
            w, h = draw.textsize(candidate, font=font)
            draw.text(((width - w) / 2, (height - h) / 2), candidate, 0, font=font)
            # find the difference between the trial and target
            diff = ImageChops.difference(target, trial)
            # yield the sum of pixels values in the difference (proportional to the number of differing pixels)
            yield sum(diff.getdata()), candidate

    def gen_words(self):
        """ Generates Word objects for all valid words on the board """
        for starting_node in self.graph:
            yield from self.recursive_find([starting_node], starting_node[1])

    def recursive_find(self, current_path, current_word):
        """ Recursively yields Word objects from a current path """
        if current_word in WORDS_DAWG:
            yield Word(current_path)
        # check for neighbours of the final node that have not yet been visited
        neighbours = set(self.graph.neighbors(current_path[-1])) - set(current_path)
        for node in neighbours:
            new_path = current_path + [node]
            new_word = current_word + node[1]
            # only continue checking if there are words in dictionary that start with the current string
            if WORDS_DAWG.has_keys_with_prefix(new_word):
                yield from self.recursive_find(new_path, new_word)

    def __repr__(self):
        rows = [self.board[4 * i: 4 * (i + 1)] for i in range(4)]
        formatted_rows = ['\t'.join(f'{let} {f"({mult})" if mult else "    "}' for let, mult in row) for row in rows]
        return '\n\n'.join(formatted_rows)


class Word:
    def __init__(self, word):
        """ Takes a list of nodes in a Board object and constructs a Word object """
        self.word = word
        self.string = ''.join(let for _, let, _ in self.word)
        self.path = [node for node, _, _ in self.word]
        self.score = self.score_word()

    def score_word(self):
        """ Scores the word according to letter values, word length and applicable multipliers"""
        total_word_mult = 1
        score = 0
        for _, letter, mult in self.word:
            letter_value = LETTER_VALUES[letter]
            letter_mult, word_mult = MULT_VALUES[mult]
            total_word_mult *= word_mult
            score += letter_value * letter_mult
        score *= total_word_mult
        score += LENGTH_BONUS[len(self.word)]
        return score

    def __repr__(self):
        return f'<Word object: string={self.string!r}, score={self.score}>'


def connect_device(host='127.0.0.1', port=5037):
    """ Connects to the ADB server with given host IP and port and returns the first device (if no IP/port is given it
    will use the default ADB values) """
    # connect to the ADB server
    adb = Client(host, port)
    try:
        # take the fist device
        device = adb.devices()[0]
    except IndexError:
        # if no devices in list, throw an exception
        raise Exception('No ADB devices could be found')
    return device


def grab_board(device, challenge=False):
    """ Takes a device object and returns a cropped Pillow image object of the gameboard """
    # grab a screenshot from the device
    screencap = device.screencap()
    # open this byte array as a pillow image
    image = Image.open(BytesIO(screencap))
    width, height = image.size
    # scale the image to a known width of 1080 pixels
    if width != 1080:
        scale_factor = 1080 / width
        new_height = round(scale_factor * height)
        image = image.resize((1080, new_height))
    # crop the gameboard area, adjusting for challenge mode as necessary
    image = image.crop((0, 373 + 68 * challenge, 1080, 1465 + 68 * challenge))
    return image


def build_kings(n):
    """ Constructs an n x n King's graph, ie. a graph representing all the possible moves of a King on an n x n
    chess board. """
    # create a graph and add nodes
    graph = nx.Graph()
    graph.add_nodes_from(product(range(n), repeat=2))
    # for each node, add edges to all adjacent nodes
    for row, col in graph.nodes:
        for i, j in product([-1, 0, 1], repeat=2):
            if (i, j) == (0, 0):
                continue
            if (row + i, col + j) in graph:
                graph.add_edge((row, col), (row + i, col + j))
    return graph


def build_inputs(board, output, challenge=False):
    """ Builds a tsv file containing words with their associated score and input sequence """
    base_x, base_y = 160, 540 + 68 * challenge
    spacing = 247
    accept = 940, 240 + 68 * challenge

    entered = set()
    total_score = 0
    with open(output, 'w') as f:
        for word in board.words:
            if word.string not in entered:
                entered.add(word.string)
                total_score += word.score
                inputs = [(base_x + spacing * x, base_y + spacing * y) for y, x in word.path]
                inputs.append(accept)
                print(word.string, word.score, inputs, sep='\t', file=f)
    print('Total score:', total_score)


def monkeyrunner(inputs):
    """ Uses Monkeyrunner tool from Android Studio to enter the desired inputs from the given tsv file """
    path = os.path.dirname(os.path.abspath(__file__))
    monkeyrunner_script = os.path.expandvars(r'%localappdata%\Android\Sdk\tools\bin\monkeyrunner.bat')
    proc = subprocess.Popen([monkeyrunner_script, f'{path}\\input.py', f'{path}\\{inputs}'])
    try:
        proc.wait()
    except KeyboardInterrupt:
        proc.kill()


def play(device, challenge=False):
    """ Plays a game of Boggle """
    board = Board(grab_board(device, challenge))
    print('Found board:')
    print(board, '\n')
    print(f'Found {len({word.string for word in board.words})} unique words!')
    inputs_path = 'inputs.tsv'
    build_inputs(board, inputs_path, challenge)
    monkeyrunner(inputs_path)


if __name__ == '__main__':
    phone = connect_device()
    play(phone)
    phone.client.remote_disconnect()