import re
import string


def pre_engine(s):
    """ Clean the message sent to the engine """

    s = re.sub("[^\t\n -~]", "", s)
    s = s.split("#")[0]
    s = s.replace("\t", " ")
    return s


def gtp_boolean(b):
    return "true" if b else "false"


def gtp_list(l):
    return "\n".join(l)


def gtp_color(color):
    return { BLACK: "B", WHITE: "W" }[color]


def coord_to_gtp(coord, board_size):
    """ From 1d coord (0 for position 0,0 on the board) to A1 """

    if coord == board_size ** 2:
        return "pass"
    return "{}{}".format("ABCDEFGHJKLMNOPQRSTYVWYZ"[int(coord % board_size)],\
                int(board_size - coord // board_size))


def gtp_to_coord(gtp_coord, board_size):
    """ From something like A1 to 1d coord like 0 (for 0,0 in 1d) """

    ## 97 to convert back to the position in the alphabet, then - 1 for index
    coord = gtp_coord.split()[1]
    if coord == "pass":
        return board_size ** 2

    x = "ABCDEFGHJKLMNOPQRSTYVWYZ".index(coord[0]) + 1
    y = board_size - int(coord[1])
    final_coord = y * board_size + x - 1
    return final_coord 


def gtp_move(color, vertex):
    return " ".join([gtp_color(color), gtp_vertex(vertex)])


def parse_message(message):
    """ Parse the command sent to the agent """

    message = pre_engine(message).strip()
    first, rest = (message.split(" ", 1) + [None])[:2]
    if first.isdigit():
        message_id = int(first)
        if rest is not None:
            command, arguments = (rest.split(" ", 1) + [None])[:2]
        else:
            command, arguments = None, None
    else:
        message_id = None
        command, arguments = first, rest

    return message_id, command, arguments


WHITE = 1
BLACK = -1
EMPTY = 0

PASS = (0, 0)
RESIGN = "resign"


def parse_color(color):
    if color.lower() in ["b", "black"]:
        return BLACK
    elif color.lower() in ["w", "white"]:
        return WHITE
    else:
        return False


MIN_BOARD_SIZE = 7
MAX_BOARD_SIZE = 19


def format_success(message_id, response=None):
    if response is None:
        response = ""
    else:
        response = " {}".format(response)
    if message_id:
        return "={}{}\n\n".format(message_id, response)
    else:
        return "={}\n\n".format(response)


def format_error(message_id, response):
    if response:
        response = " {}".format(response)
    if message_id:
        return "?{}{}\n\n".format(message_id, response)
    else:
        return "?{}\n\n".format(response)


class Engine:

    def __init__(self, game, komi=7.5, board_size=19, version="0.2", name="AlphaGo"):
        self.board_size = board_size
        self.komi = komi
        self._game = game
        self._name = name
        self._version = version
        self.disconnect = False
        self.known_commands = [
            field[4:] for field in dir(self) if field.startswith("cmd_")]

    def send(self, message):
        message_id, command, arguments = parse_message(message)
        if command in self.known_commands:
            return format_success(
                message_id, getattr(self, "cmd_" + command)(arguments))
        else:
            return format_error(message_id, "unknown command")

    def vertex_in_range(self, vertex):
        if vertex == PASS:
            return True
        if 1 <= vertex[0] <= self.size and 1 <= vertex[1] <= self.size:
            return True
        else:
            return False
    
    # commands

    def cmd_protocol_version(self, arguments):
        return 2

    def cmd_name(self, arguments):
        return self._name

    def cmd_version(self, arguments):
        return self._version

    def cmd_known_command(self, arguments):
        return gtp_boolean(arguments in self.known_commands)

    def cmd_list_commands(self, arguments):
        return gtp_list(self.known_commands)

    def cmd_quit(self, arguments):
        self.disconnect = True

    def cmd_boardsize(self, arguments):
        if arguments.isdigit():
            size = int(arguments)
            if MIN_BOARD_SIZE <= size <= MAX_BOARD_SIZE:
                self.size = size
            else:
                raise ValueError("unacceptable size")
        else:
            raise ValueError("non digit size")

    def cmd_clear_board(self, arguments):
        self._game.reset()

    def cmd_komi(self, arguments):
        try:
            komi = float(arguments)
            self.komi = komi
        except ValueError:
            raise ValueError("syntax error")

    def cmd_play(self, arguments):
        move = gtp_to_coord(arguments, self.board_size)
        if self._game.solo_play(move):
            return ""
        else:
            raise ValueError("illegal move")

    def cmd_genmove(self, arguments):
        c = parse_color(arguments)
        if c:
            move = self._game.solo_play()
            return coord_to_gtp(move, self.board_size)
        else:
            raise ValueError("unknown player: {}".format(arguments))

