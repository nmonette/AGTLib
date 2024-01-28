import enum



class Action(enum.IntEnum):
    """
    Enumeration of possible actions.
    """
    left = 0 #: Turn left
    right = enum.auto() #: Turn right
    forward = enum.auto() #: Move forward
    pickup = enum.auto() #: Pick up an object
    drop = enum.auto() #: Drop an object
    toggle = enum.auto() #: Toggle / activate an object
    done = enum.auto() #: Done completing task

class BallAction(enum.IntEnum):
    left = 0
    right = enum.auto()
    up = enum.auto()
    down = enum.auto()

class simpleAction(enum.IntEnum):
    left = 0 #: Turn left
    right = enum.auto() #: Turn right
    forward = enum.auto() #: Move forward