#
# CS2400 Introduction to AI
# rat.py
#
# Spring, 2020
#
# Author: Nigel Nelson
#
# Stub class for Lab 2 
# This class creates a Rat agent to be used to explore a Dungeon
# 
# Note: Instance variables with a single preceeding underscore are intended 
# to be protected, so setters and getters are included to enable this convention.
#
# Note: The -> notation in the function definition line is a type hint.  This 
# will make identifying the appropriate return type easier, but they are not 
# enforced by Python.  
#

from dungeon import Dungeon, Room, Direction
from typing import *


class Rat:
    """Represents a Rat agent in a dungeon. It enables navigation of the 
    dungeon space through searching.

    Attributes:
        dungeon (Dungeon): identifier for the dungeon to be explored
        start_location (Room): identifier for current location of the rat
    """


    def __init__(self, dungeon: Dungeon, start_location: Room):
        """ This constructor stores the references when the Rat is 
        initialized. """
        self.dungeon = dungeon
        self.start_location = start_location

    @property
    def dungeon(self) -> Dungeon:
        """ This function returns a reference to the dungeon.  """
        print("fixme")
        return

    def set_echo_rooms_searched(self) -> None:
        """ The _self_rooms_searched variable is used as a flag for whether
        the rat should display rooms as they are visited. """
        print("fixme")

    def path_to(self, target_location: Room) -> List[Room]:
        """ This function finds and returns a list of rooms from 
        start_location to target_location.  The list will include
        both the start and destination, and if there isn't a path
        the list will be empty. This function uses depth first search. """
        print("fixme")
        return []

    def directions_to(self, target_location: Room) -> List[str]:
        """ This function returns a list of the names of the rooms from the
        start_location to the target_location. """
        print("fixme")
        return []
