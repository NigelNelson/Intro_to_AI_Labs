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
        self._dungeon = dungeon
        self._start_location = start_location
        self._self_rooms_searched = False

    @property
    def dungeon(self) -> Dungeon:
        """ This function returns a reference to the dungeon.  """
        return self._dungeon

    def set_echo_rooms_searched(self) -> None:
        """ The _self_rooms_searched variable is used as a flag for whether
        the rat should display rooms as they are visited. """
        self._self_rooms_searched = True

    def path_to(self, target_location: Room) -> List[Room]:
        """ This function finds and returns a list of rooms from 
        start_location to target_location.  The list will include
        both the start and destination, and if there isn't a path
        the list will be empty. This function uses depth first search. """
        return self.recursive_dfs_search(target_location, self._start_location, [])

    def directions_to(self, target_location: Room) -> List[str]:
        """ This function returns a list of the names of the rooms from the
        start_location to the target_location. """
        room_names = []
        rooms = self.dfs_search(target_location)
        if rooms != [None]:
            for room in rooms:
                room_names.append(room.name)
        return room_names

    def dfs_search(self, target_location: Room):
        rooms = self.recursive_dfs_search(target_location, self._start_location, [])
        if rooms[-1] != target_location:
            rooms = [None]
        return rooms

    def recursive_dfs_search(self, target_location: Room, current_location: Room, visited: List[Room]):
        if self._self_rooms_searched:
            print(current_location.name)
        visited.append(current_location)
        for neighbor in current_location.neighbors():
            if neighbor is not None:
                if neighbor == target_location:
                    visited.append(neighbor)
                    return visited
                elif neighbor not in visited:
                    visited = self.recursive_dfs_search(target_location, neighbor, visited.copy())
                    if visited[-1] == target_location and visited is not None:
                        return visited
                    elif visited is not None:
                        visited.remove(neighbor)

        return visited



