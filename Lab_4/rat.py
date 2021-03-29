#
# CS2400 Introduction to AI
# rat.py
#
# Spring, 2020
#
# Author: Nigel Nelson
#
# Class for Lab 3
# This class creates a Rat agent to be used to explore a Dungeon
# The rat agent can employ either DFS, BFS, or ID search algorithm
# in order to find the correct room.
#
# Note: Instance variables with a single proceeding underscore are intended
# to be protected, so setters and getters are included to enable this
# convention.
#
# Note: The -> notation in the function definition line is a type hint.  This
# will make identifying the appropriate return type easier, but they are not
# enforced by Python.
#

from dungeon import Dungeon
from dungeon import Room
from typing import *


class Rat:
    """Represents a Rat agent in a dungeon. It enables navigation of the
    dungeon space through searching.

    Attributes:
        dungeon (Dungeon): identifier for the dungeon to be explored
        start_location (Room): identifier for current location of the rat
        self_rooms_searched (Boolean): flag used to indicate
        whether or not to print the visited rooms
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
        return self.__dfs_search(target_location)

    def directions_to(self, target_location: Room) -> List[str]:
        """ This function returns a list of the names of the rooms from the
        start_location to the target_location. """
        room_names = []
        rooms = self.__dfs_search(target_location)
        if rooms != [None]:
            for room in rooms:
                room_names.append(room.name)
        return room_names

    def __dfs_search(self, target_location: Room):
        """ This function returns the list of rooms to a given target location,
        and if the returned list doesn't have the desired target room at the
        end, then an empty list is returned"""
        rooms = self.__recursive_dfs_search(target_location,
                                            self._start_location, [])
        if rooms[-1] != target_location:
            rooms = [None]
        return rooms

    def __optional_echo(self, visited_room: Room):
        """ This function is called in order to check if _self_rooms_searched
        is True, and if so, prints that it has visited the specified room
        the rooms that it """
        if self._self_rooms_searched:
            print("Visiting:", visited_room.name)

    def __recursive_dfs_search(self, target_location: Room,
                               current_location: Room, visited: List[Room]):
        """ Recursive depth first search of rooms, which searches all rooms
        in a dungeon. However, in order to be recursive it must always
        return something, as such, its sometimes returns a list of rooms
        that doesn't contain the target"""
        self.__optional_echo(current_location)
        visited.append(current_location)
        for neighbor in current_location.neighbors():
            if neighbor is not None:
                if neighbor == target_location:
                    self.__optional_echo(neighbor)
                    visited.append(neighbor)
                    return visited
                elif neighbor not in visited:
                    visited = self.__recursive_dfs_search(
                        target_location, neighbor, visited.copy())
                    if visited is not None and visited[-1] == target_location:
                        return visited
                    elif visited is not None:
                        visited.pop()
        return visited

    def bfs_directions_to(self, target_location: Room) -> List[str]:
        """Return the list of rooms names from the rat's current location to
        the target location. Uses breadth-first search."""
        room_names = []
        rooms = self.bfs_path_to(target_location)
        if rooms != [None]:
            for room in rooms:
                room_names.append(room.name)
        return room_names

    def bfs_path_to(self, target_location: Room) -> List[Room]:
        """Returns the list of rooms from the start location to the
        target location, using breadth-first search to find the path."""
        frontier = [self._start_location]
        explored = set()
        room_map = {}
        while frontier:
            current_room = frontier.pop(0)
            explored.add(current_room)
            self.__optional_echo(current_room)
            if current_room is target_location:
                path = [current_room]
                while current_room is not self._start_location:
                    path.append(room_map[current_room])
                    current_room = room_map[current_room]
                path.reverse()
                return path
            else:
                for room in current_room.neighbors():
                    if room not in explored:
                        frontier.append(room)
                        explored.add(room)
                        room_map[room] = current_room
        return []

    def id_path_to(self, target_location: Room) -> List[Room]:
        """Returns the list of rooms from the start location to the
        target location, using iterative deepening."""
        max_depth = 2
        rooms, is_searchable = self.__recursive_id_search(
            target_location, self._start_location, [], max_depth, 0, False)
        while is_searchable and rooms[-1] != target_location:
            max_depth += 1
            rooms, is_searchable = self.__recursive_id_search(
                target_location, self._start_location, [], max_depth, 0, False)
        if rooms[-1] != target_location:
            rooms = [None]
        return rooms

    def id_directions_to(self, target_location: Room) -> List[str]:
        """Return the list of rooms names from the rat's current location to
        the target location. Uses iterative deepening."""
        room_names = []
        rooms = self.id_path_to(target_location)
        if rooms != [None]:
            for room in rooms:
                room_names.append(room.name)
        return room_names

    def __recursive_id_search(self, target_location: Room,
                              current_location: Room, visited: List[Room],
                              max_depth: int, current_depth: int,
                              is_searchable: bool):
        """ Recursive iterative deepening search of rooms, which searches all
        rooms in a dungeon, or stops searching once depth of search has
        reached its specified max search distance."""
        self.__optional_echo(current_location)
        visited.append(current_location)
        current_depth += 1
        if current_depth < max_depth:
            for neighbor in current_location.neighbors():
                if neighbor is not None:
                    if neighbor == target_location:
                        self.__optional_echo(neighbor)
                        visited.append(neighbor)
                        return visited, is_searchable
                    elif neighbor not in visited:
                        visited, is_searchable = self.__recursive_id_search(
                            target_location, neighbor, visited.copy(),
                            max_depth, current_depth, is_searchable)
                        current_depth += 1
                        if (visited is not None
                                and visited[-1] == target_location):
                            return visited, is_searchable
                        elif visited is not None:
                            visited.pop()
                            current_depth -= 1
        else:
            is_searchable = True
        return visited, is_searchable
