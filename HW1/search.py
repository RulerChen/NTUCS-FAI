# search.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Michael Abir (abir2@illinois.edu) on 08/28/2018
# Modified by Shang-Tse Chen (stchen@csie.ntu.edu.tw) on 03/03/2022

"""
This is the main entry point for HW1. You should only modify code
within this file -- the unrevised staff files will be used for all other
files and classes when code is run, so be careful to not modify anything else.
"""
# Search should return the path.
# The path should be a list of tuples in the form (row, col) that correspond
# to the positions of the path taken by your search algorithm.
# maze is a Maze object based on the maze from the file specified by input filename
# searchMethod is the search method specified by --method flag (bfs,dfs,astar,astar_multi,fast)

# Claude AI and Github Copilot helped me to write 95% of the code and I discussed with my friends B10705044

import heapq
import queue
import math
from copy import deepcopy


def manhattan_distance(pos1, pos2):
    """
    Returns the Manhattan distance between two positions.
    """

    return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])


def astar_ans(maze):
    def path(start, end, prev):
        return_path = [end]
        while return_path[-1] != start:
            return_path.append(prev[return_path[-1]])
        return_path.reverse()
        return return_path

    def find_path(pos_init, target, copy_maze):
        heap = []
        heapq.heappush(heap, (manhattan_distance(pos_init, target), pos_init))
        visited = set()
        visited.add(pos_init)
        prev = {}

        while heap:
            f_value, pos = heapq.heappop(heap)
            if pos == target:
                goal = pos
                return path(pos_init, goal, prev)

            neighbors = copy_maze.getNeighbors(pos[0], pos[1])

            for neighbor in neighbors:
                if neighbor not in visited:
                    visited.add(pos)
                    prev[neighbor] = pos
                    heapq.heappush(heap, (manhattan_distance(
                        neighbor, target) + len(path(pos_init, pos, prev)), neighbor))

    def heuristic(unvisited, distance, curr_pos):
        """
        By Github
        """
        if len(unvisited) == 0:
            return 0
        result = 0
        cur_v = [maze.getObjectives().index(unvisited[0])]
        vertices = []
        for i in range(1, len(unvisited)):
            vertices.append(maze.getObjectives().index(unvisited[i]))
        while len(cur_v) != len(unvisited):
            min_paths = []
            for cv in cur_v:
                min_nv = math.inf
                min_n = None
                for vert in vertices:
                    if vert < cv:
                        edge = (vert, cv)
                    else:
                        edge = (cv, vert)
                    if distance[edge] < min_nv:
                        min_nv = distance[edge]
                        min_n = vert
                min_paths.append((min_nv, min_n))
            min_p = min(min_paths)
            vertices.remove(min_p[1])
            result += min_p[0]
            cur_v.append(min_p[1])
        l = []
        for x in unvisited:
            l.append(manhattan_distance(curr_pos, x))
        return result + min(l)

    pos_init = maze.getStart()
    pos_obj = maze.getObjectives()

    paths = []
    for i in range(len(pos_obj)):
        for j in range(i + 1, len(pos_obj)):
            paths.append((i, j))
    dis = {}
    for i, j in paths:
        copy_maze = deepcopy(maze)
        dis[(i, j)] = len(find_path(pos_obj[i], pos_obj[j], copy_maze)) - 1

    prev = {(pos_init, tuple(pos_obj)): None}
    node_dis = {(pos_init, tuple(pos_obj)): 0}

    heap = []
    heapq.heappush(heap, (heuristic(tuple(pos_obj), dis,
                   pos_init), pos_init, tuple(pos_obj)))
    while heap:
        f_value, pos, unvisited = heapq.heappop(heap)
        if (len(unvisited) == 0):
            p = []
            cur = (pos, unvisited)
            while cur != None:
                p.append(cur[0])
                cur = prev[cur]
            p.reverse()
            return p

        neighbors = maze.getNeighbors(pos[0], pos[1])
        for neighbor in neighbors:
            target = tuple(
                [point for point in unvisited if point != neighbor])
            if (neighbor, target) in node_dis and node_dis[(neighbor, target)] <= node_dis[(pos, unvisited)] + 1:
                continue
            node_dis[(neighbor, target)] = node_dis[(pos, unvisited)] + 1
            prev[(neighbor, target)] = (pos, unvisited)

            heapq.heappush(heap, (
                node_dis[(neighbor, target)] + heuristic(target, dis, neighbor), neighbor, target))

        # f_value, pos, unvisited = heap.get()
        # if (len(unvisited) == 0):
        #     p = []
        #     cur = (pos, unvisited)
        #     while cur != None:
        #         p.append(cur[0])
        #         cur = prev[cur]
        #     p.reverse()
        #     return p

        # neighbors = maze.getNeighbors(pos[0], pos[1])
        # for neighbor in neighbors:
        #     target = tuple([point for point in unvisited if point != pos])
        #     if (neighbor, target) in node_dis and node_dis[(neighbor, target)] <= node_dis[(pos, unvisited)] + 1:
        #         continue
        #     node_dis[(neighbor, target)] = node_dis[(pos, unvisited)] + 1
        #     prev[(neighbor, target)] = (pos, unvisited)

        #     f = max(f_value, heuristic(target, dis, neighbor) +
        #             node_dis[(neighbor, target)])
        #     heap.put((f, neighbor, target))


def search(maze, searchMethod):
    return {
        "bfs": bfs,
        "astar": astar,
        "astar_corner": astar_corner,
        "astar_multi": astar_multi,
        "fast": fast,
    }.get(searchMethod)(maze)


def bfs(maze):
    """
    Runs BFS for part 1 of the assignment.

    @param maze: The maze to execute the search on.

    @return path: a list of tuples containing the coordinates of each state in the computed path
    """
    # TODO: Write your code here

    # get the start position and the objectives
    pos_init = maze.getStart()
    pos_obj = maze.getObjectives()

    # initialize the queue, visited set, and path
    queue = []
    visited = {}
    path = []

    # add the start position and parent node to the queue
    queue.append((-1, pos_init))

    while queue:
        parent, pos = queue.pop(0)

        # if the position is an objective, return the path
        if pos in pos_obj:
            path.append(pos)
            while parent != -1:
                path.append(parent)
                parent = visited[parent]
            return path[::-1]

        # if the position has not been visited, add it to the visited set and add its neighbors to the queue
        if pos not in visited:
            visited[pos] = parent
            for neighbor in maze.getNeighbors(pos[0], pos[1]):
                if neighbor not in visited:
                    queue.append((pos, neighbor))
    return []


def astar(maze):
    """
    Runs A star for part 1 of the assignment.

    @param maze: The maze to execute the search on.

    @return path: a list of tuples containing the coordinates of each state in the computed path
    """
    # TODO: Write your code here

    return astar_ans(maze)


def astar_corner(maze):
    """
    Runs A star for part 2 of the assignment in the case where there are four corner objectives.

    @param maze: The maze to execute the search on.

    @return path: a list of tuples containing the coordinates of each state in the computed path
    """
    # TODO: Write your code here

    return astar_ans(maze)


def astar_multi(maze):
    """
    Runs A star for part 3 of the assignment in the case where there are
    multiple objectives.

    @param maze: The maze to execute the search on.

    @return path: a list of tuples containing the coordinates of each state in the computed path
    """
    # TODO: Write your code here

    return astar_ans(maze)


def fast(maze):
    """
    Runs suboptimal search algorithm for part 4.

    @param maze: The maze to execute the search on.

    @return path: a list of tuples containing the coordinates of each state in the computed path
    """
    # TODO: Write your code here

    def find_closest(pos, unvisited_points):
        min_distance = math.inf
        for point in unvisited_points:
            distance = manhattan_distance(pos, point)
            if distance < min_distance:
                min_distance = distance
                next_pos = point
        return next_pos

    def find_path(pos, target, maze):
        heap = []
        heapq.heappush(heap, (manhattan_distance(pos, target), pos, []))
        visited = set()

        while heap:
            _, pos, path = heapq.heappop(heap)

            new_path = path.copy()
            new_path.append(pos)

            if pos == target:
                return new_path

            if pos not in visited:
                visited.add(pos)

                for neighbor in maze.getNeighbors(pos[0], pos[1]):
                    heapq.heappush(heap, (manhattan_distance(
                        neighbor, target) + len(new_path), neighbor, new_path))

    pos_init = maze.getStart()
    pos_obj = maze.getObjectives()
    unvisited = set(pos_obj)

    path = [pos_init]
    while (unvisited):
        pos = pos_init
        target = find_closest(pos, unvisited)
        path += find_path(pos, target, maze)
        unvisited.discard(target)
        pos_init = target

    return path
