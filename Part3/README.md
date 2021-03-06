# Path Planning

## Prediction

Estimates other vehicles behavior.

## Behavior

Estimates what maneuver to do, taking into account the prediction estimation.

## Trajectory

Select the path the car has to do based on behavior.

# Discrete Motion Planning

Given the following inputs:

  * A discretized map.
  * A starting location.
  * A goal location.
  * A cost, typically in terms of gas or path length.

The vehicle needs to find the **minimum cost path**.

According to the costs assigned to each possible state / choice the path can be
very different.

The Path Planning problem can be seen as a graph search problem.

## A first implementation of the A* Search

```python
# ----------
# User Instructions:
#
# Define a function, search() that returns a list
# in the form of [optimal path length, row, col]. For
# the grid shown below, your function should output
# [11, 4, 5].
#
# If there is no valid path from the start point
# to the goal, your function should return the string
# 'fail'
# ----------

# Grid format:
#   0 = Navigable space
#   1 = Occupied space

grid = [[0, 0, 1, 0, 0, 0],
        [0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 1, 0],
        [0, 0, 1, 1, 1, 0],
        [0, 0, 0, 0, 1, 0]]
init = [0, 0]
goal = [len(grid)-1, len(grid[0])-1]
cost = 1

delta = [[-1, 0], # go up
         [ 0,-1], # go left
         [ 1, 0], # go down
         [ 0, 1]] # go right

delta_name = ['^', '<', 'v', '>']

def search(grid,init,goal,cost):
    rows = len(grid)
    cols = len(grid[0])

    path = [0, init[0], init[1]]
    frontier = [[0, init[0], init[1]]]
    visited = [[0 for _ in range(cols)] for _ in range(rows)]

    print("Grid has {} cols and {} rows".format(cols, rows))
    print("The frontier is: {}".format(frontier))

    while len(frontier) != 0:
        frontier.sort(key=lambda x: x[0])
        visit = frontier.pop(0)

        visited[visit[1]][visit[2]] = 1
        print("Visiting node {}".format(visit))

        path = [visit[0], visit[1], visit[2]]
        print("Path is now {}".format(path))

        if [visit[1], visit[2]] == goal:
            print path
            return

        for move in delta:
            neigh = [path[0]+cost, path[1]+move[0], path[2]+move[1]]
            if path[1]+move[0] >= 0 and path[1]+move[0] < rows and \
                path[2]+move[1] >= 0 and path[2]+move[1] < cols and \
                grid[neigh[1]][neigh[2]] != 1 and visited[neigh[1]][neigh[2]] != 1:
                print("Adding {} to frontier".format(neigh))
                visited[neigh[1]][neigh[2]] == 1
                frontier.append(neigh)

        print("")

    print 'fail'
    return

search(grid, init, goal, cost)
```

## A* Search

The heuristic function does not need to be precise it is enough it is a good
guess such as *h(x,y) <= distance to goal from (x,y)* also, the heuristic can
freely not taking into account the boundaries in the maze.

The new computed property for the A* Search is *f = g + h(x,y)* where *g* is the
distance of the current state from the initial state.

# Prediction

Prediction process mixes two different approaches: *Model-Based* and *Data-Driven*.

![model_vs_data_approach](/img/model_vs_data_approach.png)

**Data Driven Approach** is typically Used for [trajectory clustering](http://video.udacity-data.com.s3.amazonaws.com/topher/2017/July/5978c2c6_trajectory-clustering/trajectory-clustering.pdf).

  1. Offline training.
    1. Define similarity
    1. Unsupervised clustering
    1. Define Prototype Trajectories
  1. Online Prediction.
    1. Observe Partial Trajectory
    1. Compare to Prototype Trajectories
    1. Generate Predictions

![data_approach](/img/data_approach.png)

**Model Driven Approach**

  1. Identify common driving behaviors (change lane, turn left, cross street, ...).
  1. Define the process model for each behaviors.
  1. Update belief by comparing the observation with the output of the process
     model.
  1. Trajectory Generation.

## Frenet Coordinate System

With Frenet coordinates, we use the variables *s* and *d* to describe a vehicle's
position on the road. The *s* coordinate represents distance along the road
(also known as *longitudinal displacement*) and the *d* coordinate represents
side-to-side position on the road (also known as *lateral displacement*).

![frenet](img/frenet.png)
