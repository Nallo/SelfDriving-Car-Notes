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

    print("Maze has {} rows and {} cols".format(rows, cols))
    print("--------- Maze Dump Begin ---------")
    for row in grid: print row
    print("---------- Maze Dump End ----------")

    frontier = []
    visited = []

    plan = [[-1 for _ in range(cols)] for _ in range(rows)]
    plan_count = 0

    frontier.insert(0, init)
    while len(frontier) != 0:
        node = frontier.pop(0)
        r = node[0]
        c = node[1]

        if node not in visited:
            print("Visiting node {}".format(node))
            visited.append(node)

            plan[r][c] = plan_count
            plan_count += 1

            for i in range(len(delta)):
                r2 = r + delta[i][0]
                c2 = c + delta[i][1]
                if r2 >= 0 and r2 < rows and \
                    c2 >= 0 and c2 < cols and \
                    grid[r2][c2] != 1 and [r2, c2] not in visited:
                    print("Expanding to ({},{})".format(r2,c2))
                    frontier.insert(0, [r2,c2])

            if node == goal:
                break

    for row in plan: print row
    return

search(grid,init,goal,cost)
