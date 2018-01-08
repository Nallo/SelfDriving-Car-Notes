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

    action = [[' ' for _ in range(cols)] for _ in range(rows)]
    parent = [[-1 for _ in range(cols)] for _ in range(rows)]
    plan = [[-1 for _ in range(cols)] for _ in range(rows)]
    plan_count = 0

    frontier.append(init)
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
                move = delta_name[i]
                if r2 >= 0 and r2 < rows and \
                    c2 >= 0 and c2 < cols and \
                    grid[r2][c2] != 1 and [r2, c2] not in visited:
                    print("Expanding to ({},{})"
                            " with action {}"
                            " from ({},{})".format(r2,c2,move,r,c))
                    frontier.append([r2,c2])
                    parent[r2][c2] = [r,c]
                    action[r][c] = move

            if node == goal:
                print("The path is:")
                action[r][c] = '*'
                print build_path(init,goal,parent)
                break

    for row in plan: print row
    print("")
    for row in parent: print row
    print("")
    for row in action: print row
    return

def build_path(init,goal,parent):
    steps = []
    node = goal

    while True:
        steps.append(node)
        node = parent[node[0]][node[1]]

        if node == init:
            steps.append(init)
            break

    steps.reverse()
    return steps

search(grid,init,goal,cost)
