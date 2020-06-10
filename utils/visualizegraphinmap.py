import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as img
import cv2
import random
# import matplotlib.
# sys.path.append("../")
from .utils import adj2edges, load_explore_traj, load_navigation_traj_goal, computedistance2points

def draw_node_edge(adj, coords, savepath):
    '''
    Draw the nodes and the edges
    '''
    
    fig, ax = plt.subplots()
    plt.figure(1)
    for (x, y, heading) in coords:
        plt.scatter(x,y, marker='x', alpha=0.6, c="r", s=1.8)
    
    edges = adj2edges(adj)
    for edge in edges:
        node1 = edge[0]
        node2 = edge[1]
        x1 = coords[node1][0]
        y1 = coords[node1][1]
        x2 = coords[node2][0]
        y2 = coords[node2][1]

        plt.plot([x1, x2], [y1, y2], color='dodgerblue', alpha=0.5, linewidth=0.4)
    plt.xticks([])
    plt.yticks([])
    # plt.show()
    fig.savefig(os.path.join(savepath, 'graph.png'),dpi=800)#,format='eps')

def draw_explore_traj(locations, savepath):
    '''
    Draw how the agent explore the environment
    '''
    fig, ax = plt.subplots()
    plt.figure(1)
    plt.plot([locations[0][0], locations[1][0]], [locations[0][1], locations[1][1]], color='dodgerblue', linewidth=2, label="trajectory")
    for i in range(1, len(locations)-1):
        plt.plot([locations[i][0], locations[i+1][0]], [locations[i][1], locations[i+1][1]], color='dodgerblue', linewidth=1.8)
    plt.scatter(locations[0][0], locations[0][1], marker='*', c="green", s=400, alpha=0.3, label='\nstart point')
    plt.scatter(locations[-1][0], locations[-1][1], marker='*', c="red", s=400, alpha=0.3, label='\nend point')
    plt.xticks([])
    plt.yticks([])
    plt.legend(fontsize=7,loc = 'upper left')
    fig.savefig(os.path.join(savepath,'explore_traj.png'), dpi=800)

def rl_reward(path, savepath):
    rewards = []
    with open(path,'r') as f:
        for row in f.readlines():
            if "reward" in row:
                value = row.split("reward")[1]
                rewards.append(float(value.strip()))
    # rewards = rewards[0:1400]
    print("There are total {} rewards".format(len(rewards)))
    avg = []
    for i in range(len(rewards)-1):
        avg.append((rewards[i]+rewards[i+1])/2)
    fig, ax = plt.subplots()
    plt.figure(1)
    plt.plot(range(len(rewards)), rewards, color='dodgerblue', linewidth=1, alpha=0.6)
    # plt.plot(range(len(avg)), avg, color='b', linewidth=1, alpha=1.0)
    # plt.xticks([])
    plt.ylabel("Rewards")
    plt.xlabel("Episode")
    fig.savefig(os.path.join(savepath,'rewards.png'), dpi=800)

def draw_navigation(imgpath, pathname, savepath):
    # fig = plt.figure()
    # plt.figure(1)
    # plt.axis('off')
    # bgimg = img.imread(imgpath)
    # [x_shape, y_shape,_] = bgimg.shape
    # fig.figimage(bgimg)
    distance = {}
    for name in os.listdir(pathname):
        path, target = load_navigation_traj_goal(os.path.join(pathname, name))
        dis = computedistance2points(path[0], path[-1])
        distance[dis] = [path, target]

    distance = sorted(distance.items(), key=lambda x:x[0], reverse=True)

    [dis, [path, target]] = distance[40] # 15
    print(dis)
    paths = [path]
    targets = [target]

    [dis, [path, target]] = distance[0]
    paths.append(path)
    targets.append(target)

    [dis, [path, target]] = distance[51]
    paths.append(path)
    targets.append(target)

    [dis, [path, target]] = distance[21]
    paths.append(path)
    targets.append(target)

    [dis, [path, target]] = distance[63]
    paths.append(path)
    targets.append([target[0], target[1]-8])

    [dis, [path, target]] = distance[74]
    paths.append(path)
    targets.append(target)

    # print(target)
    # print(path)
    # plt.scatter(0, 0, marker='*', c="green", s=400, alpha=0.3, label='\ntarget')
    # plt.scatter(target[0], target[1], marker='o', c="green", s=400, alpha=0.3, label='\ntarget')
    # plt.scatter(path[1][0], path[1][1], marker='*', c="red", s=400, alpha=0.3, label='\nstart point')
    # plt.scatter(path[-1][0], path[-1][1], marker='*', c="red", s=400, alpha=0.3, label='\nend point')

    # for i in range(1, len(path)-1):
    #     plt.plot([path[i][0], path[i+1][0]], [path[i][1], path[i+1][1]], color='dodgerblue', linewidth=1.8)
    # plt.xticks([])
    # plt.yticks([])
    # plt.legend(fontsize=7,loc = 'upper left')
    # plt.show()
    # fig.savefig(savepath, dpi=800)
    background = cv2.imread(imgpath)
    for path, target in zip(paths, targets):
        color = (random.randint(0,255), random.randint(0,255), random.randint(0,255))
        
        # cv2.imwrite('/home/peppa/GraphAM/weights/min_10.0_max_20.0/bg.png',background)
        radius = 15
        cv2.circle(background, (int(path[0][0]), int(path[0][1])), radius, color, 3) # start point
        cv2.rectangle(background, (int(path[-1][0]-radius), int(path[-1][1]-radius)), (int(path[-1][0]+radius), int(path[-1][1]+radius)), color, 3) # end point
        for i in range(1, len(path)-1):
            cv2.line(background, (int(path[i][0]), int(path[i][1])),(int(path[i+1][0]), int(path[i+1][1])), color, 5)
        
        radius = int(radius*2)
        pts=np.array([[target[0]-radius, target[1]],[target[0], target[1]-radius],[target[0]+radius, target[1]],[target[0], target[1]+radius]], np.int32)
        pts=pts.reshape((-1,1,2))
        cv2.polylines(background, [pts], True, color, 3) # target 
        cv2.circle(background, (int(target[0]), int(target[1])), 3, color, 3) # middle point

    cv2.imwrite(savepath,background)

def draw_explore_traj_cv(imgpath, locations, savepath):
    background = cv2.imread(imgpath)
    color = [235,72,18]
    for i in range(len(locations)-1):
        cv2.line(background, (int(locations[i][0]), int(locations[i][1])),(int(locations[i+1][0]), int(locations[i+1][1])), color, 5)

    cv2.imwrite(savepath,background)


if __name__ == "__main__":
    locations = load_explore_traj('../dataset/carla_rawdata3/')
    draw_explore_traj_cv('/home/peppa/GraphAM/weights/min_10.0_max_20.0/Town02.png', locations, '../weights/min_10.0_max_20.0/explore_traj_bg.png')
    # rl_reward("/home/peppa/GraphAM/outputs/logs/debug_2020-05-29_01-20-07/train.log", '../weights/min_10.0_max_20.0/')
    # draw_navigation('/home/peppa/GraphAM/weights/min_10.0_max_20.0/Town02.png', '../weights/drl_path/', '/home/peppa/GraphAM/weights/min_10.0_max_20.0/navigation.png')