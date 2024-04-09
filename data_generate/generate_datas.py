# -*- coding: utf-8 -*-

from .astar import DynamicAstar
from .astar import RESOLUTION,POS,G
import curses, random
import math
from pathfinding import Grid
import matplotlib.pyplot as plt
from .collision import *
from tqdm import tqdm

gen_data_number=5000
bound_box_x=25
bound_box_y=25

def randomObstacle():



    num_obstacles = random.randint(5, 11)   ## 5-10个障碍物

    # 随机生成障碍物的位置和半径
    obstacle_coords = [(random.uniform(0, bound_box_x), random.uniform(0, bound_box_y)) for _ in range(num_obstacles)]
    obstacle_ra = [random.uniform(1.2, 3) for _ in range(num_obstacles)]  ##障碍物 半径

    barriers=[]
    for i in range(num_obstacles):
        barriers.append((obstacle_coords[i][0],obstacle_coords[i][1],obstacle_ra[i]))
    
    return barriers

def main():
    iter_num=4349



    for epoch in tqdm(range(gen_data_number)):

        mygrid=Grid()
        mygrid.barriers=randomObstacle()

        pathPlanner=DynamicAstar(0.5,0.5,0.75)
        pathPlanner.setMap(mygrid)

        
        pathPlanner.setStart((0,0))
        pathPlanner.setGoal((random.uniform(0, bound_box_x), random.uniform(0, bound_box_y/4)))


        path,succues,nodes = pathPlanner.pathPlan(0, 1000)

        if not succues :
            continue
        if  len(path)<20:
            continue

        avoidance = ObstacleAvoidance()
        avoidance.setObstacleMapbyGrid(mygrid.barriers)
        avoidance.setInitTrajbyrawPath(path)
        opt_path_=avoidance.smooth()


        # print("init_path_ size",len(init_path_))

        plot_experiment(avoidance.get_init_path(),opt_path_,mygrid.barriers,number=iter_num)
        iter_num+=1


if __name__ == "__main__":
    main()


