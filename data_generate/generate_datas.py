# -*- coding: utf-8 -*-

from astar import DynamicAstar
from astar import RESOLUTION,POS,G
import curses, random
import math
from pathfinding import Grid
import matplotlib.pyplot as plt
from collision import *
from tqdm import tqdm

gen_data_number=500
bound_box_x=25
bound_box_y=25
collision_d_threshold=1

def randomObstacle():



    num_obstacles = random.randint(12, 22)   ## 5-10个障碍物

    # 随机生成障碍物的位置和半径
    obstacle_coords = [(random.uniform(0, bound_box_x), random.uniform(0, bound_box_y)) for _ in range(num_obstacles)]
    obstacle_ra = [random.uniform(1, 4) for _ in range(num_obstacles)]  ##障碍物 半径

    barriers=[]
    for i in range(num_obstacles):
        barriers.append((obstacle_coords[i][0],obstacle_coords[i][1],obstacle_ra[i]))
    
    return barriers


def getCollisionCost(barriers,opt_path):
        
    cost=0

    for point in opt_path:
        point_array=np.array([point.x(),point.y(),point.z()])
        
        for obstacle in barriers:
            ob_p=np.array([obstacle[0],obstacle[1],0])
            ob_raduis=obstacle[2]
            
            obstacle_distance = np.linalg.norm(point_array - ob_p)-ob_raduis

            if obstacle_distance > collision_d_threshold:
                continue
            cost+=4*(obstacle_distance - collision_d_threshold)**2
    
    return cost


def main():

    id_start=485
    result_id=id_start
    for iter_num in tqdm(range(gen_data_number)):

        mygrid=Grid()
        mygrid.barriers=randomObstacle()

        pathPlanner=DynamicAstar(0.5,0.5,0.75)
        pathPlanner.setMap(mygrid)

        
        pathPlanner.setStart((0,0))
        pathPlanner.setGoal((random.uniform(bound_box_x/2, bound_box_x), random.uniform(0, bound_box_y/3)))


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
        collision_cost=getCollisionCost(mygrid.barriers,opt_path_)
        plot_experiment(avoidance.get_init_path(),opt_path_,mygrid.barriers,collision_cost,number=result_id)
        result_id+=1



if __name__ == "__main__":
    main()


