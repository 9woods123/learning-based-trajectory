# -*- coding: utf-8 -*-
from astar import DynamicAstar
from astar import RESOLUTION,POS,G
import curses, random
import math

import matplotlib.pyplot as plt
from collision import *


map_data = [
        "####################",
    ]



def point_distance(point1, point2):
    return math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)


def distance_point_to_line(px, py, x1, y1, x2, y2):

    # 计算点到直线的距离
    num = abs((y2 - y1) * px - (x2 - x1) * py + x2 * y1 - y2 * x1)
    den = math.sqrt((y2 - y1)**2 + (x2 - x1)**2)+0.0001
    return num / den

def is_inside_circle(x, y, h, k, r):
    # 判断点是否在圆内（或圆上）
    return (x - h)**2 + (y - k)**2 <= r**2

def line_segment_circle_intersection(x1, y1, x2, y2, h, k, r):
    # 如果线段的两个端点都在圆内
    if is_inside_circle(x1, y1, h, k, r) or is_inside_circle(x2, y2, h, k, r):
        return True

    # 计算线段与圆心的垂直距离
    d = distance_point_to_line(h, k, x1, y1, x2, y2)

    if d > r:
        # 线段与圆心的垂直距离大于半径，则线段不与圆相交
        return False

    # 计算圆心到线段两端点的距离
    len1 = ((h - x1)**2 + (k - y1)**2)**0.5
    len2 = ((h - x2)**2 + (k - y2)**2)**0.5

    if len1 < r and len2 < r:
        # 两个端点都在圆内
        return True

    if len1 > r and len2 > r and len1 + len2 > (r + d + ((x2 - x1)**2 + (y2 - y1)**2)**0.5):
        # 两个端点都在圆外且线段不穿过圆
        return False

    return True



class Grid:

    def __init__(self) -> None:
        # self.__height=height
        # self.__width=width
        # self.barriers=[(5,15,2),(12.5,12.5,5),(18,21,3),(7,9,2.5),(7,19,2.5)]
        
        self.barriers=[(5,15,2),(7,9,5),(18,21,3),(7,19,2.5)]


    def collision_detection(self,curr_point:tuple,next_point):
        for ba in self.barriers:
            ba_center=(ba[0],ba[1])
            ba_radius=ba[2]+0.5
            if line_segment_circle_intersection(curr_point[0],curr_point[1],next_point[0],next_point[1],
                                             ba_center[0],ba_center[1],ba_radius ):
                    return True
        return False    

def draw_map(map_data,barriers):
    plt.figure(figsize=(8, 8))

    for barrier in barriers:
        x, y, radius = barrier[0:3]
        circle = plt.Circle((x,y), radius, color='black', fill=False)
        plt.gca().add_patch(circle)

def draw_path(path):
    plt.plot([p[0] for p in path], [p[1] for p in path], 'g', linewidth=2)



def draw_nodes(nodes:dict, vel:float):
        
        for index, node  in nodes.items():
            # print(key, value)
            x, y = node[POS] 

            plt.plot(x, y, 'ro')  # 使用 'ro' 表示红色圆点，你可以更改颜色和样式
            node_time = "{:.1f}".format(node[G]/vel)
            text = str(node_time)
            plt.text(x, y, text, fontsize=11, color='red')  



def main():
    # 替换为你的地图数据和路径信息

    mygrid=Grid()

    pathPlanner=DynamicAstar(1,1,1.2)
    pathPlanner.setMap(mygrid)
    pathPlanner.setStart((0,0))
    pathPlanner.setGoal((11,18))

    smooth_start_time = time.time()  # 

    path,succues,nodes = pathPlanner.pathPlan(0, 2000)

    # print (path)

    # draw_map(map_data,mygrid.barriers)
    # draw_path(path)
    # draw_nodes(nodes,pathPlanner.getVel())
    # plt.grid(True)
    # plt.show()

##=======================smooth=======================

    avoidance = ObstacleAvoidance()


    avoidance.setObstacleMapbyGrid(mygrid.barriers)
    avoidance.setInitTrajbyrawPath(path)

    # avoidance.setStart(start_point)
    # avoidance.setGoal(goal_point)
    # avoidance.setObstacle(currentObstacle)
    
    opt_path_=avoidance.smooth()
    smooth_end_time = time.time()  # 
    smooth_cost_time = smooth_end_time - smooth_start_time
    print("smooth_cost_time:",smooth_cost_time)
    # print("init_path_ size",len(init_path_))

    plot_experiment(avoidance.get_init_path(),opt_path_,mygrid.barriers)




if __name__ == "__main__":
    main()


