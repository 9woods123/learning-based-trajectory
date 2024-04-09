# -*- coding: utf-8 -*-
import math
from wayPoint import wayPoint
import copy
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import time,os

OB_R=2.1
USE_GRID=True
class ObstacleAvoidance:

    def __init__(self):

        self.__goal =None
        self.__start=None
        self.__init_path_points=None

        self.__obstacle=None

        self.__smooth_cost_weight=1
        self.__collision_cost_weight=1
        self.__reachGoal_cost_weight=1

        # self.__delta_t=0.5   ## delta_t between two near path points
        # self.vel_=2.5    ## const motion


        self.__delta_t=0.25 ## delta_t between two near path points
        self.vel_=4    ## const motion


        self.__static_obstacles=None

        self.__gridmap=None

    def get_delta_time(self):
        return self.__delta_t
    
    def setGoal(self,way_point):
         if type(way_point) is not wayPoint:
             print("please setGoal with type wayPoint")
             return False
         self.__goal =way_point

    def setStart(self,way_point):
         
         if type(way_point) is not wayPoint:
             print("please setStart with type wayPoint")
             print("your type ",type(way_point))
             return False
         self.__start =way_point



    def setObstacleMap(self,way_point_list):
        self.__static_obstacles = np.array([[point.x() ,point.y(),point.z() ]for point in way_point_list])
    

    def setObstacleMapbyGrid(self,grid):
        self.__gridmap=grid
    
    def setInitTrajbyrawPath(self, raw_path):
        self.__init_path_points=[]                                                             # 生成路径点
        for point in raw_path:
            inter_point=wayPoint(point[0],point[1],0)
            self.__init_path_points.append(inter_point)

    def get_obstacle_position(self, time):
        if self.__obstacle is None:
            print("please setObstacle before using function get_obstacle_position()")
        self.__obstacle.get_obstacle_position_by_time(time)

    def get_init_path(self):
        return self.__init_path_points

    def init_path(self):
        if  self.__goal ==None or self.__start==None:
            print("please setGoal and setStart before init_path")
            return False
        
        ## 初始路径 匀速运动假设

        distance=wayPoint.getDistance(self.__start, self.__goal)
        num_steps = int(distance / self.vel_ / self.__delta_t)        # 计算时间步数,向下取整数。


        self.__init_path_points=[]                                                             # 生成路径点
        for curr_step in range(num_steps+1):
            curr_time=curr_step*self.__delta_t

            interpolated_point = self.__start + curr_time * self.vel_* ( self.__goal -  self.__start)/distance
            interpolated_point.set_time(curr_time)
            self.__init_path_points.append(interpolated_point)

        self.__init_path_points.append(self.__goal)
        self.__init_path_points[-1].set_time(distance / self.vel_)

        return self.__init_path_points


    def get_collision_cost(self):
            cost=1
            grad=1

            return [cost,grad]

    def get_reachgoal_cost(self):
            cost=1
            grad=1
            
            return [cost,grad]

    def smooth(self,init_path=None,obstacle_set=None):
        
        init_path=self.__init_path_points
        iterations = 0
        max_iterations = 50
        path_length = len(init_path)

        if path_length < 5:
            return init_path

        # new_path = copy.deepcopy(init_path)
        # temp_path = copy.deepcopy(init_path)
        new_path = init_path.copy()
        # temp_path =  init_path.copy()
        alpha = 0.1  # You might need to adjust this based on your requirements
        
        total_cost=0
        
        while iterations < max_iterations:
            for i in range(2, path_length - 2):
                xim2 = np.array([new_path[i - 2].x(), new_path[i - 2].y(), new_path[i - 2].z()])
                xim1 = np.array([new_path[i - 1].x(), new_path[i - 1].y(), new_path[i - 1].z()])
                xi = np.array([new_path[i].x(), new_path[i].y(), new_path[i].z()])
                xip1 = np.array([new_path[i + 1].x(), new_path[i + 1].y(), new_path[i + 1].z()])
                xip2 = np.array([new_path[i + 2].x(), new_path[i + 2].y(), new_path[i + 2].z()])

                smooth_grad,smooth_cost=self.smoothness_term(xim2, xim1, xi, xip1, xip2)
                collision_grad,collision_cost=self.obstacle_term(xi)

                correction = self.__smooth_cost_weight*smooth_grad
                correction += self.__collision_cost_weight*collision_grad

                total_cost= self.__smooth_cost_weight*smooth_cost+ \
                                        self.__collision_cost_weight*collision_cost
                # print("total_cost=",total_cost)
                # correction += target_obstacle_term(xi, configurationSpace)

                xi = xi - alpha * correction 
                new_path[i]=wayPoint( xi[0], xi[1], xi[2])
                # temp_path[i]=wayPoint( xi[0], xi[1], xi[2])

            # new_path = temp_path
            iterations += 1
        
        return new_path


    def smoothness_term(self,xim2, xim1, xi, xip1, xip2):

        cost=np.linalg.norm((xip2-xip1) - (xip1 -xi))

        return xip2-4*xip1+6*xi-4*xim1+xim2, cost



    def obstacle_term(self,xi,w_obstacle=1.0, d_threshold=0.25):
        gradient = np.zeros(3)
        cost=0

        if USE_GRID:
            if self.__gridmap is None:
                return gradient,cost

            for obstacle in self.__gridmap:
                ob_p=np.array([obstacle[0],obstacle[1],0])
                ob_raduis=obstacle[2]
                obstacle_distance = np.linalg.norm(xi - ob_p)-ob_raduis

                if obstacle_distance > d_threshold:
                    continue

                cost+=(obstacle_distance-d_threshold)**2

                obstacle_gradient = 2 * (obstacle_distance - d_threshold)*(xi - ob_p) / (obstacle_distance)

                gradient += obstacle_gradient

            return gradient,cost

        else:
            d_threshold=OB_R+d_threshold

            if self.__static_obstacles is None:
                return gradient,cost


            for obstacle_position in self.__static_obstacles:

                obstacle_distance = np.linalg.norm(xi - obstacle_position)

                if obstacle_distance > d_threshold:
                    continue

                cost+=(obstacle_distance-d_threshold)**2

                obstacle_gradient = 2 * (obstacle_distance - d_threshold)*(xi - obstacle_position) / (obstacle_distance)


                gradient += obstacle_gradient

            return gradient,cost


    def target_obstacle_term(xi, configuration_space):

        return np.zeros(3)



###################################  main and  plot  #########################################


def plot_experiment(init_path,opt_path,grid_barriar,collision_cost,number=-1):

    init_path_coords_x = np.array([point.x() for point in init_path])
    init_path_coords_y = np.array([point.y() for point in init_path])

    opt_path_coords_x = np.array([point.x() for point in opt_path])
    opt_path_coords_y = np.array([point.y() for point in opt_path])

    ob_coords_x = np.array([ob[0] for ob in grid_barriar])
    ob_coords_y = np.array([ob[1] for ob in grid_barriar])
    ob_r= np.array([ob[2] for ob in grid_barriar])

    # plot_circle((2, 2), 1)
    plt.figure(figsize=(8,8))
    plt.clf()
    # 调整标签的字体大小
    plt.xlabel('x (m)', fontsize=25)
    plt.ylabel('y (m)', fontsize=25)

    # 调整图例的字体大小
    # plt.legend(['init_path', 'opt_path'], fontsize=20)
    # plt.title("Planned Path",fontsize=25)
    # plt.text(opt_path_coords_x[0], opt_path_coords_y[0], 'Start', fontsize=25, color='red', ha='left', va='bottom')
    # plt.text(opt_path_coords_x[-1], opt_path_coords_y[-1]-0.5, 'Goal', fontsize=25, color='green', ha='right', va='bottom')
    plt.axis('off')  # Set equal aspect ratio
    plt.grid(False)

    # 调整刻度的字体大小
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)

        # Plot circles for static obstacles
    for i in range(len(ob_coords_x)):
        circle = Circle((ob_coords_x[i], ob_coords_y[i]), radius=ob_r[i], fc='black', ec='black')
        plt.gca().add_patch(circle)

    plt.xlim(0, 25)
    plt.ylim(0, 25)



    save_map_dir='data/map'
    os.makedirs(save_map_dir, exist_ok=True)
    file_name= 'map_'+str(number)+'.png'
    save_path = os.path.join(save_map_dir,file_name)
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()  # 关闭图形窗口

    # plot_circle((2, 2), 1)
    plt.figure(figsize=(8,8))
    plt.clf()
    # 调整标签的字体大小
    plt.xlabel('x (m)', fontsize=25)
    plt.ylabel('y (m)', fontsize=25)

    # 调整图例的字体大小
    # plt.legend(['init_path', 'opt_path'], fontsize=20)
    # plt.title("Planned Path",fontsize=25)
    # plt.text(opt_path_coords_x[0], opt_path_coords_y[0], 'Start', fontsize=25, color='red', ha='left', va='bottom')
    # plt.text(opt_path_coords_x[-1], opt_path_coords_y[-1]-0.5, 'Goal', fontsize=25, color='green', ha='right', va='bottom')
    plt.axis('off')  # Set equal aspect ratio
    plt.grid(False)
    plt.xlim(0, 25)
    plt.ylim(0, 25)
    # 调整刻度的字体大小
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
            # Plot circles for static obstacles
    for i in range(len(ob_coords_x)):
        circle = Circle((ob_coords_x[i], ob_coords_y[i]), radius=ob_r[i], fc='black', ec='black')
        plt.gca().add_patch(circle)
    plt.plot(init_path_coords_x,init_path_coords_y, '--', linewidth=4)
    plt.plot(opt_path_coords_x,opt_path_coords_y, '.-', linewidth=3)

    save_exec_traj_dir='data/traj'
    os.makedirs(save_exec_traj_dir, exist_ok=True)
    file_name= 'result'+str(number)+'.png'
    save_path = os.path.join(save_exec_traj_dir,file_name)
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()  # 关闭图形窗口

    save_traj_result_dir='data/traj_data'
    os.makedirs(save_traj_result_dir, exist_ok=True)
    file_name= 'traj_'+str(number)+'.txt'
    save_path = os.path.join(save_traj_result_dir,file_name)
    np.savetxt(save_path, np.column_stack((opt_path_coords_x, opt_path_coords_y)), delimiter='\t')

    save_cost_result_dir='data/collision_cost'
    os.makedirs(save_cost_result_dir, exist_ok=True)
    file_name= 'cost_'+str(number)+'.txt'
    collision_cost_array = np.array([collision_cost])
    save_path = os.path.join(save_cost_result_dir,file_name)
    np.savetxt(save_path, collision_cost_array, delimiter='\t')
