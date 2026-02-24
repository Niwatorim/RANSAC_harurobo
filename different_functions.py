import time
import numpy as np
from skimage.measure import LineModelND, ransac, CircleModel
from sklearn.cluster import DBSCAN
import pandas as pd
from pandas import DataFrame
import os
import csv
import math
from matplotlib import pyplot as plt
from typing import List
from sklearn.cluster import DBSCAN
from typing import Dict, Optional, Union
from dataclasses import dataclass

#walls
MIN_WALL_LENGTH=0.25 #number of points for a valid wall
DISTANCE_DIFFERENCE= 0.2
NUMBER_TRIALS_WALLS=100
RESIDUAL_THRESHOLD_WALLS=0.025 #how much distance from a line can be counted inlier
MIN_WALL_POINTS=5 #min number of points for a wall to be called a wall
MAX_NUMBER_WALLS=2 #maximum number of walls it will draw 
WALL_MARGIN=0.05

MIN__OK_ANGLE=2.5


def wall_ransac(data:np.ndarray) -> List[dict]:
        """
        Docstring for wall_ransac
        
        :param data: Description
        """
        remaining_data=data.copy() #data to be used
        walls=[] #location of identified walls
        min_inliers= MIN_WALL_POINTS
        while len(remaining_data)>min_inliers and len(walls)<=MAX_NUMBER_WALLS:
            try:
                model_robust, inliers=ransac(
                    remaining_data,LineModelND,min_samples=5,residual_threshold=RESIDUAL_THRESHOLD_WALLS,max_trials=NUMBER_TRIALS_WALLS
                )
                if model_robust is None:
                    break
            except Exception:
                break

            inlier_points=remaining_data[inliers]
            if len(inlier_points) < min_inliers: # SUS
                break

            point_on_line= model_robust.origin
            direction=model_robust.direction

            t=np.dot(inlier_points - point_on_line, direction)
            sorted_indices = np.argsort(t)
            t_sorted=t[sorted_indices]
            gaps=np.diff(t_sorted)
            split_indices = np.where(gaps>DISTANCE_DIFFERENCE)[0] + 1
            clusters = np.split(t_sorted,split_indices)
            

            angle_rad = np.arctan2(direction[1],direction[0]) #gives the gradient
            angle_deg = np.degrees(angle_rad)
            print("angle: ", angle_deg)
            
            for cluster in clusters:
                if len(cluster)<2:
                    continue
                t_min, t_max = cluster.min(),cluster.max()
                wall_length=abs(t_max-t_min)

                if wall_length> MIN_WALL_LENGTH:
                    wall_endpoints = point_on_line + np.outer([t_min,t_max],direction)
                    walls.append({"wall":wall_endpoints,"angle":angle_deg})
            
            remaining_data = remaining_data[~inliers] #type: ignore
        return walls

def get_poles(df:DataFrame):
    pass


def get_rings(msg):
    #check distance from wall
    #check angle
    #return distance and angle
    #len(lidar_data) = 721

    #STEP 1: CHECK THE ANGLE
    angles = np.arange(msg.angle_min, msg.angle_max, msg.angle_increment)
    ranges = np.array(msg.ranges)
    ranges = ranges[320:400]
    angles = angles[320:400]

    # Filter out invalid readings
    valid_mask = np.isfinite(ranges) & (ranges > msg.range_min) & (ranges < msg.range_max)
    ranges = ranges[valid_mask]
    angles = angles[valid_mask]

    x = ranges * np.cos(angles)
    y = ranges * np.sin(angles)

    points=np.column_stack([x,y])

    wall_segments=wall_ransac(points)

    for i, wall in enumerate(wall_segments):
        label = "Wall" if i == 0 else None
        plt.plot(wall["wall"][:, 0], wall["wall"][:, 1], c="blue", linewidth=2, label=label)
        print(f"ANGLE OF WALL: {i} is {wall['angle']:.2f}")

    #STEP 2: GO TO THE LOCATION
    target_x = 0
    target_y = 0
    mid_point = msg.ranges[360]
    if wall_segments[0]["angle"] <= MIN__OK_ANGLE:
        target_x = mid_point * np.cos(np.deg2rad(90))
        target_y= mid_point * np.sin(np.deg2rad(90))
        print(f"distance x: {target_x}")
        print(f"distance y: {target_y}")

    return wall_segments[0]["angle"], target_x, target_y


    """
    limited_view = df[(df["angle"]>-10.0) & (df["angle"]<10.0)]
    
    y=limited_view["x"]
    x=limited_view["y"]
    points=np.column_stack([x,y])

    wall_segments=wall_ransac(points)

    for i, wall in enumerate(wall_segments):
        label = "Wall" if i == 0 else None
        plt.plot(wall["wall"][:, 0], wall["wall"][:, 1], c="blue", linewidth=2, label=label)
        print(f"ANGLE OF WALL: {i} is {wall['angle']:.2f}")



    #STEP 2: GO TO THE LOCATION

    if wall_segments[0]["angle"] <= MIN__OK_ANGLE:
        distance_x = df[df["angle"]==0.0]["x"]
        distance_y= df[df["angle"]==0.0]["y"]
        print(f"distance x: {distance_x}")
        print(f"distance y: {distance_y}")
        print(limited_view)
    
    
    """


def place_poles(df:DataFrame):
    #check distance from wall
    #check angle
    pass

def get_poles_2(df:DataFrame):
    #run pole check
    pass


def get_rings_2(msg):
    #check distance from wall
    #check angle
    #return distance and angle
    #len(lidar_data) = 721

    #STEP 1: CHECK THE ANGLE
    angles = np.arange(msg.angle_min, msg.angle_max, msg.angle_increment)
    ranges = np.array(msg.ranges)
    ranges = ranges[320:400]
    angles = angles[320:400]

    # Filter out invalid readings
    valid_mask = np.isfinite(ranges) & (ranges > msg.range_min) & (ranges < msg.range_max)
    ranges = ranges[valid_mask]
    angles = angles[valid_mask]

    x = ranges * np.cos(angles)
    y = ranges * np.sin(angles)

    points=np.column_stack([x,y])

    wall_segments=wall_ransac(points)

    for i, wall in enumerate(wall_segments):
        label = "Wall" if i == 0 else None
        plt.plot(wall["wall"][:, 0], wall["wall"][:, 1], c="blue", linewidth=2, label=label)
        print(f"ANGLE OF WALL: {i} is {wall['angle']:.2f}")

    #STEP 2: GO TO THE LOCATION
    target_x = 0
    target_y = 0
    mid_point = msg.ranges[360]
    if wall_segments[0]["angle"] <= MIN__OK_ANGLE:
        target_x = mid_point * np.cos(np.deg2rad(90))
        target_y= mid_point * np.sin(np.deg2rad(90))
        print(f"distance x: {target_x}")
        print(f"distance y: {target_y}")

    return wall_segments[0]["angle"], target_x, target_y


    """
    limited_view = df[(df["angle"]>-10.0) & (df["angle"]<10.0)]
    
    y=limited_view["x"]
    x=limited_view["y"]
    points=np.column_stack([x,y])

    wall_segments=wall_ransac(points)

    for i, wall in enumerate(wall_segments):
        label = "Wall" if i == 0 else None
        plt.plot(wall["wall"][:, 0], wall["wall"][:, 1], c="blue", linewidth=2, label=label)
        print(f"ANGLE OF WALL: {i} is {wall['angle']:.2f}")



    #STEP 2: GO TO THE LOCATION

    if wall_segments[0]["angle"] <= MIN__OK_ANGLE:
        distance_x = df[df["angle"]==0.0]["x"]
        distance_y= df[df["angle"]==0.0]["y"]
        print(f"distance x: {distance_x}")
        print(f"distance y: {distance_y}")
        print(limited_view)
    
    
    """


def place_poles_2(df:DataFrame):
    #check distance from wall
    #check angle
    pass

def honmaru(df:DataFrame):
    #check wall
    #check angle
    pass

if __name__ == "__main__":
    df=pd.read_csv("real_points/copy/lidar_data_taking_rings.csv")
    # get_rings(df=df)