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


"""
#TODO: add prefilter mask to remove obv outliers
def prefilter_data(self, data: np.ndarray, 
                       max_range: float = 10.0,
                       min_range: float = 0.05) -> np.ndarray:
        ""Remove obvious outliers before processing""
        distances = np.linalg.norm(data, axis=1)
        mask = (distances >= min_range) & (distances <= max_range)
        return data[mask]

"""


def find_back_wall_midpoints(wall_data:list[dict]) -> dict:
        """
        find midpoints of all walls
        find highest y value, thats back wall, 
        tell x and y coordinates from that, and return angle as well

        :param: wall_data: is a list of all walls with the endpoints and angles per wall
        """
        max_y=0
        final_wall={}
        for i in wall_data:
            angle_deg=i["angle"]
            wall_endpoints=i["wall"]
            mid_point_y=(wall_endpoints[0][1]+wall_endpoints[1][1])/2
            mid_point_x=(wall_endpoints[0][0]+wall_endpoints[1][0])/2
            if mid_point_y>max_y:
                final_wall["angle"]=angle_deg
                final_wall["mid_x"]=mid_point_x
                final_wall["mid_y"]=mid_point_y
                max_y= mid_point_y
        return final_wall

def find_back_wall_angle(wall_data:list[dict]) -> dict: #unused
        """
        find angle of all walls
        find x and y coordinates from the midpoint of wall closes to 0

        :param: wall_data: is a list of all walls with the endpoints and angles per wall
        """
        final_wall={}
        degrees=361 #no angle can be this big
        for i in wall_data:
            angle_deg=i["angle"]
            wall_endpoints=i["wall"]
            mid_point_y=(wall_endpoints[0][1]+wall_endpoints[1][1])/2
            mid_point_x=(wall_endpoints[0][0]+wall_endpoints[1][0])/2
            if abs(angle_deg)<degrees:
                final_wall["angle"]=angle_deg
                final_wall["mid_x"]=mid_point_x
                final_wall["mid_y"]=mid_point_y
                degrees=angle_deg
        
        return final_wall

def find_poles_midpoint(all_poles:List[dict])-> dict|None:
    final_point: Dict[str, Optional[float]]={
        "mid_x":None,
        "mid_y":None,
        "angle":None
    }

    midpoint_y=0
    midpoint_x=0
    if all_poles:
        if len(all_poles)%2==1: #dealing with one pole
            for pole in all_poles:
                midpoint_y+=pole["cy"]
                midpoint_x+=pole["cx"]
            final_point["mid_y"]=midpoint_y/len(all_poles)
            final_point["mid_x"]=midpoint_x/len(all_poles)
        
        else: #should be dealing with 2 poles
            for pole in all_poles:
                midpoint_y+=pole["cy"]
                midpoint_x+=pole["cx"]
            final_point["mid_y"]=midpoint_y/(len(all_poles)+1e-6)
            final_point["mid_x"]=midpoint_x/(len(all_poles)+1e-6)

            first_pole=all_poles[0]
            second_pole=all_poles[1]
            diff_y=first_pole["cy"]-second_pole["cy"]
            diff_x=first_pole["cx"]-second_pole["cx"]
            angle_rad=np.arctan2(diff_y,diff_x)
            final_point["angle"]=np.rad2deg(angle_rad)

        return final_point

    else:
        return None

def v1_both_poles_and_walls(df):
    MIN_WALL_LENGTH=0.5 #number of points for a valid wall
    DISTANCE_DIFFERENCE= 0.2
    NUMBER_TRIALS_POLES=500
    NUMBER_TRIALS_WALLS=500

    def segment_jumps(raw_distances,jump_threshold=0.08):
        """Basically it breaks different parts up by the jumps it sees in the distances"""
        
         #find the ddifferences and find indices where there are jumps
        distance_difference=np.abs(np.diff(raw_distances))
        jump_indices=np.where(distance_difference>jump_threshold)[0]

        segments=[]
        prev=0
        for j in jump_indices:
            if j-prev >= 3:
                segments.append((prev, j + 1))
            prev=j+1 #to go to next index
        
        #basically if a segment is more than at least like 3 points then count it
        if len(raw_distances)-prev>=3:
            segments.append((prev,len(raw_distances)))
        
        return segments

    def classify_cluster(points): #this checks clusters as wall or pole]
        """
        Try fit a line, then a circle, see which one works better and then fit the pole regarding it
        
        :param points: Description
        """
        #if not enough points, no ned
        if len(points)<5: return None
        
        #try fit a line
        try:
            line_model,line_inliers=ransac(
                points,LineModelND,min_samples=2,residual_threshold=0.02,max_trials=NUMBER_TRIALS_WALLS
            )
            line_residuals = line_model.residuals(points) #type: ignore

            #find the error
            line_rmse=np.sqrt(np.mean(line_residuals ** 2))
        except Exception:
            line_rmse = float("inf")
        
        #try fit a circle
        try:
            circle_model,circle_inliers = ransac(
                points,CircleModel,min_samples=3,residual_threshold=0.02,max_trials=NUMBER_TRIALS_POLES
            )
            circle_residuals = circle_model.residuals(points) #type: ignore
            circle_rmse = np.sqrt(np.mean(circle_residuals ** 2))
            cx,cy = circle_model.center #type: ignore
            r = circle_model.radius #type: ignore
        except Exception:
            return None

        #see the spread of the clusters
        extent_x = points[:,0].max() - points[:,0].min()
        extent_y = points[:,1].max() - points[:,1].min()
        max_extent=max(extent_x,extent_y)

        #conditions for pole (radius between 1 - 15 cm), circle must fit better than line
        is_pole=(0.01<r<0.15 and max_extent<0.3 and circle_rmse<line_rmse)

        if not is_pole: return None

        inlier_points = points[circle_inliers]
        if len(inlier_points)<5: return None

        angles = np.arctan2(inlier_points[:,1]-cy,
                            inlier_points[:,0]-cx
                            )
        arc_span=angles.max()-angles.min()
        
        #if angle is less than 15 degrees
        if arc_span<np.deg2rad(15): return None

        #set angle to create arc
        theta = np.linspace(angles.min(),angles.max(),100)
        arc_x=cx + r*np.cos(theta)
        arc_y=cy + r*np.sin(theta)

        return {
            "arc_x": arc_x,
            "arc_y": arc_y,
            "inlier_points": inlier_points,
            "cx": cx,
            "cy": cy,
            "radius": r,
            "circle_rmse": circle_rmse,
            "line_rmse": line_rmse
        }

    def wall_ransac(data)->List[int]:
        """
        Docstring for wall_ransac
        
        :param data: Description
        """
        remaining_data=data.copy() #data to be used
        walls=[] #location of identified walls
        min_inliers= 15
        residual_threshold= 0.05
        while len(remaining_data)>min_inliers:
            model_robust, inliers=ransac(
                remaining_data,LineModelND,min_samples=5,residual_threshold=residual_threshold,max_trials=NUMBER_TRIALS_WALLS
            )
            inlier_points=remaining_data[inliers]
            if len(inlier_points) < min_inliers: # SUS
                break

            point_on_line= model_robust.origin #type: ignore
            direction=model_robust.direction #type: ignore

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


    #main pipeline
    df = df[df["y"]>-0.6] #just for placing rings cuz of noise
    y = df["x"]
    x = df["y"]

    #read raw values for segmentation
    raw = df["raw"].astype(float)
    data=np.column_stack([x,y])

    #segment everything
    segments = segment_jumps(raw.values)
    print(f"segments are: {len(segments)}")

    all_poles = []
    for segment_start, segment_end in segments:
        segment_points=data[segment_start:segment_end]

        if len(segment_points)<5: continue

    
        #making sure points are in the same segment
        clustering = DBSCAN(eps=0.04, min_samples=3).fit(segment_points)
        labels = clustering.labels_

        for label in set(labels) - {-1}:
            cluster_points = segment_points[labels == label]
            pole_info = classify_cluster(cluster_points)
            if pole_info is not None:
                all_poles.append(pole_info)
    
    #display the walls]
    wall_segments = wall_ransac(data)



    print(f"Poles found: {len(all_poles)}")


    #debug
    # print("Actual pole data")
    # pole_data:DataFrame = df[(df["y"]>0.08)&(df["y"]<0.2)]
    # print(pole_data.head(100))


    #plot stuff
    for i, pole in enumerate(all_poles):
        print(f"  Pole {i+1}: center=({pole['cx']:.3f}, {pole['cy']:.3f}), "
                f"radius={pole['radius']:.3f}m, "
                f"circle_rmse={pole['circle_rmse']:.4f}, line_rmse={pole['line_rmse']:.4f}")

    #plot
    plt.figure(figsize=(9, 9))
    plt.scatter(x, y, c="green", s=5, alpha=0.5, label="All points")


    for i, wall in enumerate(wall_segments):
        label = "Wall" if i == 0 else None
        plt.plot(wall["wall"][:, 0], wall["wall"][:, 1], c="blue", linewidth=2, label=label) #type: ignore
        
        #find the angle
        print(f"ANGLE OF WALL: {i} is {wall['angle']}")


    for i, pole in enumerate(all_poles):
        label = "Pole" if i == 0 else None
        plt.plot(pole["arc_x"], pole["arc_y"], color="red", linewidth=2, label=label)
        plt.scatter(pole["cx"], pole["cy"], c="red", marker='x', s=100, zorder=5)
        plt.scatter(pole["inlier_points"][:, 0], pole["inlier_points"][:, 1],
                    c="yellow", s=20, zorder=4, label="Pole inliers" if i == 0 else None)

    plt.axis("equal")
    plt.xlabel("X (meters)")
    plt.ylabel("Y (meters)")
    plt.title("LiDAR Detection: Poles (Red) & Walls (Blue)")
    plt.legend()

def v2_both_poles_and_walls(df:DataFrame):
    #walls
    MIN_WALL_LENGTH=0.25 #number of points for a valid wall
    DISTANCE_DIFFERENCE= 0.2
    NUMBER_TRIALS_WALLS=200
    RESIDUAL_THRESHOLD_WALLS=0.025 #how much distance from a line can be counted inlier
    MIN_WALL_POINTS=15 #min number of points for a wall to be called a wall
    MAX_NUMBER_WALLS=6 #maximum number of walls it will draw 

    #segmentation
    JUMP_THRESHOLD=0.08 #jump distance between points in the segment
    POINTS_PER_SEGMENT=3 #min number of points for segment

    #poles
    POLE_RMSE=0.1 #How much error allowed for a pole
    NUMBER_TRIALS_POLES=200
    MIN_POLE_POINTS=4 #min number of points counted as a pole
    RESIDUAL_THRESHOLD_POLES=0.02 #how much distance from a line can be counted inlier
    MIN_RADIUS=0.005 #min radius of a pole to be counted
    MAX_RADIUS=0.25 # max radius
    MIN_ARC_DEGREES=0.5

    def segment_jumps(raw_distances,jump_threshold=JUMP_THRESHOLD):
        """Basically it breaks different parts up by the jumps it sees in the distances"""
        
         #find the ddifferences and find indices where there are jumps
        distance_difference=np.abs(np.diff(raw_distances))
        jump_indices=np.where(distance_difference>jump_threshold)[0]

        segments=[]
        prev=0
        for j in jump_indices:
            if j-prev >= 3:
                segments.append((prev, j + 1))
            prev=j+1 #to go to next index
        
        #basically if a segment is more than at least like 3 points then count it
        if len(raw_distances)-prev>=POINTS_PER_SEGMENT:
            segments.append((prev,len(raw_distances)))
        
        return segments

    def classify_cluster(points): #this checks clusters for a pole
        """
        Try fit a circle circle, see which one works better and then fit the pole regarding it
        
        :param points: Description
        """
        #if not enough points, no ned
        if len(points)<MIN_POLE_POINTS: return None
        
        #try fit a line
        # try:
        #     line_model,line_inliers=ransac(
        #         points,LineModelND,min_samples=2,residual_threshold=0.02,max_trials=NUMBER_TRIALS_WALLS
        #     )
        #     line_residuals = line_model.residuals(points)
        #     #find the error
        #     line_rmse=np.sqrt(np.mean(line_residuals ** 2))

        # except Exception:
        #     line_rmse = float("inf")
        
        #try fit a circle
        try:
            circle_model,circle_inliers = ransac(
                points,CircleModel,min_samples=3,residual_threshold=RESIDUAL_THRESHOLD_POLES,max_trials=NUMBER_TRIALS_POLES
            )
            circle_residuals = circle_model.residuals(points) #type: ignore
            circle_rmse = np.sqrt(np.mean(circle_residuals ** 2))
            cx,cy = circle_model.center #type: ignore
            r = circle_model.radius #type: ignore
        except Exception:
            return None

        #see the spread of the clusters
        extent_x = points[:,0].max() - points[:,0].min()
        extent_y = points[:,1].max() - points[:,1].min()
        max_extent=max(extent_x,extent_y)

        #conditions for pole (radius between 1 - 15 cm), circle must fit better than line
        is_pole=(MIN_RADIUS<r<MAX_RADIUS and max_extent<0.3 and circle_rmse<POLE_RMSE)

        # if is_wall:
        #     walls=find_wall_smol(line_inliers=line_inliers,line_model=line_model,points=points)
        #     return {"type": "wall_list", "data": walls}
        
        if is_pole:
            inlier_points = points[circle_inliers]
            if len(inlier_points)<MIN_POLE_POINTS: return None

            angles = np.arctan2(inlier_points[:,1]-cy,
                                inlier_points[:,0]-cx
                                )
            arc_span=angles.max()-angles.min()
            
            #if angle is less than 15 degrees
            if arc_span<np.deg2rad(MIN_ARC_DEGREES): return None

            #set angle to create arc
            theta = np.linspace(angles.min(),angles.max(),100)
            arc_x=cx + r*np.cos(theta)
            arc_y=cy + r*np.sin(theta)

            return {
                "type":"pole",
                "arc_x": arc_x,
                "arc_y": arc_y,
                "inlier_points": inlier_points,
                "cx": cx,
                "cy": cy,
                "radius": r,
                "circle_rmse": circle_rmse,
                "inliers": circle_inliers
                }
        
        else: return None

    def wall_ransac(data) -> List[dict]:
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

    #start timer:
    start_time = time.time()

    #main pipeline
    df = df[df["y"]>-0.7] #just for placing rings cuz of noise
    y = df["x"]
    x = df["y"]

    #read raw values for segmentation
    raw = df["raw"].astype(float)
    data= np.column_stack([x,y])

    #segment everything
    segments = segment_jumps(raw.values)
    print(f"segments are: {len(segments)}")

    wall_segments=[]
    all_poles = []
    

    for segment_start, segment_end in segments:
        segment_points = data[segment_start:segment_end]
        if len(segment_points) < 5: continue

        # clustering = DBSCAN(eps=0.04, min_samples=3).fit(segment_points)
        # labels = clustering.labels_

        # for label in set(labels) - {-1}:
        #     cluster_points = segment_points[labels == label]


        res = classify_cluster(segment_points)
        
        if res:
            if res["type"] == "pole":
                all_poles.append(res)
            elif res["type"] == "wall_list" and res["data"] is not None:
                # Only extend if data actually exists
                wall_segments.extend(res["data"])

    plt.figure(figsize=(9, 9))
    plt.scatter(x, y, c="green", s=5, alpha=0.5, label="All points")


    pole_inlier_mask = np.zeros(len(data), dtype=bool)
    for i, pole in enumerate(all_poles):
        if 'cx' in pole:
            print(f"  Pole {i+1}: center=({pole['cx']:.3f}, {pole['cy']:.3f}), radius={pole['radius']:.3f}m")

            label = "Pole" if i == 0 else None
            plt.plot(pole["arc_x"], pole["arc_y"], color="red", linewidth=2, label=label)
            plt.scatter(pole["cx"], pole["cy"], c="red", marker='x', s=100, zorder=5)
            plt.scatter(pole["inlier_points"][:, 0], pole["inlier_points"][:, 1],
                        c="yellow", s=20, zorder=4, label="Pole inliers" if i == 0 else None)
            if "indices" in pole:
                pole_inlier_mask[pole["indices"]] = True

    
    remaining_data = data[~pole_inlier_mask]
    wall_segments.extend(wall_ransac(remaining_data))
    
    #plotting for Walls
    for i, wall in enumerate(wall_segments):
        label = "Wall" if i == 0 else None
        plt.plot(wall["wall"][:, 0], wall["wall"][:, 1], c="blue", linewidth=2, label=label)
        print(f"ANGLE OF WALL: {i} is {wall['angle']:.2f}")


    print(f"Poles found: {len(all_poles)}")


    #wall midpoints
    print("---- middle wall ------")
    back_wall=find_back_wall_midpoints(wall_segments)
    print(f"mid_point x: {back_wall['mid_x']}\t mid_point y: {back_wall['mid_y']}")
    print(f"angle of wall is {back_wall['angle']}")
    plt.scatter(back_wall["mid_x"],back_wall["mid_y"],c="red",marker="x",s=50,zorder=5 )



    #pole midpoints
    print("----- pole center ----")
    if all_poles:
        pole_center=find_poles_midpoint(all_poles)
        print(f"mid_point x: {pole_center['mid_x']}\t mid_point y: {pole_center['mid_y']}")
        if pole_center["angle"] is not None:
            print(f"angle of pole is {pole_center['angle']}")
        plt.scatter(pole_center["mid_x"],pole_center["mid_y"],c="green",marker="x",s=50,zorder=5 )


    #debug
    # print("Actual pole data")
    # pole_data:DataFrame = df[(df["y"]>0.08)&(df["y"]<0.2)]
    # print(pole_data.head(100))
    processing_time = (time.time() - start_time) #in seconds
    print(f"\n\n------ processing_time is: {processing_time}\n frequency is {1/processing_time} ")
    #plot stuff
    plt.axis("equal")
    plt.xlabel("X (meters)")
    plt.ylabel("Y (meters)")
    plt.title("LiDAR Detection: Poles (Red) & Walls (Blue)")
    plt.legend()

def v2_both_poles_and_walls_no_plot(df:DataFrame) -> dict:
    #walls
    MIN_WALL_LENGTH=0.25 #number of points for a valid wall
    DISTANCE_DIFFERENCE= 0.2
    NUMBER_TRIALS_WALLS=200
    RESIDUAL_THRESHOLD_WALLS=0.025 #how much distance from a line can be counted inlier
    MIN_WALL_POINTS=15 #min number of points for a wall to be called a wall
    MAX_NUMBER_WALLS=6 #maximum number of walls it will draw 

    #segmentation
    JUMP_THRESHOLD=0.08 #jump distance between points in the segment
    POINTS_PER_SEGMENT=3 #min number of points for segment

    #poles
    POLE_RMSE=0.1 #How much error allowed for a pole
    NUMBER_TRIALS_POLES=200
    MIN_POLE_POINTS=4 #min number of points counted as a pole
    RESIDUAL_THRESHOLD_POLES=0.02 #how much distance from a line can be counted inlier
    MIN_RADIUS=0.005 #min radius of a pole to be counted
    MAX_RADIUS=0.25 # max radius
    MIN_ARC_DEGREES=0.5

    def segment_jumps(raw_distances,jump_threshold=JUMP_THRESHOLD):
        """Basically it breaks different parts up by the jumps it sees in the distances"""
        
         #find the ddifferences and find indices where there are jumps
        distance_difference=np.abs(np.diff(raw_distances))
        jump_indices=np.where(distance_difference>jump_threshold)[0]

        segments=[]
        prev=0
        for j in jump_indices:
            if j-prev >= 3:
                segments.append((prev, j + 1))
            prev=j+1 #to go to next index
        
        #basically if a segment is more than at least like 3 points then count it
        if len(raw_distances)-prev>=POINTS_PER_SEGMENT:
            segments.append((prev,len(raw_distances)))
        
        return segments

    def classify_cluster(points): #this checks clusters as wall or pole
        """
        Try fit a circle circle, see which one works better and then fit the pole regarding it
        
        :param points: Description
        """
        #if not enough points, no ned
        if len(points)<MIN_POLE_POINTS: return None
        
        #try fit a line
        # try:
        #     line_model,line_inliers=ransac(
        #         points,LineModelND,min_samples=2,residual_threshold=0.02,max_trials=NUMBER_TRIALS_WALLS
        #     )
        #     line_residuals = line_model.residuals(points)
        #     #find the error
        #     line_rmse=np.sqrt(np.mean(line_residuals ** 2))

        # except Exception:
        #     line_rmse = float("inf")
        
        #try fit a circle
        try:
            circle_model,circle_inliers = ransac(
                points,CircleModel,min_samples=3,residual_threshold=RESIDUAL_THRESHOLD_POLES,max_trials=NUMBER_TRIALS_POLES
            )
            circle_residuals = circle_model.residuals(points) #type: ignore
            circle_rmse = np.sqrt(np.mean(circle_residuals ** 2))
            cx,cy = circle_model.center #type: ignore
            r = circle_model.radius #type: ignore
        except Exception:
            return None

        #see the spread of the clusters
        extent_x = points[:,0].max() - points[:,0].min()
        extent_y = points[:,1].max() - points[:,1].min()
        max_extent=max(extent_x,extent_y)

        #conditions for pole (radius between 1 - 15 cm), circle must fit better than line
        is_pole=(MIN_RADIUS<r<MAX_RADIUS and max_extent<0.3 and circle_rmse<POLE_RMSE)

        # if is_wall:
        #     walls=find_wall_smol(line_inliers=line_inliers,line_model=line_model,points=points)
        #     return {"type": "wall_list", "data": walls}
        
        if is_pole:
            inlier_points = points[circle_inliers]
            if len(inlier_points)<MIN_POLE_POINTS: return None

            angles = np.arctan2(inlier_points[:,1]-cy,
                                inlier_points[:,0]-cx
                                )
            arc_span=angles.max()-angles.min()
            
            #if angle is less than 15 degrees
            if arc_span<np.deg2rad(15): return None

            #set angle to create arc
            theta = np.linspace(angles.min(),angles.max(),100)
            arc_x=cx + r*np.cos(theta)
            arc_y=cy + r*np.sin(theta)

            return {
                "type":"pole",
                "arc_x": arc_x,
                "arc_y": arc_y,
                "inlier_points": inlier_points,
                "cx": cx,
                "cy": cy,
                "radius": r,
                "circle_rmse": circle_rmse,
                "inliers": circle_inliers
                }
        
        else: return None

    def wall_ransac(data) -> List[dict]:
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

    #start timer:
    start_time = time.time()

    #main pipeline
    df = df[df["y"]>-0.6] #just for placing rings cuz of noise
    y = df["x"]
    x = df["y"]

    #read raw values for segmentation
    raw = df["raw"].astype(float)
    data= np.column_stack([x,y])

    #segment everything
    segments = segment_jumps(raw.values)
    print(f"segments are: {len(segments)}")

    wall_segments=[]
    all_poles = []
    

    for segment_start, segment_end in segments:
        segment_points = data[segment_start:segment_end]
        if len(segment_points) < 5: continue

        # clustering = DBSCAN(eps=0.04, min_samples=3).fit(segment_points)
        # labels = clustering.labels_

        # for label in set(labels) - {-1}:
        #     cluster_points = segment_points[labels == label]


        res = classify_cluster(segment_points)
        
        if res:
            if res["type"] == "pole":
                all_poles.append(res)
            elif res["type"] == "wall_list" and res["data"] is not None:
                # Only extend if data actually exists
                wall_segments.extend(res["data"])

    pole_inlier_mask = np.zeros(len(data), dtype=bool)
    for i, pole in enumerate(all_poles):
        if 'cx' in pole:
            print(f"  Pole {i+1}: center=({pole['cx']:.3f}, {pole['cy']:.3f}), radius={pole['radius']:.3f}m")

            if "indices" in pole:
                pole_inlier_mask[pole["indices"]] = True

    
    remaining_data = data[~pole_inlier_mask]
    wall_segments.extend(wall_ransac(remaining_data))
    
    #plotting for Walls
    for i, wall in enumerate(wall_segments):
        label = "Wall" if i == 0 else None
        print(f"ANGLE OF WALL: {i} is {wall['angle']:.2f}")


    print(f"Poles found: {len(all_poles)}")

    processing_time = (time.time() - start_time) #in seconds
    print(f"\n\n------ processing_time is: {processing_time}\n frequency is {1/processing_time} ")

    return {
        'walls': wall_segments,
        'poles': all_poles,
        'processing_time': processing_time,
        'frequency': 1/processing_time
    }

def v3_both_poles_and_walls(df:DataFrame):
    #walls
    MIN_WALL_LENGTH=0.25 #number of points for a valid wall
    DISTANCE_DIFFERENCE= 0.2
    NUMBER_TRIALS_WALLS=100
    RESIDUAL_THRESHOLD_WALLS=0.025 #how much distance from a line can be counted inlier
    MIN_WALL_POINTS=15 #min number of points for a wall to be called a wall
    MAX_NUMBER_WALLS=6 #maximum number of walls it will draw 

    #segmentation
    JUMP_THRESHOLD=0.08 #jump distance between points in the segment
    POINTS_PER_SEGMENT=3 #min number of points for segment

    #poles
    POLE_RMSE=0.3 #How much error allowed for a pole
    NUMBER_TRIALS_POLES=500
    MIN_POLE_POINTS=4 #min number of points counted as a pole
    RESIDUAL_THRESHOLD_POLES=0.2 #how much distance from a line can be counted inlier
    MIN_RADIUS=0.005 #min radius of a pole to be counted
    MAX_RADIUS=0.25 # max radius
    MIN_ARC_DEGREES=0.05

    MIN_POINTS = 6
    MIN_RADIUS = 0.005
    MAX_RADIUS = 0.15
    MIN_ARC_DEG = 10

    def segment_jumps(raw_distances,jump_threshold=JUMP_THRESHOLD):
        """Basically it breaks different parts up by the jumps it sees in the distances"""
        
         #find the ddifferences and find indices where there are jumps
        distance_difference=np.abs(np.diff(raw_distances))
        jump_indices=np.where(distance_difference>jump_threshold)[0]

        segments=[]
        prev=0
        for j in jump_indices:
            if j-prev >= 3:
                segments.append((prev, j + 1))
            prev=j+1 #to go to next index
        
        #basically if a segment is more than at least like 3 points then count it
        if len(raw_distances)-prev>=POINTS_PER_SEGMENT:
            segments.append((prev,len(raw_distances)))
        
        return segments

    def classify_cluster(points):
        """
        Smart version: Tries circle fit, then parabola, then geometry
        Better detection rate for parabolic arcs
        """
        
        
        if len(points) < MIN_POINTS:
            return None
        
        # Strategy 1: Try lenient circle fit
        result = _try_circle_fit(points, MIN_POINTS, MIN_RADIUS, MAX_RADIUS, MIN_ARC_DEG)
        if result:
            return result
        
        # Strategy 2: Try parabola fit (for distorted arcs)
        result = _try_parabola_fit(points, MIN_RADIUS, MAX_RADIUS)
        if result:
            return result
        
        # Strategy 3: Geometry-only detection
        result = _try_geometry_detection(points, MIN_RADIUS, MAX_RADIUS, MIN_ARC_DEG)
        if result:
            return result
        
        return None

    def _try_circle_fit(points, min_points, min_radius, max_radius, min_arc_deg):
        """Try circle fitting with lenient parameters"""
        try:
            circle_model, circle_inliers = ransac(
                points, CircleModel,
                min_samples=3,
                residual_threshold=0.04,
                max_trials=100
            )
            
            cx, cy = circle_model.params[0], circle_model.params[1]
            r = circle_model.params[2]
            
            if not (min_radius < r < max_radius):
                return None
            
            inlier_points = points[circle_inliers]
            if len(inlier_points) < min_points:
                return None
            
            angles = np.arctan2(inlier_points[:, 1] - cy,
                            inlier_points[:, 0] - cx)
            arc_span_deg = np.degrees(angles.max() - angles.min())
            
            if arc_span_deg < min_arc_deg:
                return None
            
            residuals = circle_model.residuals(points)
            rmse = np.sqrt(np.mean(residuals ** 2))
            
            return {
                "type": "pole",
                "cx": cx,
                "cy": cy,
                "radius": r,
                "circle_rmse": rmse,
                "arc_span_deg": arc_span_deg,
                "method": "circle",
                "inliers": circle_inliers
            }
        except:
            return None

    def _try_parabola_fit(points, min_radius, max_radius):
        """Fit parabola for distorted/parabolic arcs"""
        if len(points) < 4:
            return None
        
        try:
            # Center and rotate points
            mean = points.mean(axis=0)
            centered = points - mean
            _, _, vt = np.linalg.svd(centered, full_matrices=False)
            rotated = centered @ vt.T
            
            # Fit parabola: y = ax² + bx + c
            x = rotated[:, 0]
            y = rotated[:, 1]
            A = np.column_stack([x**2, x, np.ones_like(x)])
            coeffs, residuals, _, _ = np.linalg.lstsq(A, y, rcond=None)
            
            if len(residuals) > 0:
                fit_error = np.sqrt(residuals[0] / len(points))
                
                if fit_error < 0.05:  # Good parabolic fit
                    # Find vertex (center estimate)
                    a, b, c = coeffs
                    vertex_x = -b / (2 * a) if abs(a) > 1e-6 else 0
                    vertex_y = a * vertex_x**2 + b * vertex_x + c
                    
                    vertex_original = (np.array([vertex_x, vertex_y]) @ vt) + mean
                    
                    # Estimate radius
                    distances = np.linalg.norm(points - vertex_original, axis=1)
                    estimated_radius = distances.mean()
                    
                    if min_radius < estimated_radius < max_radius:
                        angles = np.arctan2(points[:, 1] - vertex_original[1],
                                        points[:, 0] - vertex_original[0])
                        arc_span_deg = np.degrees(angles.max() - angles.min())
                        
                        return {
                            "type": "pole",
                            "cx": vertex_original[0],
                            "cy": vertex_original[1],
                            "radius": estimated_radius,
                            "circle_rmse": fit_error,
                            "arc_span_deg": arc_span_deg,
                            "method": "parabola"
                        }
        except:
            pass
        
        return None

    def _try_geometry_detection(points, min_radius, max_radius, min_arc_deg):
        """Detect by geometry alone - no curve fitting"""
        if len(points) < 4:
            return None
        
        centroid = points.mean(axis=0)
        
        # Check extent
        extent_x = points[:, 0].max() - points[:, 0].min()
        extent_y = points[:, 1].max() - points[:, 1].min()
        if max(extent_x, extent_y) > 0.4:
            return None
        
        # Check radius
        distances = np.linalg.norm(points - centroid, axis=1)
        mean_radius = distances.mean()
        std_radius = distances.std()
        
        if not (min_radius < mean_radius < max_radius):
            return None
        
        # Check circularity
        circularity = std_radius / (mean_radius + 1e-6)
        if circularity > 0.4:  # Not circular enough
            return None
        
        # Check arc span
        angles = np.arctan2(points[:, 1] - centroid[1],
                        points[:, 0] - centroid[0])
        arc_span_deg = np.degrees(angles.max() - angles.min())
        
        if arc_span_deg < min_arc_deg:
            return None
        
        return {
            "type": "pole",
            "cx": centroid[0],
            "cy": centroid[1],
            "radius": mean_radius,
            "circle_rmse": std_radius,
            "arc_span_deg": arc_span_deg,
            "method": "geometry"
        }

    def wall_ransac(data) -> List[dict]:
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

    #start timer:
    start_time = time.time()

    #main pipeline
    df = df[df["y"]>-0.6] #just for placing rings cuz of noise
    y = df["x"]
    x = df["y"]

    #read raw values for segmentation
    raw = df["raw"].astype(float)
    data= np.column_stack([x,y])

    # ============================================================
    # STEP 1: Detect walls first (dominant linear features)
    # ============================================================
    wall_segments = wall_ransac(data)
    print(f"Walls detected: {len(wall_segments)}")

    # ============================================================
    # STEP 2: Build a mask (True/False array) of points that belong to walls
    # For each wall, we check every point: is it close to this wall line?
    # If yes, mark it True so we can remove it later and find poles in the leftovers
    # ============================================================
    wall_inlier_mask = np.zeros(len(data), dtype=bool)  # all False initially

    for wall in wall_segments:
        wall_endpoints = wall["wall"]  # the 2 endpoints of the wall, shape (2, 2)
        
        # direction vector: points from endpoint[0] to endpoint[1]
        direction = wall_endpoints[1] - wall_endpoints[0]
        wall_len = np.linalg.norm(direction)  # length of the wall
        if wall_len < 1e-9:  # skip walls that are basically zero length
            continue
        direction_norm = direction / wall_len  # unit vector along the wall

        # For every data point, compute a vector from wall start to that point
        vecs = data - wall_endpoints[0]

        # "t" = how far along the wall each point is (dot product with direction)
        #   t=0 means at wall start, t=wall_len means at wall end
        #   basically: t[i] = vecs[i][0]*direction_norm[0] + vecs[i][1]*direction_norm[1]
        t = np.dot(vecs, direction_norm)

        # "perp" = the leftover vector after removing the along-wall component
        # np.outer(t, direction_norm) stretches each t value into a 2D vector along the wall
        # subtracting that from vecs gives the perpendicular (sideways) part
        perp = vecs - np.outer(t, direction_norm)
        perp_dist = np.linalg.norm(perp, axis=1)  # perpendicular distance from wall line

        # A point belongs to this wall if:
        #   1) it's close enough to the wall line (perp_dist is small)
        #   2) it's between the two endpoints (with a small 5cm margin)
        margin = 0.05  # 5cm margin past the endpoints
        is_close_to_line = perp_dist < (RESIDUAL_THRESHOLD_WALLS * 2)
        is_between_endpoints = (t > -margin) & (t < wall_len + margin)
        on_wall = is_close_to_line & is_between_endpoints

        # Mark these points as wall points (combine with previous walls)
        # if on_wall[i] is True OR wall_inlier_mask[i] is already True, keep it True
        wall_inlier_mask = wall_inlier_mask | on_wall

    # ============================================================
    # STEP 3: Cluster remaining (non-wall) points with DBSCAN
    # ============================================================
    remaining_data = data[~wall_inlier_mask]
    print(f"Points after wall removal: {len(remaining_data)} / {len(data)}")

    all_poles = []
    if len(remaining_data) >= MIN_POINTS:
        clustering = DBSCAN(eps=0.03, min_samples=3).fit(remaining_data)
        labels = clustering.labels_

        for label in set(labels) - {-1}:
            cluster_points = remaining_data[labels == label]
            if len(cluster_points) < MIN_POINTS:
                continue

            # ============================================================
            # STEP 4: Classify each cluster as a potential pole
            # ============================================================
            res = classify_cluster(cluster_points)
            if res and res["type"] == "pole":
                all_poles.append(res)

    # ============================================================
    # Plotting
    # ============================================================
    plt.figure(figsize=(9, 9))
    plt.scatter(x, y, c="green", s=5, alpha=0.5, label="All points")

    # Plot remaining (non-wall) points for debugging
    if len(remaining_data) > 0:
        plt.scatter(remaining_data[:, 0], remaining_data[:, 1],
                    c="orange", s=10, alpha=0.7, label="Non-wall points")

    for i, pole in enumerate(all_poles):
        if 'cx' in pole:
            print(f"  Pole {i+1}: center=({pole['cx']:.3f}, {pole['cy']:.3f}), radius={pole['radius']:.3f}m, method={pole.get('method','?')}")

            label = "Pole" if i == 0 else None
            plt.scatter(pole["cx"], pole["cy"], c="black", marker='x', s=200, zorder=5, label=label)

    #plotting for Walls
    for i, wall in enumerate(wall_segments):
        label = "Wall" if i == 0 else None
        plt.plot(wall["wall"][:, 0], wall["wall"][:, 1], c="blue", linewidth=2, label=label)
        print(f"ANGLE OF WALL: {i} is {wall['angle']:.2f}")

    print(f"Poles found: {len(all_poles)}")

    #wall midpoints
    print("---- middle wall ------")
    back_wall=find_back_wall_midpoints(wall_segments)
    print(f"mid_point x: {back_wall['mid_x']}\t mid_point y: {back_wall['mid_y']}")
    print(f"angle of wall is {back_wall['angle']}")
    plt.scatter(back_wall["mid_x"],back_wall["mid_y"],c="red",marker="x",s=50,zorder=5 )


    #pole midpoints
    print("----- pole center ----")
    if all_poles:
        pole_center=find_poles_midpoint(all_poles)
        print(f"mid_point x: {pole_center['mid_x']}\t mid_point y: {pole_center['mid_y']}")
        if pole_center["angle"] is not None:
            print(f"angle of pole is {pole_center['angle']}")
        plt.scatter(pole_center["mid_x"],pole_center["mid_y"],c="green",marker="x",s=50,zorder=5 )

    processing_time = (time.time() - start_time) #in seconds
    print(f"\n\n------ processing_time is: {processing_time}\n\n frequency is {1/processing_time} \n\n")
    #plot stuff
    plt.axis("equal")
    plt.xlabel("X (meters)")
    plt.ylabel("Y (meters)")
    plt.title("LiDAR Detection: Poles (Red) & Walls (Blue)")
    plt.legend()

    return {
        'walls': wall_segments,
        'poles': all_poles,
        'processing_time': processing_time,
        'frequency': 1/processing_time
    }

def v3_both_poles_and_walls_no_plot(df:DataFrame) -> dict:
    """
    Same as v3_both_poles_and_walls but without plotting.
    Used by ScanMatchingTracker for full detection.
    
    Pipeline: walls first via RANSAC -> remove wall points -> DBSCAN cluster remaining -> classify as poles
    """
    #walls
    MIN_WALL_LENGTH=0.25
    DISTANCE_DIFFERENCE= 0.2
    NUMBER_TRIALS_WALLS=100
    RESIDUAL_THRESHOLD_WALLS=0.025
    MIN_WALL_POINTS=15
    MAX_NUMBER_WALLS=6

    #poles
    MIN_POINTS = 6
    MIN_RADIUS = 0.005
    MAX_RADIUS = 0.15
    MIN_ARC_DEG = 10

    def wall_ransac_inner(data) -> List[dict]:
        remaining_data=data.copy()
        walls=[]
        min_inliers= MIN_WALL_POINTS
        while len(remaining_data)>min_inliers and len(walls)<=MAX_NUMBER_WALLS:
            try:
                model_robust, inliers=ransac(
                    remaining_data,LineModelND,min_samples=5,
                    residual_threshold=RESIDUAL_THRESHOLD_WALLS,max_trials=NUMBER_TRIALS_WALLS
                )
                if model_robust is None:
                    break
            except Exception:
                break

            inlier_points=remaining_data[inliers]
            if len(inlier_points) < min_inliers:
                break

            point_on_line= model_robust.origin
            direction=model_robust.direction
            t=np.dot(inlier_points - point_on_line, direction)
            sorted_indices = np.argsort(t)
            t_sorted=t[sorted_indices]
            gaps=np.diff(t_sorted)
            split_indices = np.where(gaps>DISTANCE_DIFFERENCE)[0] + 1
            clusters = np.split(t_sorted,split_indices)

            angle_rad = np.arctan2(direction[1],direction[0])
            angle_deg = np.degrees(angle_rad)

            for cluster in clusters:
                if len(cluster)<2:
                    continue
                t_min, t_max = cluster.min(),cluster.max()
                wall_length=abs(t_max-t_min)
                if wall_length> MIN_WALL_LENGTH:
                    wall_endpoints = point_on_line + np.outer([t_min,t_max],direction)
                    walls.append({"wall":wall_endpoints,"angle":angle_deg})

            remaining_data = remaining_data[~inliers]
        return walls

    def classify_pole(points):
        """Try circle fit on a cluster to see if it's a pole"""
        if len(points) < MIN_POINTS:
            return None
        try:
            circle_model, circle_inliers = ransac(
                points, CircleModel,
                min_samples=3, residual_threshold=0.04, max_trials=100
            )
            cx, cy = circle_model.params[0], circle_model.params[1]
            r = circle_model.params[2]

            if not (MIN_RADIUS < r < MAX_RADIUS):
                return None

            inlier_points = points[circle_inliers]
            if len(inlier_points) < MIN_POINTS:
                return None

            angles = np.arctan2(inlier_points[:, 1] - cy, inlier_points[:, 0] - cx)
            arc_span_deg = np.degrees(angles.max() - angles.min())
            if arc_span_deg < MIN_ARC_DEG:
                return None

            residuals = circle_model.residuals(points)
            rmse = np.sqrt(np.mean(residuals ** 2))

            return {
                "type": "pole", "cx": cx, "cy": cy, "radius": r,
                "circle_rmse": rmse, "arc_span_deg": arc_span_deg, "method": "circle"
            }
        except:
            return None

    start_time = time.time()

    df = df[df["y"]>-0.6]
    y = df["x"]
    x = df["y"]
    data = np.column_stack([x, y])

    # STEP 1: Detect walls
    wall_segments = wall_ransac_inner(data)

    # STEP 2: Remove wall points
    wall_inlier_mask = np.zeros(len(data), dtype=bool)
    for wall in wall_segments:
        wall_endpoints = wall["wall"]
        direction = wall_endpoints[1] - wall_endpoints[0]
        wall_len = np.linalg.norm(direction)
        if wall_len < 1e-9:
            continue
        direction_norm = direction / wall_len
        vecs = data - wall_endpoints[0]
        t = np.dot(vecs, direction_norm)
        perp = vecs - np.outer(t, direction_norm)
        perp_dist = np.linalg.norm(perp, axis=1)
        margin = 0.05
        is_close = perp_dist < (RESIDUAL_THRESHOLD_WALLS * 2)
        is_between = (t > -margin) & (t < wall_len + margin)
        wall_inlier_mask = wall_inlier_mask | (is_close & is_between)

    # STEP 3: Cluster remaining and find poles
    remaining_data = data[~wall_inlier_mask]
    all_poles = []
    if len(remaining_data) >= MIN_POINTS:
        clustering = DBSCAN(eps=0.03, min_samples=3).fit(remaining_data)
        labels = clustering.labels_
        for label in set(labels) - {-1}:
            cluster_points = remaining_data[labels == label]
            if len(cluster_points) < MIN_POINTS:
                continue
            res = classify_pole(cluster_points)
            if res and res["type"] == "pole":
                all_poles.append(res)

    processing_time = (time.time() - start_time)
    return {
        'walls': wall_segments,
        'poles': all_poles,
        'processing_time': processing_time,
        'frequency': 1/processing_time
    }


#----------------------
#Incremental tracker

@dataclass #TODO: wut this
class Wall:
    """
    Walls
    """
    endpoints: np.ndarray
    angle: float
    confidence: float = 1.0
    last_seen: int = 0

@dataclass
class Pole:
    """
    Pole
    """
    cx: float
    cy: float
    radius: float
    confidence: float = 1
    last_seen: int = 0

#scan matching

#walls
MIN_WALL_LENGTH=0.25 #number of points for a valid wall
DISTANCE_DIFFERENCE= 0.2
NUMBER_TRIALS_WALLS=200
RESIDUAL_THRESHOLD_WALLS=0.025 #how much distance from a line can be counted inlier
MIN_WALL_POINTS=15 #min number of points for a wall to be called a wall
MAX_NUMBER_WALLS=6 #maximum number of walls it will draw 

#segmentation
JUMP_THRESHOLD=0.08 #jump distance between points in the segment
POINTS_PER_SEGMENT=5 #min number of points for segment
MIN_SEGMENT_POINTS=5

#poles
POLE_RMSE=0.1 #How much error allowed for a pole
NUMBER_TRIALS_POLES=200
MIN_POLE_POINTS=4 #min number of points counted as a pole
RESIDUAL_THRESHOLD_POLES=0.02 #how much distance from a line can be counted inlier
MIN_RADIUS=0.005 #min radius of a pole to be counted
MAX_RADIUS=0.25 # max radius
MIN_ARC_DEGREES=0.5

WEIGHT = 0.01
NEW_DISTANCE_JUMPS_WALL = 0.3
NEW_DISTANCE_JUMPS_POLE = 0.2
class ScanMatchingTracker:

    """
    Tracker that estimates motion
    """

    def __init__(self) -> None:

        #how many walls we know about
        self.known_walls: List[Wall] = []
        self.known_poles: List[Pole] = []

        #estimated motion for robot
        self.estimated_dx = 0.0
        self.estimated_dy = 0.0
        self.estimated_dtheta = 0.0

        #frame counter to see how much time has passed
        self.frame_count = 0
        self.full_frame_count = 10 #number of frames till we do a full rescan

        #this allows us to use prev logic for a total rescan (v3 = walls-first approach)
        self.full_detector = v3_both_poles_and_walls_no_plot

    def proces_frame(self, df: DataFrame) -> Dict:
        """
        every scan u see, process the data

        Check -> need full scan or no
            1) if yes then redo everything from full detector
            2) if no then use a fast tracker
            
        :param self: Description
        :param df: Description
        :type df: DataFrame
        :return: Description
        :rtype: Dict[Any, Any]
        """

        start_time = time.time()
        self.frame_count+=1



        #check do I need full recheck

        if self.full_detect_check():
            result = self.full_detection(df)
            mode = "full_detect"
            self.estimated_dx=0.0
            self.estimated_dtheta=0.0
            self.estimated_dy=0.0

        else: #if we dont need full scan then can just do fast checking with calc
            result = self.fast_detect(df)
            mode="fast_detect"

        #can delete, how fast it is
        process_time = time.time() - start_time

        return{
            "walls": result["walls"],
            "poles": result["poles"],
            "mode": mode,
            "estimated_motion":{
                "dx":self.estimated_dx,
                "dy":self.estimated_dy,
                "dtheta":self.estimated_dtheta
            },
            "processing_time": process_time,
            "frame_count": self.frame_count
        }
    
    def full_detect_check(self):
        """Decide if we need to check everything from begining"""
        
        #if we have reached full number of frames, restart the check
        if self.frame_count % self.full_frame_count == 0:
            return True
        
        #TODO: might wanna change this
        if len(self.known_walls) < 2:
            return True
        
        return False
    
    def full_detection(self,df: DataFrame) -> Dict:
        """ how to process the actual full detection """
        #run the full detection
        result = self.full_detector(df)
        self.known_poles=[]
        self.known_walls = []

        for wall in result["walls"]:
            self.known_walls.append(
                Wall(
                    endpoints=wall["wall"],
                    angle=wall["angle"],
                    confidence=1.0,
                    last_seen=self.frame_count
                )
            )
        
        for pole in result["poles"]:
            self.known_poles.append(
                Pole(
                    cx= pole["cx"],
                    cy= pole["cy"],
                    radius=pole["radius"],
                    confidence=1.0,
                    last_seen=self.frame_count
                )
            )
        return {
            "walls": result["walls"],
            "poles": result["poles"]
        }


    def fast_detect(self,df: DataFrame) -> dict:
        """
        Fast tracking: 


        """

        # check current frame
        walls_now, poles_now = self.detect_features_fast(df)

        #match features to known stuff
        wall_matches = self.match_features(walls_now, self.known_walls, feature_type = "wall")
        pole_matches = self.match_features(poles_now, self.known_poles, feature_type= "pole")

        #estimate where robot is from the scan
        self.estimate_motion(wall_matches,pole_matches)

        #update our known wall and stuff
        self.known_walls= walls_now
        self.known_poles= poles_now

        #put everything inside list
        walls_dict = [{"wall": w.endpoints, "angle": w.angle} for w in walls_now]
        poles_dict = [{"cx": p.cx, "cy": p.cy, "radius": p.radius} for p in poles_now]

        return {
            "walls": walls_dict,
            "poles": poles_dict
        }

    def detect_features_fast(self, df: DataFrame) -> tuple[List[Wall], List[Pole]]:
        """
        Fast detection using walls-first approach (no RANSAC, uses SVD + DBSCAN).
        
        Pipeline:
        1. Segment by jumps (to separate far-apart objects)
        2. Fit walls to large segments using SVD (fast, no RANSAC)
        3. Remove wall inlier points from the data
        4. Cluster remaining points with DBSCAN
        5. Fit circles to each cluster to find poles
        """

        x = df["y"].values
        y = df["x"].values
        points = np.column_stack([x, y])  # type: ignore
        raw = df["raw"].astype(float).values

        # Segment by jumps first (to separate far-apart objects)
        segments = self.segment_by_jumps(raw)

        walls = []

        # STEP 1: Find walls in each segment using SVD (fast)
        for start, end in segments:
            segment_points = points[start:end]
            if len(segment_points) < POINTS_PER_SEGMENT:
                continue

            # Check if this segment looks like a wall (long and thin)
            feature = self.classify_geometry(segment_points)
            if feature == "wall":
                wall = self.fit_wall(segment_points)
                if wall:
                    walls.append(wall)

        # STEP 2: Remove points that belong to walls
        wall_inlier_mask = np.zeros(len(points), dtype=bool)
        for wall in walls:
            wall_endpoints = wall.endpoints  # shape (2, 2)
            direction = wall_endpoints[1] - wall_endpoints[0]
            wall_len = np.linalg.norm(direction)
            if wall_len < 1e-9:
                continue
            direction_norm = direction / wall_len

            # How far along the wall + how far away from the wall
            vecs = points - wall_endpoints[0]
            t = np.dot(vecs, direction_norm)
            perp = vecs - np.outer(t, direction_norm)
            perp_dist = np.linalg.norm(perp, axis=1)

            # Mark points close to the wall line and between endpoints
            margin = 0.05
            is_close = perp_dist < (RESIDUAL_THRESHOLD_WALLS * 2)
            is_between = (t > -margin) & (t < wall_len + margin)
            wall_inlier_mask = wall_inlier_mask | (is_close & is_between)

        # STEP 3: Cluster remaining points with DBSCAN and fit poles
        remaining_points = points[~wall_inlier_mask]
        poles = []

        if len(remaining_points) >= MIN_POLE_POINTS:
            clustering = DBSCAN(eps=0.03, min_samples=3).fit(remaining_points)
            labels = clustering.labels_

            for label in set(labels) - {-1}:
                cluster_points = remaining_points[labels == label]
                if len(cluster_points) < MIN_POLE_POINTS:
                    continue

                # Try to fit a pole (circle) to this cluster
                pole = self.fit_pole(cluster_points)
                if pole:
                    poles.append(pole)

        return walls, poles

    def segment_by_jumps(self, raw_distances, threshold = JUMP_THRESHOLD) -> List[tuple[int,int]]:
        """ cut into segments """
        
        #calc difference between raw values
        diff = np.abs(np.diff(raw_distances))
        jump_indices = np.where(diff>threshold)[0]

        segments = []
        prev = 0

        #find the different indices for splits
        for j in jump_indices:
            if j-prev>=MIN_SEGMENT_POINTS:
                segments.append((prev, j+1))
            prev=j+1

        #for the final segment
        if len(raw_distances) - prev>=3:
            segments.append((prev, len(raw_distances)))
        
        return segments


    def classify_geometry(self,points: np.ndarray) -> str:
        """ checks if something is a wall or pole then returns name of what it is"""
        #if the thing is too small to identify yk
        if len(points)<5:
            return "unknown"
        
        #all the ranges of x values
        extent_x = points[:, 0].max() - points[:, 0].min()

        #all the ranges of y values
        extent_y = points[:, 1].max() - points[:, 1].min()
        
        #the max extent of both of those
        max_extent = max(extent_x,extent_y)
        
        #the smallest values of all of those
        min_extent = min(extent_x,extent_y)

        #we find the aspect ratio
        aspect_ratio = max_extent/ (min_extent + 1e-6)
        
        #circle
        #aspect ratio tells us if its long and thin (wall) or fat and stocky (pole)
        central = points.mean(axis = 0)
        distances = np.linalg.norm(points - central,axis=1)
        mean_radius= distances.mean()
        std_radius = distances.std()

        # if pole
        if (max_extent < 0.3 and
            MIN_RADIUS < mean_radius < MAX_RADIUS and
            std_radius / (mean_radius + 1e-6) < 0.3):
            return "pole"

        if aspect_ratio > 3.0 and max_extent > 0.25:
            return "wall"
        
        return "unknown"

    def fit_wall(self, points: np.ndarray) -> Optional[Wall]:
        """ Fit wall bro """
        
        #if not enough 
        if len(points) < 5:
            return None
        
        # the points in the middle
        mean = points.mean(axis=0)
        centered = points - mean

        #math time
        #svd gets the matrix of points, and vt is the directions, and the first component is max variance
        #so max variance is line direction
        unused,unused, vt = np.linalg.svd(centered,full_matrices=False)
        direction = vt[0]

        #find the endpoints
        t= np.dot(centered, direction)
        t_min, t_max = t.min(), t.max()
        if abs(t_max-t_min) < MIN_WALL_LENGTH:
            return None

        endpoints = mean + np.outer([t_min,t_max], direction)
        angle = np.degrees(np.arctan2(direction[1],direction[0]))


        return Wall(
            endpoints=endpoints,
            angle=angle,
            confidence=1.0,
            last_seen=self.frame_count
        )

    def fit_pole(self, points: np.ndarray) -> Optional[Pole]:
        """ fit pole """

        #if enough points:
        if len(points) < MIN_SEGMENT_POINTS:
            return None
        
        try:
            from scipy.optimize import least_squares

            #initial guess: thru center and radius
            circle = points.mean(axis=0)
            distances = np.linalg.norm(points - circle, axis=1)
            radius = distances.mean()

            #calc residuals from the point:
            def residuals(params, pts):
                cx,cy, r = params
                dist = np.sqrt((pts[:,0] - cx)**2 + (pts[:,1] - cy)**2)
                return dist - r
            
            result = least_squares(
                residuals,
                [circle[0],circle[1],radius],
                args=(points,),
                max_nfev=20
            )

            cx,cy,r=result.x

            #check if the circle is a circle or not
            if not (MIN_RADIUS<r<MAX_RADIUS):
                return None
            
            return Pole(
                cx=cx,
                cy=cy,
                radius=r,
                confidence=1.0,
                last_seen=self.frame_count
            )
        
        except:
            return None
        
    def match_features(self, current_features: List, known_features: List, feature_type: str) -> List[tuple]: #type: ignore
        """match the current features to the known ones so that it can speed up stuff"""
        matches=[]
        if feature_type == "wall":
            for wall_now in current_features:
                
                best_match= None
                best_distance = float("inf") #setting to infinite so we can bottle it down


                for wall_known in known_features:
                    #difference in position and angle of each value to see closest one
                    pos_diff = np.linalg.norm(
                        wall_now.endpoints.mean(axis=0) - wall_known.endpoints.mean(axis=0)
                    )

                    angle_diff = abs(wall_now.angle - wall_known.angle)
                    angle_diff = min(angle_diff,360-angle_diff)

                    #combined distance
                    distance = pos_diff + WEIGHT * angle_diff

                    if distance < best_distance and distance < NEW_DISTANCE_JUMPS_WALL:
                        best_distance = distance
                        best_match=wall_known

                if best_match:
                    matches.append((wall_now,best_match))

        
        elif feature_type == "pole":
            for pole_now in current_features:
                best_match = None
                best_distance = float("inf")


                for pole_known in known_features:
                    #distance between the centers
                    distance = np.sqrt(
                        (pole_now.cx - pole_known.cx)**2 +
                        (pole_now.cy - pole_known.cy)**2
                    )

                    if distance < best_distance and distance < NEW_DISTANCE_JUMPS_POLE:
                        best_distance = distance
                        best_match = pole_known
                
                if best_match:
                    matches.append((pole_now,best_match))
        
        return matches
    
    def estimate_motion(self, walls: List[tuple], poles: List[tuple]): #type: ignore
        """ Estimate robot motion
        
        basically difference between current frame and known features, so its the distance moved from points to points
        """
        dx_estimates=[]
        dy_estimates=[]
        dtheta_estimates=[]

        for current_wall, previous_wall in walls:
            current_center = current_wall.endpoints.mean(axis=0)
            previous_center = previous_wall.endpoints.mean(axis=0)


            # if walls moved, robot in opposite direction
            dx_estimates.append(previous_center[0]-current_center[0])
            dy_estimates.append(previous_center[1]-current_center[1])

            angle_change= previous_wall.angle - current_wall.angle

            #make the anlges between -180 -> 180
            while angle_change >180:
                angle_change-=360
            while angle_change<-180:
                angle_change+=360
            
            dtheta_estimates.append(np.radians(angle_change))

        for current_pole, previous_pole in poles:
            dx_estimates.append(previous_pole.cx - current_pole.cx)
            dy_estimates.append(previous_pole.cy - current_pole.cy)


        #average out the estimates
        if dx_estimates:
            self.estimated_dx = np.median(dx_estimates)
        else:
            self.estimated_dx = 0.0
        
        if dy_estimates:
            self.estimated_dy = np.median(dy_estimates)
        else:
            self.estimated_dy = 0.0
        
        if dtheta_estimates:
            self.estimated_dtheta = np.median(dtheta_estimates)
        else:
            self.estimated_dtheta = 0.0

