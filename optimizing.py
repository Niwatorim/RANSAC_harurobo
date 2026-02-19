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

#TODO: try other algorithms

def find_back_wall_midpoints(angle_ded, t_min):
        """
        find midpoints of all walls
        find highest y value, thats back wall, tell x and y coordinates from that, and return angle

        :param: angle_deg: angle in degrees of each wall
        :param: t_min: 
        """
        pass

def find_back_wall_angle():
        """
        find angle of all walls
        find x and y coordinates from the midpoint of wall closes to 0
        """
        pass



def v2_both_poles_and_walls(df:DataFrame):
    #walls
    MIN_WALL_LENGTH=0.25 #number of points for a valid wall
    DISTANCE_DIFFERENCE= 0.2
    NUMBER_TRIALS_WALLS=500
    RESIDUAL_THRESHOLD_WALLS=0.025 #how much distance from a line can be counted inlier
    MIN_WALL_POINTS=15 #min number of points for a wall to be called a wall
    
    #segmentation
    JUMP_THRESHOLD=0.08 #jump distance between points in the segment
    POINTS_PER_SEGMENT=3 #min number of points for segment

    #poles
    POLE_RMSE=0.1 #How much error allowed for a pole
    NUMBER_TRIALS_POLES=500
    MIN_POLE_POINTS=5 #min number of points counted as a pole
    RESIDUAL_THRESHOLD_POLES=0.02 #how much distance from a line can be counted inlier
    MIN_RADIUS=0.01 #min radius of a pole to be counted
    MAX_RADIUS=0.15 # max radius


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
        Try fit a line, then a circle, see which one works better and then fit the pole regarding it
        
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
            circle_residuals = circle_model.residuals(points)
            circle_rmse = np.sqrt(np.mean(circle_residuals ** 2))
            cx,cy = circle_model.center
            r = circle_model.radius
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

    def find_wall_smol(line_model,points,line_inliers): #unused
        min_inliers=15
        walls=[]
        
        inlier_points=points[line_inliers]
        if len(inlier_points) < min_inliers: # SUS
            return

        point_on_line= line_model.origin
        direction=line_model.direction

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
                walls.append({"wall":wall_endpoints,
                              "angle":angle_deg})
        
        return walls       

    def wall_ransac(data) -> List[dict]:
        """
        Docstring for wall_ransac
        
        :param data: Description
        """
        remaining_data=data.copy() #data to be used
        walls=[] #location of identified walls
        min_inliers= MIN_WALL_POINTS
        while len(remaining_data)>min_inliers:
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
            
            remaining_data = remaining_data[~inliers]
        return walls

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
    
    plt.figure(figsize=(9, 9))
    plt.scatter(x, y, c="green", s=5, alpha=0.5, label="All points")

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
            # print("\n---- inlier points ----")
            # print(pole["inlier_points"])
            # print("------ end ------ \n")
            label = "Pole" if i == 0 else None
            plt.plot(pole["arc_x"], pole["arc_y"], color="red", linewidth=2, label=label)
            plt.scatter(pole["cx"], pole["cy"], c="red", marker='x', s=100, zorder=5)
            plt.scatter(pole["inlier_points"][:, 0], pole["inlier_points"][:, 1],
                        c="yellow", s=20, zorder=4, label="Pole inliers" if i == 0 else None)
            
            # Identify indices of this pole's inliers in the original 'data' array
            # pole["segment_indices"] needs to be stored or recalculated
            if "indices" in pole:
                pole_inlier_mask[pole["indices"]] = True

    remaining_data = data[~pole_inlier_mask]
    wall_segments.extend(wall_ransac(remaining_data))
    # Safe Plotting for Walls
    for i, wall in enumerate(wall_segments):
        label = "Wall" if i == 0 else None
        plt.plot(wall["wall"][:, 0], wall["wall"][:, 1], c="blue", linewidth=2, label=label)
        print(f"ANGLE OF WALL: {i} is {wall['angle']:.2f}")


    print(f"Poles found: {len(all_poles)}")

    #debug
    # print("Actual pole data")
    # pole_data:DataFrame = df[(df["y"]>0.08)&(df["y"]<0.2)]
    # print(pole_data.head(100))


    #plot stuff
    plt.axis("equal")
    plt.xlabel("X (meters)")
    plt.ylabel("Y (meters)")
    plt.title("LiDAR Detection: Poles (Red) & Walls (Blue)")
    plt.legend()
