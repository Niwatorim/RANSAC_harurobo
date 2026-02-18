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

from optimizing import v2_both_poles_and_walls

path= "real_points"
honmaru="honmaru.csv"
place_rings="placing_rings.csv"
take_rings="taking_rings.csv"

#fine
honmaru_path=os.path.join(path,f"lidar_data_{honmaru}") #this is when placing rings imo
placing_rings_path=os.path.join(path,f"lidar_data_{place_rings}") #this is when putting a pole
taking_rings_path=os.path.join(path,f"lidar_data_{take_rings}") #yeah checks out, just two walls


def clean_data(data_path,path):
    """
    Cleans and creates copy of data with x and y coordinates 
    
    :param data_path: is the path to ur csv file
    :param path: is the path to root
    """
    data_path_copy=os.path.join(path,"copy",data_path)
    df=pd.read_csv(data_path)
    array=str(df.iat[0,0])
    data=array.split("[")
    final_data=data[1].split("]")[0]
    array_final=final_data.split(",")
    print(array_final[0])

    angle_step=0.25
    start_angle=-90.0

    with open(data_path_copy,"w") as f:
        csv_writer=csv.writer(f)
        csv_writer.writerow(["index","raw","x","y","angle"])


        for i,df_str in enumerate(array_final):

            current_angle= start_angle+ (float(i) * angle_step)
            current_angle_rad=math.radians(current_angle)


            x = float(df_str) * math.cos(current_angle_rad) #type: ignore
            y= float(df_str) * math.sin(current_angle_rad) #type: ignore
            csv_writer.writerow([i,df_str,x,y,current_angle])

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
            line_residuals = line_model.residuals(points)

            #find the error
            line_rmse=np.sqrt(np.mean(line_residuals ** 2))
        except Exception:
            line_rmse = float("inf")
        
        #try fit a circle
        try:
            circle_model,circle_inliers = ransac(
                points,CircleModel,min_samples=3,residual_threshold=0.02,max_trials=NUMBER_TRIALS_POLES
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
        plt.plot(wall["wall"][:, 0], wall["wall"][:, 1], c="blue", linewidth=2, label=label)
        
        #find the angle
        print(f"ANGLE OF WALL: {i} is {wall["angle"]}")


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

if __name__ == "__main__":
    data_path=taking_rings_path
    # to_read=place_rings
    # df=pd.read_csv(f"real_points/copy/lidar_data_{to_read}")
    # both_poles_and_walls(df)

    to_read=take_rings
    df=pd.read_csv(f"real_points/copy/lidar_data_{to_read}")
    v2_both_poles_and_walls(df)
    
    # to_read=take_rings
    # df=pd.read_csv(f"real_points/copy/lidar_data_{to_read}")
    # both_poles_and_walls(df)
    plt.show()