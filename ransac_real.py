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

from optimizing import v2_both_poles_and_walls, v2_both_poles_and_walls_no_plot, ScanMatchingTracker, find_back_wall_midpoints, find_poles_midpoint, v3_both_poles_and_walls

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
    os.makedirs(os.path.join(path, "copy"), exist_ok=True)
    basename = os.path.basename(data_path)
    data_path_copy=os.path.join(path,"copy",basename)
    
    with open(data_path, 'r') as f:
        array=f.read()
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



def example_trial():
    tracker = ScanMatchingTracker()
    #FIXME: LOWKEY I THINK WE NEED TO REMOVE ALL THE POINTS AND ONLY SEE WHATS AHEAD CUZ RANGE IS KINDA INSANE RN



    while True:
        scanned_df = read_lidar_scan() #type: ignore

        result = tracker.proces_frame(scanned_df)

        walls = result["walls"]
        poles = result["poles"]
        back_wall = find_back_wall_midpoints(walls)
        pole_midpoint = find_poles_midpoint(poles)
        print(f"back_wall is {back_wall['mid_x']},{back_wall['mid_y']}, angle: {back_wall['angle']}")
        print(f"pole is {pole_midpoint['mid_x']},{pole_midpoint['mid_y']}") #type: ignore
        print(f"Motion estimate:{result['estimated_motion']['dx']},{result['estimated_motion']['dy']}")

if __name__ == "__main__":
    # to_read=place_rings
    # df=pd.read_csv(f"real_points/copy/lidar_data_{to_read}")
    # both_poles_and_walls(df)

    # to_read=honmaru
    # df=pd.read_csv(f"real_points/copy/lidar_data_{to_read}")
    # v2_both_poles_and_walls(df)

    # for file in os.listdir("lidar_data_new"):
    #     if file.endswith(".csv"):
    #         data=os.path.join("lidar_data_new",file)
    #         clean_data(data_path=data, path="lidar_data_new")

    for x in range(1,6):
        path_to_fix = os.path.join("lidar_data_new","copy",f"lidar_data_fix{x}.csv")
        df = pd.read_csv(path_to_fix)
        v3_both_poles_and_walls(df)
    plt.show()

    
    # to_read=take_rings
    # df=pd.read_csv(f"real_points/copy/lidar_data_{to_read}")
    # both_poles_and_walls(df)
    # plt.show()