import json
import numpy as np
import os
import sys
import time
import loguru
import pandas as pd

sys.path.append("/datadrive/codes/retail/delfino")

logger = loguru.logger
logger.add(sys.stdout, level="INFO", format="{time} {level} {message}", filter="my_module")


def calculate_angle(task_id):
    json_path = f"/datadrive/codes/opensource/features/LightGlue/assets/uspg_test_jsons/{task_id}_input.json"
    json_dict = json.load(open(json_path, "r"))
    
    stitch_type = json_dict["type"]
    print("type: ", stitch_type)
    
    Hs = []
    if stitch_type == "VIDEO":
        stitch_info = eval(json_dict["stitchingInfo"])
        # print("stitch_info: ", stitch_info)
        pairs = stitch_info["images"]
        for pair in pairs:
            H = np.array(pair['homographyOfPano'])
            H = H.reshape(3,3)
            Hr = H[0:2,0:2]
            # Hr = Hr/np.linalg.norm(Hr)
            Hr = Hr.reshape(2,2)
            Hs.append(Hr)
    else:
        # parse json
        pairs = json_dict["pair"]
        print("pairs: ", len(pairs))
        
        num = len(pairs)
        for i in range(num):
            H = np.array(pairs[i]['homography'])
            H = H.reshape(3,3)
            Hs.append(H)
        
        Hs_pano = [np.eye(3,3)]
        for H in Hs:
            Hp = np.dot(Hs[-1], np.float32(H).reshape(3, 3))
            Hs_pano.append(Hp)
        
        Hs = Hs_pano.copy()
    
    
    angles = {}
    for i, pair in enumerate(Hs):
        # Perform singular value decomposition  
        U, S, V = np.linalg.svd(pair)  
        # The rotation matrix R is given by the product U*V  
        R = np.dot(U, V)  
        # The rotation angle can be computed from the rotation matrix  
        angle = np.arctan2(R[1, 0], R[0, 0])  
        # Convert to degrees  
        angle = np.degrees(angle)  
        angles[i] = np.abs(angle)
    # print("angles: ", angles)
    mean_angle = np.mean(list(angles.values()))
    max_angle = np.max(list(angles.values()))
    print("mean angle: ", mean_angle)
    print("max angle: ", max_angle)
    
    # differences = [np.linalg.norm(H - mean_pair, 'fro') for H in Hs]
    # print("differences: ", differences)
  
    # worst_homography_index = np.argmax(differences)  
    # worst_homography = Hs[worst_homography_index]
    # print("worst_homography_index: ", worst_homography_index, Hs[worst_homography_index])
    
    # dets = [np.abs(np.linalg.det(H) - 1) for H in Hs]
    # print("dets: ", dets)
    # worst_homography_index = np.argmax(dets)
    # worst_homography = Hs[worst_hdomography_index]
    # print("worst_homography_index: ", worst_homography_index, dets[worst_homography_index])
    return mean_angle, max_angle
    

def process(csv_file):
    df = pd.read_csv(csv_file)
    mean_angles = []
    max_angles = []
    
    for i in range(len(df)):
        task_id = df.iloc[i]["task"]
        mean_angle, max_angle = calculate_angle(task_id)
        mean_angles.append(mean_angle)
        max_angles.append(max_angle)
    
    df["mean_angle"] = mean_angles
    df["max_angle"] = max_angles
    df.to_csv(f"{csv_file[:-4]}_angle.csv", index=False)

    
if __name__ == "__main__":
    csv_path = "/datadrive/codes/opensource/features/LightGlue/assets/uspg_test_jsons/stitching_eval_compare.csv"
    process(csv_path)