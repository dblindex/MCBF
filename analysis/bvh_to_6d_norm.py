
#%%
'''
calculate norm
'''
import os
import torch
import numpy as np
import json
import pickle
import rotation_conversions as rc
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
from bvh import Bvh



def bvh_to_6d_tensor(bvh_path, rotation_order="XZY", downsample_step=4):
    root_pos, euler_rot = parse_bvh_file_optimized(bvh_path)
    euler_rot = euler_rot[::downsample_step]
    
    # Rotations (Frames, Joints, 3) -> 6D (Frames, Joints, 6)
    euler_tensor = torch.from_numpy(euler_rot).float()
    euler_radians = torch.deg2rad(euler_tensor)
    matrices = rc.euler_angles_to_matrix(euler_radians, convention=rotation_order)
    rotation_6d = rc.matrix_to_rotation_6d(matrices) 
    
    frames_num = rotation_6d.shape[0]
    rotation_6d_flat = rotation_6d.view(frames_num, -1)
    
    return rotation_6d_flat


def get_file_6d_stats(args):
    bvh_path, order = args
    try:
        rot6d = bvh_to_6d_tensor(bvh_path, order, downsample_step=4)
        combined = rot6d.numpy() # Shape: (Frames, D)
        
        return {
            'n': combined.shape[0],
            'sum': np.sum(combined, axis=0),
            'sum_sq': np.sum(combined**2, axis=0)
        }
    except Exception as e:
        return None


def calculate_6d_speaker_baselines(bvh_dir, output_json='speaker_norm_6D_fps30_baselines.json'):
    
    all_files = []
    for root, _, files in os.walk(bvh_dir):
        for f in files:
            if f.lower().endswith('.bvh'):
                all_files.append(os.path.join(root, f))
    
    files_by_speaker = {}
    for fp in all_files:
        spk_id = os.path.basename(fp).split('_')[1].zfill(3)
        if spk_id not in files_by_speaker:
            files_by_speaker[spk_id] = []
        files_by_speaker[spk_id].append(fp)

    final_stats = {}
    ROT_ORDER = "XZY"
    
    for spk_id, file_list in files_by_speaker.items():
        print(f"Processing Speaker {spk_id} for 6D Norm...")
        
        
        tasks = [(f, ROT_ORDER) for f in file_list]
        with ProcessPoolExecutor(max_workers=10) as executor:
            results = list(tqdm(executor.map(get_file_6d_stats, tasks), total=len(tasks), leave=False))
        
        
        valid_res = [r for r in results if r is not None]
        if not valid_res: continue
        
        total_n = sum(r['n'] for r in valid_res)
        total_sum = sum(r['sum'] for r in valid_res)
        total_sum_sq = sum(r['sum_sq'] for r in valid_res)
        
        mean = total_sum / total_n
        var = (total_sum_sq / total_n) - (mean ** 2)
        std = np.sqrt(np.maximum(var, 1e-8))
        
        final_stats[spk_id] = {
            'mean': mean.tolist(),
            'std': std.tolist()
        }

    with open(output_json, 'w') as f:
        json.dump(final_stats, f)
    print(f"6D Baselines saved to {output_json}")

if __name__ == "__main__":
    BVH_INPUT = '/InterAct_Public/Raw_Body_Motions_BVH'
    calculate_6d_speaker_baselines(BVH_INPUT)