import os
import subprocess
import pandas as pd
import time
import re


python_script = "training_dynamic_window.py" 
ref_bvh = "InterAct_Public/Raw_Body_Motions_BVH/20231119_001_052.bvh"
gpu_id = "0"
base_dir = 'InterAct_pause_analysis'


manifests = [
    f"{base_dir}/manifest_window_0.5s_onlyangular_v.csv",
    f"{base_dir}/manifest_window_1.0s_onlyangular_v.csv",
    f"{base_dir}/manifest_window_1.5s_onlyangular_v.csv",
    f"{base_dir}/manifest_window_2.0s_onlyangular_v.csv",
    f"{base_dir}/manifest_window_2.5s_onlyangular_v.csv",
    f"{base_dir}/manifest_window_3.0s_onlyangular_v.csv"
]


def run_all():
    all_summaries = []
    
    for manifest in manifests:
        print(f"\n{'#'*40}")
        print(f"Running Experiment for: {os.path.basename(manifest)}")
        print(f"{'#'*40}")
        
        cmd = [
            "python", python_script,
            "--gpu", gpu_id,
            "--ref_bvh", ref_bvh,
            "--manifest", manifest
        ]
        
        subprocess.run(cmd, check=True)
        
        summary_name = f"results_summary_{os.path.basename(manifest).replace('.csv', '')}.csv"
        base_dir = '/homes/ying_hsusn/PantoMatrix_new/InterAct_pause_analysis'
        summary_path = os.path.join(base_dir, summary_name)
        
        if os.path.exists(summary_path):
            df = pd.read_csv(summary_path)
            
            match = re.search(r'window_([0-9\.]+)s', manifest)
            window_label = f"{match.group(1)}s" if match else "Unknown"
            
            df['Window'] = window_label
            all_summaries.append(df)
            
    if all_summaries:
        final_df = pd.concat(all_summaries, ignore_index=True)
        
        cols = ['Window', 'BodyPart'] + [c for c in final_df.columns if c not in ['Window', 'BodyPart']]
        final_df = final_df[cols]
        
        output_csv_file = os.path.join(base_dir, 'part3_full_results_onlyangular.csv')
        final_df.to_csv(output_csv_file, index=False)
        print(f"\n All experiments done! Saved to {output_csv_file}")
        
        print(final_df.to_string())

if __name__ == "__main__":
    run_all()