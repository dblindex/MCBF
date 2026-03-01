# # import os
# # import numpy as np
# # import pandas as pd
# # import json
# # import torch
# # import concurrent.futures
# # from functools import partial
# # import matplotlib.pyplot as plt
# # import seaborn as sns

# # # ==========================================
# # # 1. BVH 讀取 (30fps)
# # # ==========================================
# # class SimpleBVHReader:
# #     def __init__(self, file_path):
# #         self.file_path = file_path
# #         self.frame_time = 0.033333  # 預設 30fps
# #         self.motion_data = None 
# #         self.downsample_step = 4    
# #         self._parse()

# #     def _parse(self):
# #         try:
# #             with open(self.file_path, 'r') as f:
# #                 lines = f.readlines()
            
# #             motion_start_idx = 0
# #             original_frame_time = 0.008333

# #             for i, line in enumerate(lines):
# #                 if "Frame Time:" in line:
# #                     original_frame_time = float(line.split(":")[1].strip())
# #                     self.frame_time = original_frame_time * self.downsample_step
                
# #                 if "MOTION" in line:
# #                     motion_start_idx = i + 3 
# #                     break
            
# #             raw_data = []
# #             data_lines = lines[motion_start_idx:][::self.downsample_step]

# #             for line in data_lines:
# #                 try:
# #                     vals = [float(x) for x in line.strip().split()]
# #                     if len(vals) > 0: 
# #                         raw_data.append(vals)
# #                 except ValueError: 
# #                     continue
            
# #             if len(raw_data) > 0:
# #                 self.motion_data = np.array(raw_data)
                
# #         except Exception as e:
# #             # print(f"BVH Parse Error: {e}") 
# #             self.motion_data = None

# #     def get_slice(self, start_sec, end_sec):
# #         if self.motion_data is None: return None
        
# #         start_frame = int(start_sec / self.frame_time)
# #         end_frame = int(end_sec / self.frame_time)
        
# #         max_frame = self.motion_data.shape[0]
# #         start_frame = max(0, start_frame)
# #         end_frame = min(max_frame, end_frame)
        
# #         if start_frame >= end_frame: return None
        
# #         return self.motion_data[start_frame:end_frame, :]

# # def get_normalized_score(raw_value, spk_id, metric_type, baseline_dict):
# #     spk_id_str = str(int(spk_id)).zfill(3)
# #     if spk_id_str not in baseline_dict:
# #         return None
# #     stats = baseline_dict[spk_id_str]
# #     mu = stats[f'{metric_type}_mean']
# #     sigma = stats[f'{metric_type}_std']
    
# #     if sigma == 0: return 0
# #     return (raw_value - mu) / sigma

# # # ==========================================
# # # 2. 處理單一列資料 (Worker Function)
# # # ==========================================
# # def process_single_row(row_data, bvh_root_dir, baseline_dict, interp_samples=100):
# #     subject_str = str(int(row_data['subject_id'])).zfill(3)
# #     scenario_str = str(int(row_data['scenario'])).zfill(3)
# #     date_str = str(row_data['date'])
    
# #     bvh_filename = f"{date_str}_{subject_str}_{scenario_str}.bvh"
# #     bvh_path = os.path.join(bvh_root_dir, bvh_filename)

# #     if not os.path.exists(bvh_path):
# #         return None

# #     # --- 時間區間邏輯 [t-3.5, t-0.5] ---
# #     # CSV 中的 start_time 就是我們的 Anchor t
# #     anchor_time = row_data['start_time']
# #     offset_seconds = 0.5
# #     window_duration = 3.0
    
# #     window_end_time = anchor_time - offset_seconds
# #     window_start_time = window_end_time - window_duration

# #     if window_start_time < 0:
# #         return None

# #     try:
# #         bvh = SimpleBVHReader(bvh_path)
# #         motion_clip = bvh.get_slice(window_start_time, window_end_time)
# #     except Exception as e:
# #         return None 
    
# #     if motion_clip is None or motion_clip.shape[0] < 2:
# #         return None

# #     # 計算 Velocity
# #     frame_diff = np.diff(motion_clip, axis=0)
# #     velocity = np.linalg.norm(frame_diff, axis=1) # / 4.0 
    

# #     if len(velocity) < 2: return None
# #     x_old = np.linspace(0, 1, len(velocity))
# #     x_new = np.linspace(0, 1, interp_samples)
# #     vel_interp = np.interp(x_new, x_old, velocity)
    
# #     norm_vel_array = get_normalized_score(
# #         vel_interp, 
# #         subject_str, 
# #         'vel', 
# #         baseline_dict
# #     )
# #     if norm_vel_array is None: return None 

# #     # 計算 Diversity
# #     diversity_score = np.mean(np.std(motion_clip, axis=0))
# #     norm_div_scalar = get_normalized_score(
# #         diversity_score, 
# #         subject_str, 
# #         'div', 
# #         baseline_dict
# #     )
# #     if norm_div_scalar is None: return None

# #     # 定義 X 軸為 -3.5 ~ -0.5
# #     time_axis = np.linspace(-3.5, -0.5, interp_samples)
    
# #     vel_results = []
# #     for t, v in zip(time_axis, norm_vel_array):
# #         vel_results.append({
# #             'Time (s)': t,
# #             'Velocity': v,
# #             'Case': row_data['case'],
# #             'Subject': subject_str
# #         })
        
# #     div_result = {
# #         'Diversity': norm_div_scalar,
# #         'Case': row_data['case'],
# #         'Subject': subject_str
# #     }
    
# #     return vel_results, div_result

# # # ==========================================
# # # 3. 繪圖與統計表函數
# # # ==========================================
# # sns.set_theme(style="whitegrid")

# # def plot_focus_4(df, title, file_name, x_lim=(-3.5, -0.5)):
# #     if df.empty:
# #         print("Warning: 沒有資料。")
# #         return

# #     plt.figure(figsize=(8, 5))
    
# #     sns.lineplot(
# #         data=df,
# #         x='Time (s)',
# #         y='Velocity',
# #         hue='Case', 
# #         hue_order=['A', 'B', 'C', 'D'], # 固定順序避免顏色亂跳
# #         palette='tab10',  
# #         linewidth=2.5,
# #         errorbar=None
# #     )
    
# #     plt.title(f"{title}\nWindow: [t-3.5, t-0.5]", fontsize=14, fontweight='bold')
# #     plt.ylabel("Normalized Motion Velocity (Z-Score)", fontsize=12)
# #     plt.xlabel("Time relative to Pause End t (s)", fontsize=12)
    
# #     plt.axhline(0, color='gray', linestyle='--', alpha=0.5, label='Personal Baseline')
    
# #     if x_lim:
# #         plt.xlim(x_lim)
        
# #     plt.legend(
# #     title='Case ', 
# #     bbox_to_anchor=(1.01, 1), # 1.01 代表 X 軸的框外一點點，1 代表 Y 軸的最頂部
# #     loc='upper left',         # 以圖例的左上角來對齊剛剛設定的座標點
# #     borderaxespad=0.
# #     )

# #     plt.tight_layout()
# #     plt.savefig(f"{file_name}_offset_0.5.png")
# #     plt.show()

# # def print_summary_tables(vel_df, div_df):
# #     print("\n" + "="*50)
# #     print("【Velocity Summary Table (Z-Score)】 Window: [t-3.5, t-0.5]")
# #     print("="*50)
    
# #     vel_summary = vel_df.groupby('Case')['Velocity'].mean().round(3).reset_index()
# #     print(vel_summary.to_string(index=False))

# #     print("\n" + "="*50)
# #     print("【Diversity Summary Table (Z-Score)】 Window: [t-3.5, t-0.5]")
# #     print("="*50)
    
# #     div_summary = div_df.groupby('Case')['Diversity'].mean().round(3).reset_index()
# #     print(div_summary.to_string(index=False))

# # # ==========================================
# # # 主程式
# # # ==========================================
# # if __name__ == '__main__':
# #     with open('speaker_norm_baselines.json', 'r') as f:
# #         baseline_dict = json.load(f)
        
# #     BVH_ROOT_DIR = '/homes/ying_hsusn/PantoMatrix_new/InterAct_Public/Raw_Body_Motions_BVH'
# #     CSV_FILE = '/homes/ying_hsusn/PantoMatrix_new/InterAct_pause_analysis/manifest_window_3.0s.csv'
    
# #     df = pd.read_csv(CSV_FILE)
# #     print(f"現在分析: {CSV_FILE}")
# #     print(f"總筆數: {len(df)}")
    
# #     rows_to_process = df.to_dict('records')
    
# #     all_vel_records = []
# #     all_div_records = []
    
# #     MAX_WORKERS = 11
# #     print(f"開始處理資料，使用 {MAX_WORKERS} 個核心進行運算...")
    
# #     func = partial(process_single_row, bvh_root_dir=BVH_ROOT_DIR, baseline_dict=baseline_dict)
    
# #     with concurrent.futures.ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:    
# #         results = list(executor.map(func, rows_to_process))
        
# #     for res in results:
# #         if res is not None:
# #             vel_list, div_item = res
# #             all_vel_records.extend(vel_list)
# #             all_div_records.append(div_item)
            
# #     print(f"處理完成。共收集 {len(all_vel_records)} 筆 Velocity 點, {len(all_div_records)} 筆 Diversity 資料。")        
    
# #     vel_df = pd.DataFrame(all_vel_records)
# #     div_df = pd.DataFrame(all_div_records)

# #     # 執行繪圖與產出表格
# #     if not vel_df.empty:
# #         plot_focus_4(
# #             vel_df, 
# #             title="Interlocutor's Reaction", 
# #             file_name='Focus4_reaction'
# #         )
# #         print_summary_tables(vel_df, div_df)
# #     else:
# #         print("沒有成功解析的資料，請檢查路徑與 JSON 是否匹配。")

# import os
# import glob
# import numpy as np
# import pandas as pd
# import json
# import concurrent.futures
# from functools import partial
# import matplotlib.pyplot as plt
# import seaborn as sns

# # ==========================================
# # 設定與常數
# # ==========================================
# TARGET_BODY_PART = 'UpperBody'  # 隨時可改為 'FullBody', 'Hands', 'Head' 來跑不同實驗

# PARTS_DEF = {
#     'FullBody': [''], 
#     'Hands': ['Hand', 'Wrist'],
#     'Head': ['Head', 'Neck'],
#     'UpperBody': ['Spine', 'Neck', 'Head', 'Shoulder', 'Arm', 'Hand']
# }

# # 🌟【新增】將 TARGET_BODY_PART 映射到 JSON 的 Key prefix
# JSON_PREFIX_MAP = {
#     'FullBody': 'fullbody',
#     'Hands': 'hands',
#     'Head': 'head',
#     'UpperBody': 'upper_body'
# }

# # ==========================================
# # 1. 身體部位索引器
# # ==========================================
# class BodyPartIndexer:
#     def __init__(self, ref_bvh_path):
#         self.joint_names = []
#         self.channel_map = []
#         self._parse_header(ref_bvh_path)

#     def _parse_header(self, file_path):
#         current_channel_idx = 0
#         current_joint = None
#         try:
#             with open(file_path, 'r') as f:
#                 for line in f:
#                     line = line.strip()
#                     if line.startswith("MOTION"): break
#                     if line.startswith("End Site"):
#                         current_joint = None 
#                         continue
                        
#                     if line.startswith("ROOT") or line.startswith("JOINT"):
#                         current_joint = line.split()[1]
#                         self.joint_names.append(current_joint)
                        
#                     elif line.startswith("CHANNELS"):
#                         if current_joint is None: continue
#                         parts = line.split()
#                         num_channels = int(parts[1])
#                         indices = list(range(current_channel_idx, current_channel_idx + num_channels))
#                         self.channel_map.append({
#                             "name": current_joint,
#                             "indices": indices
#                         })
#                         current_channel_idx += num_channels
#         except Exception as e:
#             print(f"Error parsing reference BVH: {e}")

#     def get_indices(self, part_name):
#         keywords = PARTS_DEF.get(part_name, [''])
#         target_indices = []
#         for item in self.channel_map:
#             if keywords == ['']:
#                 target_indices.extend(item['indices'])
#             elif any(k.lower() in item['name'].lower() for k in keywords):
#                 target_indices.extend(item['indices'])
#         target_indices.sort()
#         return target_indices

# # ==========================================
# # 2. BVH 讀取器
# # ==========================================
# class SimpleBVHReader:
#     def __init__(self, file_path):
#         self.file_path = file_path
#         self.frame_time = 0.033333
#         self.motion_data = None 
#         self.downsample_step = 4    
#         self._parse()

#     def _parse(self):
#         try:
#             with open(self.file_path, 'r') as f:
#                 lines = f.readlines()
            
#             motion_start_idx = 0
#             original_frame_time = 0.008333

#             for i, line in enumerate(lines):
#                 if "Frame Time:" in line:
#                     original_frame_time = float(line.split(":")[1].strip())
#                     self.frame_time = original_frame_time * self.downsample_step
                
#                 if "MOTION" in line:
#                     motion_start_idx = i + 3 
#                     break
            
#             raw_data = []
#             data_lines = lines[motion_start_idx:][::self.downsample_step]

#             for line in data_lines:
#                 try:
#                     vals = [float(x) for x in line.strip().split()]
#                     if len(vals) > 0: raw_data.append(vals)
#                 except ValueError: continue
            
#             if len(raw_data) > 0:
#                 self.motion_data = np.array(raw_data)
#         except Exception as e:
#             self.motion_data = None

#     def get_slice(self, start_sec, end_sec):
#         if self.motion_data is None: return None
#         start_frame = int(start_sec / self.frame_time)
#         end_frame = int(end_sec / self.frame_time)
#         max_frame = self.motion_data.shape[0]
#         start_frame = max(0, start_frame)
#         end_frame = min(max_frame, end_frame)
        
#         if start_frame >= end_frame: return None
#         return self.motion_data[start_frame:end_frame, :]

# # ==========================================
# # 3. Z-Score 正規化函式 (🌟已更新讀取邏輯🌟)
# # ==========================================
# def get_normalized_score(raw_value, spk_id, metric_type, baseline_dict, target_part_name):
#     spk_id_str = str(int(spk_id)).zfill(3)
#     if spk_id_str not in baseline_dict:
#         return None
        
#     stats = baseline_dict[spk_id_str]
    
#     # 將 'UpperBody' 轉為 'upper_body'
#     prefix = JSON_PREFIX_MAP.get(target_part_name, 'fullbody')
    
#     # 組合出正確的 JSON Key (例: upper_body_vel_mean)
#     mean_key = f"{prefix}_{metric_type}_mean"
#     std_key = f"{prefix}_{metric_type}_std"
    
#     if mean_key not in stats or std_key not in stats:
#         print(f"Warning: JSON missing keys {mean_key} or {std_key} for speaker {spk_id_str}")
#         return None
        
#     mu = stats[mean_key]
#     sigma = stats[std_key]
    
#     if sigma == 0: return 0
#     return (raw_value - mu) / sigma

# # ==========================================
# # 4. 單筆資料處理 (Worker)
# # ==========================================
# def process_single_row(row_data, bvh_root_dir, baseline_dict, target_indices, target_part_name, interp_samples=100):
#     subject_str = str(int(row_data['subject_id'])).zfill(3)
#     scenario_str = str(int(row_data['scenario'])).zfill(3)
#     date_str = str(row_data['date'])
    
#     bvh_filename = f"{date_str}_{subject_str}_{scenario_str}.bvh"
#     bvh_path = os.path.join(bvh_root_dir, bvh_filename)

#     if not os.path.exists(bvh_path): return None

#     anchor_time = row_data['start_time']
#     offset_seconds = 0.5
#     window_duration = 3.0
#     window_end_time = anchor_time - offset_seconds
#     window_start_time = window_end_time - window_duration

#     if window_start_time < 0: return None

#     try:
#         bvh = SimpleBVHReader(bvh_path)
#         motion_clip = bvh.get_slice(window_start_time, window_end_time)
#     except Exception as e:
#         return None 
    
#     if motion_clip is None or motion_clip.shape[0] < 2: return None

#     # 只切出目標部位的 Channels
#     motion_clip = motion_clip[:, target_indices]

#     # 計算 Velocity
#     # frame_diff = np.diff(motion_clip, axis=0)
#     # velocity = np.linalg.norm(frame_diff, axis=1) 
    
    

#     frame_diff = np.diff(motion_clip, axis=0)
#     # 強制將角度差值收斂到 [-180, 180] 之間
#     frame_diff = (frame_diff + 180) % 360 - 180     
#     velocity = np.linalg.norm(frame_diff, axis=1)
#     # 2. 除以 frame_time 轉為「秒速」
#     dt = bvh.frame_time 
#     velocity = np.linalg.norm(frame_diff, axis=1) / dt 

#     if len(velocity) < 2: return None
#     x_old = np.linspace(0, 1, len(velocity))
#     x_new = np.linspace(0, 1, interp_samples)
#     vel_interp = np.interp(x_new, x_old, velocity)
    
#     # 傳入 target_part_name 以精準讀取 JSON 對應數值
#     norm_vel_array = get_normalized_score(vel_interp, subject_str, 'vel', baseline_dict, target_part_name)
#     if norm_vel_array is None: return None 

#     # 計算 Diversity
#     diversity_score = np.mean(np.std(motion_clip, axis=0))
#     norm_div_scalar = get_normalized_score(diversity_score, subject_str, 'div', baseline_dict, target_part_name)
#     if norm_div_scalar is None: return None

#     time_axis = np.linspace(-3.5, -0.5, interp_samples)
    
#     vel_results = []
#     for t, v in zip(time_axis, norm_vel_array):
#         vel_results.append({
#             'Time (s)': t,
#             'Velocity': v,
#             'Case': row_data['case'],
#             'Subject': subject_str
#         })
        
#     div_result = {
#         'Diversity': norm_div_scalar,
#         'Case': row_data['case'],
#         'Subject': subject_str
#     }
    
#     return vel_results, div_result

# # ==========================================
# # 5. 繪圖與統計表函數
# # ==========================================
# sns.set_theme(style="whitegrid")

# def plot_focus_4(df, title, file_name, x_lim=(-3.5, -0.5)):
#     if df.empty:
#         print("Warning: 沒有資料。")
#         return

#     plt.figure(figsize=(8, 5))
    
#     sns.lineplot(
#         data=df,
#         x='Time (s)',
#         y='Velocity',
#         hue='Case', 
#         hue_order=['A', 'B', 'C', 'D'], 
#         palette='tab10',  
#         linewidth=2.5,
#         errorbar=None
#     )
    
#     plt.title(f"{title}\nWindow: [t-3.5, t-0.5]", fontsize=14, fontweight='bold')
#     plt.ylabel("Normalized Motion Velocity (Z-Score)", fontsize=12)
#     plt.xlabel("Time relative to Pause End t (s)", fontsize=12)
    
#     plt.axhline(0, color='gray', linestyle='--', alpha=0.5, label='Personal Baseline')
    
#     if x_lim: plt.xlim(x_lim)
        
#     plt.legend(title='Case', bbox_to_anchor=(1.01, 1), loc='upper left', borderaxespad=0.)
#     plt.tight_layout()
#     plt.savefig(f"{file_name}_offset_0.5.png")
#     # plt.show()

# def print_summary_tables(vel_df, div_df, part_name):
#     print("\n" + "="*50)
#     print(f"【Velocity Summary Table (Z-Score) - {part_name}】 Window: [t-3.5, t-0.5]")
#     print("="*50)
#     vel_summary = vel_df.groupby('Case')['Velocity'].mean().round(3).reset_index()
#     print(vel_summary.to_string(index=False))

#     print("\n" + "="*50)
#     print(f"【Diversity Summary Table (Z-Score) - {part_name}】 Window: [t-3.5, t-0.5]")
#     print("="*50)
#     div_summary = div_df.groupby('Case')['Diversity'].mean().round(3).reset_index()
#     print(div_summary.to_string(index=False))

# # ==========================================
# # 6. 主程式
# # ==========================================
# if __name__ == '__main__':
#     BVH_ROOT_DIR = '/homes/ying_hsusn/PantoMatrix_new/InterAct_Public/Raw_Body_Motions_BVH'
#     CSV_FILE = '/homes/ying_hsusn/PantoMatrix_new/InterAct_pause_analysis/manifest_window_3.0s.csv'
#     BASELINE_JSON = 'speaker_norm_baselines_parts.json' 
    
#     with open(BASELINE_JSON, 'r') as f:
#         baseline_dict = json.load(f)

#     sample_bvh_list = glob.glob(os.path.join(BVH_ROOT_DIR, "*.bvh"))
#     if not sample_bvh_list:
#         raise FileNotFoundError(f"找不到任何 BVH 檔案於 {BVH_ROOT_DIR}")
    
#     ref_bvh_path = sample_bvh_list[0]
#     indexer = BodyPartIndexer(ref_bvh_path)
#     target_indices = indexer.get_indices(TARGET_BODY_PART)
#     print(f"[{TARGET_BODY_PART}] 成功萃取出 {len(target_indices)} 個 Channels 作為特徵！")

#     df = pd.read_csv(CSV_FILE)
#     print(f"現在分析: {CSV_FILE} | 總筆數: {len(df)}")
    
#     rows_to_process = df.to_dict('records')
#     all_vel_records = []
#     all_div_records = []
    
#     MAX_WORKERS = 11
#     print(f"開始處理資料，使用 {MAX_WORKERS} 個核心進行運算...")
    
#     # 傳入 target_part_name
#     func = partial(process_single_row, bvh_root_dir=BVH_ROOT_DIR, baseline_dict=baseline_dict, target_indices=target_indices, target_part_name=TARGET_BODY_PART)
    
#     with concurrent.futures.ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:   
#         results = list(executor.map(func, rows_to_process))
        
#     for res in results:
#         if res is not None:
#             vel_list, div_item = res
#             all_vel_records.extend(vel_list)
#             all_div_records.append(div_item)
            
#     print(f"處理完成。共收集 {len(all_vel_records)} 筆 Velocity 點, {len(all_div_records)} 筆 Diversity 資料。")        
    
#     vel_df = pd.DataFrame(all_vel_records)
#     div_df = pd.DataFrame(all_div_records)

#     if not vel_df.empty:
#         plot_focus_4(
#             vel_df, 
#             title=f"Interlocutor's Reaction ({TARGET_BODY_PART})", 
#             file_name=f'Focus4_reaction_{TARGET_BODY_PART}'
#         )
#         print_summary_tables(vel_df, div_df, TARGET_BODY_PART)
#     else:
#         print("沒有成功解析的資料，請檢查路徑與 JSON 是否匹配。")

import os
import glob
import numpy as np
import pandas as pd
import json
import concurrent.futures
from functools import partial

# ==========================================
# 設定與常數
# ==========================================
PARTS_DEF = {
    'FullBody': [''], 
    'Hands': ['Hand', 'Wrist'],
    'Head': ['Head', 'Neck'],
    'UpperBody': ['Spine', 'Neck', 'Head', 'Shoulder', 'Arm', 'Hand']
}

JSON_PREFIX_MAP = {
    'FullBody': 'fullbody',
    'Hands': 'hands',
    'Head': 'head',
    'UpperBody': 'upper_body'
}

# ==========================================
# 1. 身體部位索引器 (修復：嚴格過濾 Rotation)
# ==========================================
class BodyPartIndexer:
    def __init__(self, ref_bvh_path):
        self.joint_names = []
        self.channel_map = []
        self._parse_header(ref_bvh_path)

    def _parse_header(self, file_path):
        current_channel_idx = 0
        current_joint = None
        try:
            with open(file_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line.startswith("MOTION"): break
                    if line.startswith("End Site"):
                        current_joint = None 
                        continue
                        
                    if line.startswith("ROOT") or line.startswith("JOINT"):
                        current_joint = line.split()[1]
                        self.joint_names.append(current_joint)
                        
                    elif line.startswith("CHANNELS"):
                        if current_joint is None: continue
                        parts = line.split()
                        num_channels = int(parts[1])
                        channel_types = parts[2:]
                        
                        # ✨ 修復 1: 只紀錄 rotation 的 Index，捨棄 X/Y/Z position ✨
                        rot_indices = []
                        for j, c_type in enumerate(channel_types):
                            if 'rotation' in c_type.lower():
                                rot_indices.append(current_channel_idx + j)
                        
                        self.channel_map.append({
                            "name": current_joint,
                            "rot_indices": rot_indices
                        })
                        current_channel_idx += num_channels
        except Exception as e:
            print(f"Error parsing reference BVH: {e}")

    # ✨ 修復 2: 保持 List of Lists 結構，讓各個關節的 x,y,z 獨立計算 norm ✨
    def get_joint_rotation_indices(self, part_name):
        keywords = PARTS_DEF.get(part_name, [''])
        target_joint_indices = []
        for item in self.channel_map:
            if keywords == [''] or any(k.lower() in item['name'].lower() for k in keywords):
                if len(item['rot_indices']) > 0:
                    target_joint_indices.append(item['rot_indices'])
        return target_joint_indices

# ==========================================
# 2. BVH 讀取器 (無更動)
# ==========================================
class SimpleBVHReader:
    def __init__(self, file_path):
        self.file_path = file_path
        self.frame_time = 0.033333
        self.motion_data = None 
        self.downsample_step = 4    
        self._parse()

    def _parse(self):
        try:
            with open(self.file_path, 'r') as f:
                lines = f.readlines()
            motion_start_idx = 0
            original_frame_time = 0.008333

            for i, line in enumerate(lines):
                if "Frame Time:" in line:
                    original_frame_time = float(line.split(":")[1].strip())
                    self.frame_time = original_frame_time * self.downsample_step
                if "MOTION" in line:
                    motion_start_idx = i + 3 
                    break
            
            raw_data = []
            data_lines = lines[motion_start_idx:][::self.downsample_step]
            for line in data_lines:
                try:
                    vals = [float(x) for x in line.strip().split()]
                    if len(vals) > 0: raw_data.append(vals)
                except ValueError: continue
            
            if len(raw_data) > 0:
                self.motion_data = np.array(raw_data)
        except Exception as e:
            self.motion_data = None

    def get_slice(self, start_sec, end_sec):
        if self.motion_data is None: return None
        start_frame = int(start_sec / self.frame_time)
        end_frame = int(end_sec / self.frame_time)
        max_frame = self.motion_data.shape[0]
        start_frame = max(0, start_frame)
        end_frame = min(max_frame, end_frame)
        if start_frame >= end_frame: return None
        return self.motion_data[start_frame:end_frame, :]

# ==========================================
# 3. Z-Score 正規化函式 (無更動)
# ==========================================
def get_normalized_score(raw_value, spk_id, metric_type, baseline_dict, target_part_name):
    spk_id_str = str(int(spk_id)).zfill(3)
    if spk_id_str not in baseline_dict: return None
    speaker_stats = baseline_dict[spk_id_str]
    prefix = JSON_PREFIX_MAP.get(target_part_name, 'fullbody')
    if prefix not in speaker_stats: return None
        
    part_stats = speaker_stats[prefix]
    mean_key = f"{metric_type}_mean"
    std_key = f"{metric_type}_std"
    if mean_key not in part_stats or std_key not in part_stats: return None
        
    mu = part_stats[mean_key]
    sigma = part_stats[std_key]
    if sigma == 0: return 0
    return (raw_value - mu) / sigma

# ==========================================
# 4. 單筆資料處理 (修復：完全對齊 Baseline 邏輯)
# ==========================================
def process_single_row(row_data, bvh_root_dir, baseline_dict, target_joint_indices, target_part_name, tau):
    subject_str = str(int(row_data['subject_id'])).zfill(3)
    scenario_str = str(int(row_data['scenario'])).zfill(3)
    date_str = str(row_data['date'])
    
    bvh_filename = f"{date_str}_{subject_str}_{scenario_str}.bvh"
    bvh_path = os.path.join(bvh_root_dir, bvh_filename)

    if not os.path.exists(bvh_path): return None

    anchor_time = row_data['start_time']
    offset_seconds = 0.5
    window_duration = 3.0
    window_end_time = anchor_time - offset_seconds
    window_start_time = window_end_time - window_duration

    if window_start_time < 0: return None

    try:
        bvh = SimpleBVHReader(bvh_path)
        motion_clip = bvh.get_slice(window_start_time, window_end_time)
    except Exception as e:
        return None 
    
    if motion_clip is None or motion_clip.shape[0] < 2: return None

    dt = bvh.frame_time 
    WINDOW_SEC = 0.5 
    window_frames = max(2, int(WINDOW_SEC / dt))

    v_i_t_list = []
    d_i_t_list = []

    # ✨ 修復 3: 完全還原 Baseline 的計算數學式 (Deg2Rad, Unwrap, Rolling STD) ✨
    for indices in target_joint_indices:
        # 取出該關節的 3 軸旋轉
        joint_rot = motion_clip[:, indices]
        
        # 1. 轉弧度並 Unwrap (這是最關鍵的差異！)
        joint_rot_rad = np.deg2rad(joint_rot)
        joint_rot_unwrapped = np.unwrap(joint_rot_rad, axis=0)
        
        # 2. 獨立計算每個關節的 Velocity
        diff = np.diff(joint_rot_unwrapped, axis=0)
        v_i = np.linalg.norm(diff, axis=1) / dt
        v_i_t_list.append(v_i)
        
        # 3. 獨立計算每個關節的 0.5秒 Rolling Diversity
        df_joint = pd.DataFrame(joint_rot_unwrapped)
        rolling_std = df_joint.rolling(window=window_frames, center=True, min_periods=window_frames).std()
        d_i = rolling_std.mean(axis=1).dropna().values
        d_i_t_list.append(d_i)

    if not v_i_t_list or not d_i_t_list:
        return None

    # 平均所有關節，算出整段序列的時間折線圖
    V_t = np.mean(v_i_t_list, axis=0)
    D_t = np.mean(d_i_t_list, axis=0)

    # 取此 3 秒窗口的平均值
    mean_raw_vel = float(np.mean(V_t))
    mean_raw_div = float(np.mean(D_t))

    # 取 Z-Score 正規化
    norm_vel_scalar = get_normalized_score(mean_raw_vel, subject_str, 'vel', baseline_dict, target_part_name)
    norm_div_scalar = get_normalized_score(mean_raw_div, subject_str, 'div', baseline_dict, target_part_name)

    if norm_vel_scalar is None or norm_div_scalar is None: 
        return None 

    result = {
        'Tau': tau,
        'BodyPart': target_part_name,
        'Case': row_data['case'],
        'Subject': subject_str,
        'Velocity': norm_vel_scalar,
        'Diversity': norm_div_scalar
    }
    
    return result

# ==========================================
# 5. 主程式
# ==========================================
if __name__ == '__main__':
    BVH_ROOT_DIR = '/homes/ying_hsusn/PantoMatrix_new/InterAct_Public/Raw_Body_Motions_BVH'
    BASELINE_JSON = 'speaker_norm_speakerwise_baselines_parts_vel_div.json' 
    
    if not os.path.exists(BASELINE_JSON):
        raise FileNotFoundError(f"找不到 JSON 檔案: {BASELINE_JSON}")
    with open(BASELINE_JSON, 'r') as f:
        baseline_dict = json.load(f)

    sample_bvh_list = glob.glob(os.path.join(BVH_ROOT_DIR, "*.bvh"))
    if not sample_bvh_list:
        raise FileNotFoundError(f"找不到任何 BVH 檔案於 {BVH_ROOT_DIR}")
    
    ref_bvh_path = sample_bvh_list[0]
    indexer = BodyPartIndexer(ref_bvh_path)

    MAX_WORKERS = 11
    taus = [0.0, 0.5, 1.0, 1.5]
    body_parts = ['FullBody', 'Hands', 'Head', 'UpperBody']

    global_records = []

    for tau in taus:
        CSV_FILE = f'/homes/ying_hsusn/PantoMatrix_new/InterAct_pause_analysis/manifest_shifted_tau_{tau}.csv'
        
        if not os.path.exists(CSV_FILE):
            print(f"找不到 CSV 檔案: {CSV_FILE}，跳過此 Tau...")
            continue
            
        df = pd.read_csv(CSV_FILE)
        rows_to_process = df.to_dict('records')
        print(f"\n" + "#"*70)
        print(f"開始分析 Tau={tau} | 總筆數: {len(df)}")
        print("#"*70)

        for part in body_parts:
            # ✨ 改呼叫新的函式，取回分好組的關節 indices ✨
            target_joint_indices = indexer.get_joint_rotation_indices(part)
            total_channels = sum(len(idx) for idx in target_joint_indices)
            print(f">>> 正在處理部位: [{part}] (萃取出 {total_channels} 個 Channels 進行運算)")
            
            func = partial(
                process_single_row, 
                bvh_root_dir=BVH_ROOT_DIR, 
                baseline_dict=baseline_dict, 
                target_joint_indices=target_joint_indices, 
                target_part_name=part,
                tau=tau 
            )
            
            with concurrent.futures.ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:   
                results = list(executor.map(func, rows_to_process))
                
            for res in results:
                if res is not None:
                    global_records.append(res)
                    
            print(f"    [{part}] 處理完成。")

    print("\n所有實驗執行完畢！準備匯出 CSV...")

    if len(global_records) > 0:
        results_df = pd.DataFrame(global_records)

        # 1. 儲存 Raw Data (這份表每一行對應到 CSV 中的一筆有效資料)
        results_df.to_csv('part4_all_taus_parts_raw.csv', index=False)
        print("已儲存原始細節資料: part4_all_taus_parts_raw.csv")

        # 2. 製作並儲存綜合 Summary Table (將 Subject 群組平均，只看不同 Tau, Part, Case 的大趨勢)
        summary_df = results_df.groupby(['Tau', 'BodyPart', 'Case'])[['Velocity', 'Diversity']].mean().reset_index()
        summary_df = summary_df.round(4)
        summary_df.to_csv('part4_all_taus_parts_summary.csv', index=False)
        
        print("\n已儲存綜合統計表: part4_all_taus_parts_summary.csv")
        print("\n資料預覽 (Summary):")
        print(summary_df.head(10).to_string(index=False))
    else:
        print("警告：未收集到任何有效資料，無法匯出 CSV。")