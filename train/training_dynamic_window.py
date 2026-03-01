import os
import copy
import time
import re
import numpy as np
import pandas as pd
import json
import torch
import torch.nn as nn
import argparse
import math
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import balanced_accuracy_score, accuracy_score, f1_score
from torch.utils.data.dataloader import default_collate



PARTS_DEF = {
    'fullbody': [''], 
    'hands': ['Hand', 'Wrist'],
    'upper_body': ['Spine', 'Neck', 'Head', 'Shoulder', 'Arm', 'Hand'],
    'head': ['Head', 'Neck']
}

class BodyPartIndexer:
    def __init__(self, ref_bvh_path):
        self.joint_names = []
        self.channel_map = []
        self._parse_header(ref_bvh_path)

    def _parse_header(self, file_path):
        current_channel_idx = 0
        try:
            with open(file_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line.startswith("MOTION"): break
                    if line.startswith("ROOT") or line.startswith("JOINT"):
                        self.joint_names.append(line.split()[1])
                    if line.startswith("CHANNELS"):
                        parts = line.split()
                        num_channels = int(parts[1])
                        channel_types = parts[2:]
                        
                        rot_indices = []
                        for j, c_type in enumerate(channel_types):
                            if 'rotation' in c_type.lower():
                                rot_indices.append(current_channel_idx + j)
                        
                        self.channel_map.append({
                            "name": self.joint_names[-1],
                            "indices": rot_indices
                        })
                        current_channel_idx += num_channels
        except Exception as e:
            print(f"Error parsing reference BVH: {e}")

    def get_indices(self, part_name):
        keywords = PARTS_DEF.get(part_name, [''])
        target_indices = []
        for item in self.channel_map:
            if keywords == [''] or any(k.lower() in item['name'].lower() for k in keywords):
                target_indices.extend(item['indices'])
        target_indices.sort()
        return target_indices

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

class TurnTakingTransformer(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.d_model = 256
        self.nhead = 4
        self.d_ffn = 768
        self.dropout_prob = 0.1
        self.num_layers = 4 
        
        self.input_proj = nn.Linear(input_dim, self.d_model)
        self.bn_in = nn.BatchNorm1d(self.d_model)
        self.pos_encoder = PositionalEncoding(self.d_model, dropout=self.dropout_prob)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model, nhead=self.nhead,
            dim_feedforward=self.d_ffn, dropout=self.dropout_prob,
            activation='gelu', batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=self.num_layers)
        
        self.classifier = nn.Sequential(
            nn.Linear(self.d_model, self.d_model // 2),
            nn.GELU(),
            nn.Dropout(self.dropout_prob),
            nn.Linear(self.d_model // 2, 1)
        )

    def forward(self, x):
        x = self.input_proj(x)
        x = x.permute(0, 2, 1)
        x = self.bn_in(x)
        x = x.permute(0, 2, 1)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)
        final_state = x.mean(dim=1)
        logits = self.classifier(final_state)
        return logits


def load_baselines(json_path):
    if not os.path.exists(json_path):
        print("Warning: Baseline file not found.")
        return None
    with open(json_path, 'r') as f:
        return json.load(f)


class FineGrainedMotionDataset(Dataset):
    def __init__(self, df, motion_folder, baselines=None, feature_indices=None, target_fps=120):
        self.motion_folder = motion_folder
        self.baselines = baselines
        self.fps = target_fps
        self.downsample_step = 4
        self.feature_indices = feature_indices 

        if 'window_duration' in df.columns:
            self.max_duration = df.iloc[0]['window_duration']
        else:
            self.max_duration = 0.5 
            
        self.max_raw_frames = int(round(self.max_duration * self.fps))
        self.model_input_len = self.max_raw_frames // self.downsample_step

        df['subject_str'] = df['subject_id'].apply(lambda x: str(int(x)).zfill(3))
        df['scenario_str'] = df['scenario'].apply(lambda x: str(int(x)).zfill(3))
        df['date_str'] = df['date'].astype(str)
        df['label'] = df['label'].astype(float)
        
        if 'case' not in df.columns:
            df['case'] = 'unknown'
        df['case'] = df['case'].astype(str)

        self.data = df.to_dict('records')

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data[idx]
        filename = f"{row['date_str']}_{row['subject_str']}_{row['scenario_str']}.npy"
        file_path = os.path.join(self.motion_folder, filename)
        
        if not os.path.exists(file_path): return None

        try:
            full_motion = np.load(file_path, mmap_mode='r')
            
            if self.baselines is not None:
                spk_id = row['subject_str']
                if spk_id in self.baselines:
                    mean = np.array(self.baselines[spk_id]['mean'])
                    std = np.array(self.baselines[spk_id]['std'])
                    full_motion = (full_motion - mean) / (std + 1e-8)
            
            total_frames = full_motion.shape[0]
            
            start_frame = int(round(row['start_time'] * self.fps))
            end_frame = int(round(row['end_time'] * self.fps))
            
            start_frame = max(0, start_frame)
            end_frame = min(total_frames, end_frame)
            
            if end_frame <= start_frame: return None

            clip_data = full_motion[start_frame : end_frame : self.downsample_step].copy()
            
            if self.feature_indices is not None:
                clip_data = clip_data[:, self.feature_indices]
                
            clip = torch.from_numpy(clip_data).float()
            
            current_len = clip.shape[0]
            target_len = self.model_input_len
            feat_dim = clip.shape[1]
            
            if current_len < target_len:
                pad_len = target_len - current_len
                pad_tensor = torch.zeros((pad_len, feat_dim))
                clip = torch.cat([clip, pad_tensor], dim=0)

            elif current_len > target_len:
                clip = clip[:target_len]
            
            return clip, torch.tensor(row['label'], dtype=torch.float32), row['case']

        except Exception as e:
            return None

def collate_fn_skip_none(batch):
    batch = [item for item in batch if item is not None]
    if len(batch) == 0: return None
    return default_collate(batch)


def train_one_epoch(model, dataloader, optimizer, loss_fn, device):
    model.train()
    total_loss = 0
    valid_batches = 0
    for batch in dataloader:
        if batch is None: continue
        features, labels, cases = batch
        features, labels = features.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(features).reshape(-1)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        valid_batches += 1
    if valid_batches == 0: return 0.0
    return total_loss / valid_batches

def evaluate(model, dataloader, device):
    model.eval()
    results = {'preds': [], 'labels': [], 'case': []}
    
    with torch.no_grad():
        for batch in dataloader:
            if batch is None: continue
            features, labels, cases = batch
            features, labels = features.to(device), labels.to(device)
            
            outputs = model(features).reshape(-1)
            probs = torch.sigmoid(outputs)
            preds = (probs > 0.5).float()
            
            results['preds'].extend(preds.cpu().tolist())
            results['labels'].extend(labels.cpu().tolist())
            results['case'].extend(cases)
            
    df_res = pd.DataFrame(results)
    if df_res.empty: return 0.0, 0.0, 0.0, 0.0, pd.DataFrame()
    
    # Overall Metrics 
    overall_f1_shift = f1_score(df_res['labels'], df_res['preds'], zero_division=0, pos_label=1)
    overall_f1_weighted = f1_score(df_res['labels'], df_res['preds'], zero_division=0, average='weighted')
    overall_bacc = balanced_accuracy_score(df_res['labels'], df_res['preds'])
    overall_acc = accuracy_score(df_res['labels'], df_res['preds'])
    
    # Per-Case Metrics
    case_metrics_list = []
    for case_id, group in df_res.groupby('case'):
        c_f1 = f1_score(group['labels'], group['preds'], zero_division=0, pos_label=1)
        c_acc = accuracy_score(group['labels'], group['preds'])
        case_metrics_list.append({
            'case': case_id,
            'f1_shift': c_f1,
            'acc': c_acc,
            'count': len(group)
        })
    df_case_metrics = pd.DataFrame(case_metrics_list)
    
    return overall_f1_shift, overall_f1_weighted, overall_bacc, overall_acc, df_case_metrics

def get_session_id_splits(df_base, n_cv_splits=5, seed=42):
    df_base['session_key'] = df_base['date'].astype(str) + '_' + df_base['scenario'].astype(str) + '_' + df_base['pair'].astype(str)
    groups = pd.factorize(df_base['session_key'])[0]
    y = df_base['label'].values
    X = np.arange(len(df_base))
    
    sgkf_test = StratifiedGroupKFold(n_splits=20, shuffle=True, random_state=seed)
    train_val_idxs, test_idxs = next(sgkf_test.split(X, y, groups=groups))
    test_session_keys = df_base.iloc[test_idxs]['session_key'].unique()
    
    X_remain = X[train_val_idxs]
    y_remain = y[train_val_idxs]
    groups_remain = groups[train_val_idxs]
    
    sgkf_cv = StratifiedGroupKFold(n_splits=n_cv_splits, shuffle=True, random_state=seed)
    cv_folds_ids = []
    
    for t_idx_rel, v_idx_rel in sgkf_cv.split(X_remain, y_remain, groups=groups_remain):
        train_idxs = X_remain[t_idx_rel]
        val_idxs = X_remain[v_idx_rel]
        train_keys = df_base.iloc[train_idxs]['session_key'].unique()
        val_keys = df_base.iloc[val_idxs]['session_key'].unique()
        cv_folds_ids.append((train_keys, val_keys))
        
    return test_session_keys, cv_folds_ids

def analyze_temporal_importance(model, dataloader, device, window_duration):

    model.eval()
    all_saliency = []
    all_cases = []
    
    limit_batches = 30 
    
    for i, batch in enumerate(dataloader):
        if batch is None: continue
        if i >= limit_batches: break
        
        features, labels, cases = batch
        features = features.to(device)
        labels = labels.to(device)
        
        features.requires_grad_()
        outputs = model(features).reshape(-1)
        outputs.sum().backward()
        
        gradients = features.grad.abs() # (Batch, Time, Dim)
        time_importance = gradients.mean(dim=2) # (Batch, Time)
        
        all_saliency.append(time_importance.detach().cpu().numpy())
        all_cases.extend(cases)
        
    if not all_saliency:
        return None, {}

    saliency_matrix = np.concatenate(all_saliency, axis=0) # (Total_Samples, Time)
    case_array = np.array(all_cases)
    
    results_dict = {}
    time_steps = saliency_matrix.shape[1]
    time_axis = np.linspace(-window_duration, 0, time_steps)
    
    avg_overall = np.mean(saliency_matrix, axis=0)
    norm_overall = (avg_overall - avg_overall.min()) / (avg_overall.max() - avg_overall.min() + 1e-8)
    results_dict['Overall'] = norm_overall
    
    unique_cases = np.unique(case_array)
    for c in unique_cases:
        indices = np.where(case_array == c)[0]
        if len(indices) < 5: continue 
        
        avg_case = np.mean(saliency_matrix[indices], axis=0)
        norm_case = (avg_case - avg_case.min()) / (avg_case.max() - avg_case.min() + 1e-8)
        results_dict[c] = norm_case
        
    return time_axis, results_dict



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Turn-taking Prediction")
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--ref_bvh', type=str, required=True)
    parser.add_argument('--manifest', type=str, required=True)
    args = parser.parse_args()
    baselines = 'speaker_norm_6D_fps30_baselines.json'
    # Settings
    base_dir = ''
    motion_npy_folder = os.path.join(base_dir, 'processed_motion_npy')
    
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(f"Running on {device} | Manifest: {args.manifest}")

    match = re.search(r'window_([0-9\.]+)s', args.manifest)
    current_duration = float(match.group(1)) if match else 0.5
    print(f"Detected Window Duration for Saliency: {current_duration}s")

    # 1. Load Data
    full_df = pd.read_csv(args.manifest)
    test_keys, cv_fold_keys = get_session_id_splits(full_df, n_cv_splits=5)
    df_test = full_df[full_df['session_key'].isin(test_keys)].reset_index(drop=True)
    
    indexer = BodyPartIndexer(args.ref_bvh)

    final_results = []
    final_case_breakdown = []
    
    target_parts = ['fullbody', 'hands', 'head', 'upper_body']
    
    
    for part_name in target_parts:
        print(f"\n{'='*30}")
        print(f"Processing Body Part: {part_name}")
        
        selected_indices = indexer.get_indices(part_name)
        input_feat_dim = len(selected_indices)
        
        test_loader = DataLoader(
            FineGrainedMotionDataset(df_test, motion_npy_folder, baselines=baselines, feature_indices=selected_indices), 
            batch_size=64, shuffle=False, collate_fn=collate_fn_skip_none, num_workers=4
        )
        
        fold_scores = []
        fold_case_scores = []

        for fold_k, (train_keys, val_keys) in enumerate(cv_fold_keys):
            print(f"  Fold {fold_k+1}/5...", end=' ', flush=True)
            
            df_train = full_df[full_df['session_key'].isin(train_keys)].reset_index(drop=True)
            df_val = full_df[full_df['session_key'].isin(val_keys)].reset_index(drop=True)
            
            train_loader = DataLoader(
                FineGrainedMotionDataset(df_train, motion_npy_folder, baselines=baselines, feature_indices=selected_indices), 
                batch_size=64, shuffle=True, num_workers=4, collate_fn=collate_fn_skip_none
            )
            val_loader = DataLoader(
                FineGrainedMotionDataset(df_val, motion_npy_folder, baselines=baselines, feature_indices=selected_indices), 
                batch_size=64, shuffle=False, num_workers=4, collate_fn=collate_fn_skip_none
            )
            
            train_labels = df_train['label'].values
            num_pos = np.sum(train_labels)
            pos_weight = torch.tensor([(len(train_labels)-num_pos)/num_pos if num_pos > 0 else 1.0]).to(device)

            model = TurnTakingTransformer(input_dim=input_feat_dim).to(device)
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
            loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
            
            best_val_f1 = -1
            best_wts = None
            patience = 5
            counter = 0
            
            for epoch in range(25):
                train_loss = train_one_epoch(model, train_loader, optimizer, loss_fn, device)
                val_f1_shift, val_f1_weighted, val_bacc, val_acc, _ = evaluate(model, val_loader, device)

                if val_f1_shift > best_val_f1:
                    best_val_f1 = val_f1_shift
                    best_wts = copy.deepcopy(model.state_dict())
                    counter = 0
                else:
                    counter += 1
                if counter >= patience: break
            
            model.load_state_dict(best_wts)
            test_f1_s, test_f1_w, test_bacc, test_acc, df_test_cases = evaluate(model, test_loader, device)
            
            fold_scores.append({
                'f1_shift': test_f1_s, 
                'f1_weighted': test_f1_w, 
                'bacc': test_bacc, 
                'acc': test_acc
            })
            df_test_cases['fold'] = fold_k
            fold_case_scores.append(df_test_cases)
            print(f"Test F1(Shift): {test_f1_s:.4f} | BACC: {test_bacc:.4f}")

        print(f"  Analyzing Temporal Importance for {part_name}...")
        time_axis, importance_dict = analyze_temporal_importance(model, test_loader, device, current_duration)

        if importance_dict:
            imp_df_list = []
            for c_key, c_imp in importance_dict.items():
                temp_df = pd.DataFrame({'time': time_axis, 'importance': c_imp})
                temp_df['case'] = c_key
                imp_df_list.append(temp_df)
            
            if imp_df_list:
                full_imp_df = pd.concat(imp_df_list)
                full_imp_df.to_csv(os.path.join(base_dir, f"saliency_data_{part_name}_{current_duration}s.csv"), index=False)

        avg_f1_s = np.mean([x['f1_shift'] for x in fold_scores])
        avg_f1_w = np.mean([x['f1_weighted'] for x in fold_scores])
        avg_bacc = np.mean([x['bacc'] for x in fold_scores])
        avg_acc = np.mean([x['acc'] for x in fold_scores])
        
        final_results.append({
            'BodyPart': part_name,
            'Avg_F1_Shift': avg_f1_s,
            'Avg_F1_Weighted': avg_f1_w,
            'Avg_BACC': avg_bacc,
            'Avg_Acc': avg_acc,
            'Window_File': os.path.basename(args.manifest)
        })
        
        if fold_case_scores:
            all_cases_df = pd.concat(fold_case_scores, ignore_index=True)
            avg_case_metrics = all_cases_df.groupby('case').agg({'f1_shift': 'mean', 'acc': 'mean', 'count': 'sum'}).reset_index()
            avg_case_metrics['BodyPart'] = part_name
            avg_case_metrics['Window_File'] = os.path.basename(args.manifest)
            final_case_breakdown.append(avg_case_metrics)

    file_suffix = os.path.basename(args.manifest).replace('.csv', '')
    
    df_summary = pd.DataFrame(final_results)
    df_summary.to_csv(os.path.join(base_dir, f"results_summary_{file_suffix}.csv"), index=False)
    
    if final_case_breakdown:
        df_cases = pd.concat(final_case_breakdown, ignore_index=True)
        df_cases.to_csv(os.path.join(base_dir, f"results_cases_{file_suffix}.csv"), index=False)
    
    print("\n=== All Done ===")