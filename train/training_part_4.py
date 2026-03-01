import os
import copy
import time
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import argparse
import math
from collections import defaultdict
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import (
    accuracy_score, 
    balanced_accuracy_score, 
    precision_recall_fscore_support,
    roc_auc_score,           
    average_precision_score  
)
from torch.utils.data.dataloader import default_collate


CASE_MAP = {'A': 0, 'B': 1, 'C': 2, 'D': 3}
inv_CASE_MAP = {v: k for k, v in CASE_MAP.items()}



PARTS_DEF = {
    'FullBody': [''], 
    'Hands': ['Hand', 'Wrist'],
    'Head': ['Head', 'Neck'],
    'UpperBody': ['Spine', 'Neck', 'Head', 'Shoulder', 'Arm', 'Hand']
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
                        indices = list(range(current_channel_idx, current_channel_idx + num_channels))
                        self.channel_map.append({
                            "name": self.joint_names[-1],
                            "indices": indices
                        })
                        current_channel_idx += num_channels
        except Exception as e:
            print(f"Error parsing reference BVH: {e}")

    def get_indices(self, part_name):
        keywords = PARTS_DEF.get(part_name, [''])
        target_indices = []
        for item in self.channel_map:
            if keywords == ['']:
                target_indices.extend(item['indices'])
                continue
            if any(k.lower() in item['name'].lower() for k in keywords):
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

class MotionTurnTakingDataset(Dataset):
    def __init__(self, df, motion_folder, baselines=None, feature_indices=None, target_fps=120):
        self.motion_folder = motion_folder
        self.baselines = baselines
        self.feature_indices = feature_indices
        self.fps = target_fps
        self.downsample_step = 4
        self.offset_seconds = 0.0 

        self.window_duration = 3.0
        self.raw_window_size = int(self.window_duration * target_fps)
        self.model_input_len = self.raw_window_size // self.downsample_step

        df['subject_str'] = df['subject_id'].apply(lambda x: str(int(x)).zfill(3))
        if 'scenario' in df.columns:
            df['scenario_str'] = df['scenario'].apply(lambda x: str(int(x)).zfill(3))
            
        df['date_str'] = df['date'].astype(str)
        df['label'] = df['case'].apply(lambda x: 1 if x in ['B', 'C'] else 0)
        df['case_id'] = df['case'].map(CASE_MAP)
        
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
            
            if self.feature_indices is not None:
                full_motion = full_motion[:, self.feature_indices]

            total_frames = full_motion.shape[0]
            
            anchor_time = row['start_time'] 
            window_end_time = anchor_time - self.offset_seconds
            window_start_time = window_end_time - self.window_duration
            
            start_frame = int(round(window_start_time * self.fps))
            end_frame = int(round(window_end_time * self.fps))
            
            if start_frame < 0 or end_frame > total_frames: return None
            if (end_frame - start_frame) < self.raw_window_size: return None
            
            clip_data = full_motion[start_frame : end_frame : self.downsample_step].copy()
            clip = torch.from_numpy(clip_data).float()
            
            target_len = self.model_input_len
            if clip.shape[0] < target_len:
                pad_len = target_len - clip.shape[0]
                clip = torch.cat([clip, torch.zeros((pad_len, clip.shape[1]))], dim=0)
            if clip.shape[0] > target_len:
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
        features, labels, _ = batch
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

def evaluate_detailed(model, dataloader, device):

    model.eval()
    all_preds, all_probs, all_labels, all_cases = [], [], [], []

    with torch.no_grad():
        for batch in dataloader:
            if batch is None: continue
            features, labels, cases = batch
            features, labels = features.to(device), labels.to(device)
            
            outputs = model(features).reshape(-1)
            probs = torch.sigmoid(outputs)
            preds = (probs > 0.5).float()
            
            all_probs.extend(probs.cpu().tolist())
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())
            all_cases.extend(cases)
            
    if not all_preds: 
        print("!! WARNING: No data in evaluation loop !!")
        return {}, {}

    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)

    overall_metrics = {}
    
    overall_metrics['acc'] = accuracy_score(all_labels, all_preds)
    overall_metrics['bal_acc'] = balanced_accuracy_score(all_labels, all_preds)
    precision, recall, f1, support = precision_recall_fscore_support(
        all_labels, all_preds, labels=[0, 1], zero_division=0
    )
    
    overall_metrics['f1_hold'] = f1[0]
    overall_metrics['f1_shift'] = f1[1]
    overall_metrics['prec_hold'] = precision[0]
    overall_metrics['prec_shift'] = precision[1]
    overall_metrics['rec_hold'] = recall[0]
    overall_metrics['rec_shift'] = recall[1]
    
    total_supp = support.sum()
    overall_metrics['f1_wgt'] = (f1[0]*support[0] + f1[1]*support[1])/total_supp if total_supp > 0 else 0

    try:
        overall_metrics['auroc'] = roc_auc_score(all_labels, all_probs)
        overall_metrics['auprc'] = average_precision_score(all_labels, all_probs)
    except ValueError:
        overall_metrics['auroc'] = np.nan
        overall_metrics['auprc'] = np.nan

    # 4. Case-wise Metrics
    case_metrics = {}
    df_res = pd.DataFrame({'case': all_cases, 'label': all_labels, 'pred': all_preds})
    
    for c in ['A', 'B', 'C', 'D']:
        sub = df_res[df_res['case'] == c]
        if len(sub) > 0:
            case_metrics[f'Acc_{c}'] = accuracy_score(sub['label'], sub['pred'])
        else:
            case_metrics[f'Acc_{c}'] = np.nan
            
    return overall_metrics, case_metrics

def get_session_id_splits(df_base, n_cv_splits=5, seed=42):
    df_base['session_key'] = df_base['date'].astype(str) + '_' + df_base['scenario'].astype(str) + '_' + df_base['pair'].astype(str)
    groups = pd.factorize(df_base['session_key'])[0]
    
    y = df_base['case'].apply(lambda x: 1 if x in ['B', 'C'] else 0).values
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=str, default='1')
    parser.add_argument('--ref_bvh', type=str, required=True, help="Path to reference BVH for skeleton structure")
    parser.add_argument('--manifest', type=str, required=True, help="CSV containing [motion_start, case, etc.]")
    parser.add_argument('--baselines', type=str, required=True, help="JSON file for normalization baselines")
    args = parser.parse_args()

    base_dir = 'InterAct_pause_analysis'
    motion_npy_folder = os.path.join(base_dir, 'processed_motion_npy')
    
    manifest_name = os.path.basename(args.manifest).replace('.csv', '')
    
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(f"Running Prediction for: {manifest_name} on {device}")
    

    indexer = BodyPartIndexer(args.ref_bvh)
    baselines_data = load_baselines(args.baselines)
    full_df = pd.read_csv(args.manifest)
    
    if 'case' not in full_df.columns and 'tmp_case' in full_df.columns:
        full_df.rename(columns={'tmp_case': 'case'}, inplace=True)

    test_keys, cv_fold_keys = get_session_id_splits(full_df, n_cv_splits=5)
    df_test = full_df[full_df['session_key'].isin(test_keys)].reset_index(drop=True)
    

    all_parts_results = []
    
    for target_body_part in PARTS_DEF.keys():
        
        part_indices = indexer.get_indices(target_body_part)
        input_dim = len(part_indices)
        
        print(f"\n{'='*60}")
        print(f"Evaluating Body Part: {target_body_part} | Dim: {input_dim}")
        print(f"{'='*60}")
        
        test_ds = MotionTurnTakingDataset(
            df_test, motion_npy_folder, 
            baselines=baselines_data, 
            feature_indices=part_indices
        )
        test_loader = DataLoader(test_ds, batch_size=64, shuffle=False, collate_fn=collate_fn_skip_none)
        
        fold_results = defaultdict(list)
        
        for fold_k, (train_keys, val_keys) in enumerate(cv_fold_keys):
            print(f"  Fold {fold_k+1}/5...", end=' ', flush=True)
            
            df_train = full_df[full_df['session_key'].isin(train_keys)].reset_index(drop=True)
            df_val = full_df[full_df['session_key'].isin(val_keys)].reset_index(drop=True)
            
            train_loader = DataLoader(
                MotionTurnTakingDataset(df_train, motion_npy_folder, baselines=baselines_data, feature_indices=part_indices),
                batch_size=64, shuffle=True, num_workers=4, collate_fn=collate_fn_skip_none
            )
            val_loader = DataLoader(
                MotionTurnTakingDataset(df_val, motion_npy_folder, baselines=baselines_data, feature_indices=part_indices),
                batch_size=64, shuffle=False, num_workers=4, collate_fn=collate_fn_skip_none
            )
            
            labels_train = df_train['case'].apply(lambda x: 1 if x in ['B','C'] else 0).values
            num_pos = np.sum(labels_train)
            num_neg = len(labels_train) - num_pos
            pos_weight_val = num_neg / num_pos if num_pos > 0 else 1.0
            pos_weight = torch.tensor([pos_weight_val]).to(device)
            
            model = TurnTakingTransformer(input_dim=input_dim).to(device)
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
            loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
            
            best_val_f1 = -1
            best_wts = None
            patience = 5
            counter = 0
            
            for epoch in range(25):
                _ = train_one_epoch(model, train_loader, optimizer, loss_fn, device)
                val_overall, _ = evaluate_detailed(model, val_loader, device)
                val_f1_wgt = val_overall.get('f1_wgt', 0.0)
                
                if val_f1_wgt > best_val_f1:
                    best_val_f1 = val_f1_wgt
                    best_wts = copy.deepcopy(model.state_dict())
                    counter = 0
                else:
                    counter += 1
                if counter >= patience: break
            
            model.load_state_dict(best_wts)
            test_overall, test_case = evaluate_detailed(model, test_loader, device)
            
            for k, v in test_overall.items(): fold_results[k].append(v)
            for k, v in test_case.items(): fold_results[k].append(v)
            
            print(f"Done. AUROC: {test_overall.get('auroc',0):.4f} | AUPRC: {test_overall.get('auprc',0):.4f}")


        avg_results = {k: np.nanmean(v) for k, v in fold_results.items()}
        
        percent_metrics = [
            'acc', 'bal_acc', 'f1_wgt', 'f1_hold', 'f1_shift',
            'prec_hold', 'prec_shift', 'rec_hold', 'rec_shift',
            'Acc_A', 'Acc_B', 'Acc_C', 'Acc_D'
        ]
        for m in percent_metrics:
            if m in avg_results:
                avg_results[m] = round(avg_results[m] * 100, 2)
                
        if 'auroc' in avg_results: avg_results['auroc'] = round(avg_results['auroc'], 4)
        if 'auprc' in avg_results: avg_results['auprc'] = round(avg_results['auprc'], 4)

        res_str = (
            f"\nResult for {target_body_part} ({manifest_name}):\n"
            f"  > Metrics:\n"
            f"    - Acc       : {avg_results.get('acc', 0):.2f}% | Bal Acc : {avg_results.get('bal_acc', 0):.2f}%\n"
            f"    - F1 Wgt    : {avg_results.get('f1_wgt', 0):.2f}%\n"
            f"    - AUROC     : {avg_results.get('auroc', 0):.4f}  | AUPRC   : {avg_results.get('auprc', 0):.4f}\n"
            f"  > Class-wise (Hold=0, Shift=1):\n"
            f"    - Precision : Hold={avg_results.get('prec_hold', 0):.2f}% | Shift={avg_results.get('prec_shift', 0):.2f}%\n"
            f"    - Recall    : Hold={avg_results.get('rec_hold', 0):.2f}% | Shift={avg_results.get('rec_shift', 0):.2f}%\n"
            f"    - F1        : Hold={avg_results.get('f1_hold', 0):.2f}% | Shift={avg_results.get('f1_shift', 0):.2f}%\n"
            f"  > Case Acc: A={avg_results.get('Acc_A', np.nan):.2f}%, "
            f"B={avg_results.get('Acc_B', np.nan):.2f}%, "
            f"C={avg_results.get('Acc_C', np.nan):.2f}%, "
            f"D={avg_results.get('Acc_D', np.nan):.2f}%\n"
            f"\n"
        )
        print(res_str)

        txt_report_path = os.path.join(base_dir, 'shifted_window_full_report.txt')
        with open(txt_report_path, 'a') as f:
            f.write(res_str)

 
        avg_results['Manifest'] = manifest_name
        avg_results['BodyPart'] = target_body_part
        all_parts_results.append(avg_results)


    df_out = pd.DataFrame(all_parts_results)
    
 
    cols = ['Manifest', 'BodyPart'] + [c for c in df_out.columns if c not in ['Manifest', 'BodyPart']]
    df_out = df_out[cols]
    
    csv_out_path = os.path.join(base_dir, f'results_summary_{manifest_name}.csv')
    df_out.to_csv(csv_out_path, index=False)
    