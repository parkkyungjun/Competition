import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import cv2
import os
import glob
import sys
import random
import numpy as np
import shutil
import timm
from tqdm import tqdm
from decord import VideoReader, cpu
from sklearn.metrics import roc_auc_score, f1_score

# --- [Logger Class 추가] ---
class Logger(object):
    def __init__(self, file_name="training_log.txt"):
        self.terminal = sys.stdout
        self.log = open(file_name, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush() # 바로 파일에 쓰도록 flush

    def flush(self):
        self.terminal.flush()
        self.log.flush()

# --- [Effort Layer & Model 정의] ---
class EffortLinear(nn.Module):
    """
    [Paper Implementation] SVD-based Orthogonal Decomposition Layer.
    W = W_semantic (Frozen) + W_forgery (Trainable)
    """
    def __init__(self, original_linear: nn.Linear, rank_ratio: float = 0.8):
        super().__init__()
        
        weight = original_linear.weight.data # (Out, In)
        out_features, in_features = weight.shape
        
        if original_linear.bias is not None:
            self.bias = nn.Parameter(original_linear.bias.data)
        else:
            self.register_parameter('bias', None)

        # SVD 수행
        U, S, Vh = torch.linalg.svd(weight.float(), full_matrices=False)
        
        full_rank = min(out_features, in_features)
        k = int(full_rank * rank_ratio)
        if k == 0: k = 1 
        
        # Frozen (Semantic)
        self.register_buffer('U_p', U[:, :k].clone())
        self.register_buffer('S_p', S[:k].clone())
        self.register_buffer('Vh_p', Vh[:k, :].clone())
        
        # Trainable (Forgery/Residual)
        self.U_r = nn.Parameter(U[:, k:].clone())
        self.S_r = nn.Parameter(S[k:].clone())
        self.Vh_r = nn.Parameter(Vh[k:, :].clone())

    def forward(self, x):
        W_semantic = self.U_p @ torch.diag(self.S_p) @ self.Vh_p
        W_trainable = self.U_r @ torch.diag(self.S_r) @ self.Vh_r
        W_total = W_semantic + W_trainable
        
        return nn.functional.linear(x, W_total.type_as(x), self.bias)

def replace_modules(model, rank_ratio, target_suffixes):
    """재귀적으로 모델 트리를 내려가며 레이어를 교체"""
    for name, child in model.named_children():
        if isinstance(child, nn.Linear):
            # 현재 레이어 이름이 타겟 접미사 중 하나로 끝나는지 확인 (예: qkv, fc1 ...)
            if any(name.endswith(suffix) for suffix in target_suffixes):
                # print(f"  -> Replacing Layer: {name}")
                new_layer = EffortLinear(child, rank_ratio=rank_ratio)
                setattr(model, name, new_layer)
        else:
            replace_modules(child, rank_ratio, target_suffixes)

class DeepfakeEffortModel(nn.Module):
    def __init__(self, model_name='swin_base_patch4_window12_384', num_classes=1, rank_ratio=0.75):
        super().__init__()
        
        print(f"Loading Backbone: {model_name}...")
        # num_classes=1로 설정 (BCEWithLogitsLoss 사용 위함)
        self.backbone = timm.create_model(model_name, pretrained=True, num_classes=num_classes)
        
        # Swin Transformer 타겟 모듈
        target_layers = ['qkv', 'proj', 'fc1', 'fc2', 'head']
        
        print(f"Injecting Effort Layers (Rank Ratio: {rank_ratio})...")
        replace_modules(self.backbone, rank_ratio, target_layers)
        
        # 전체 Freeze 후 Effort 파트만 Unfreeze
        for param in self.backbone.parameters():
            param.requires_grad = False
            
        trainable_count = 0
        for name, module in self.backbone.named_modules():
            if isinstance(module, EffortLinear):
                module.U_r.requires_grad = True
                module.S_r.requires_grad = True
                module.Vh_r.requires_grad = True
                if module.bias is not None:
                    module.bias.requires_grad = True
                
                trainable_count += module.U_r.numel() + module.S_r.numel() + module.Vh_r.numel()

        print(f"Model Ready. Trainable Parameters (SVD Residuals): {trainable_count:,}")

    def forward(self, x):
        return self.backbone(x)


# --- [Utils] ---
def seed_torch(seed=8746):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# --- [Dataset & Loader] ---
CROP_SIZE = 384  # Swin Base 384 모델에 맞춤

class MixedContentDataset(Dataset):
    def __init__(self, class1_files, class0_files, transform):
        self.transform = transform
        self.samples = []
        for f in class1_files: self.samples.append((f, 1))
        for f in class0_files: self.samples.append((f, 0))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        filepath, label = self.samples[idx]

        try:
            if filepath.endswith(('.mp4', '.avi', '.mkv')):
                vr = VideoReader(filepath, ctx=cpu(0))
                random_idx = random.randint(0, len(vr) - 1)
                frame = vr[random_idx]
                img = Image.fromarray(frame.asnumpy())
            else:
                img = Image.open(filepath).convert('RGB')
            
            # 간단한 리사이징 (필요 시 유지)
            # w, h = img.size
            # if w % 2 == 1: w += 1
            # if h % 2 == 1: h += 1
            # img = img.resize((w, h))
            
            img = self.transform(img)
            return img, torch.tensor(label, dtype=torch.float32), filepath
            
        except Exception as e:
            print(f"Error loading {filepath}: {e}")
            # 에러 발생 시 빈 텐서 혹은 랜덤 텐서 반환 (학습 중단 방지용)
            return torch.zeros((3, CROP_SIZE, CROP_SIZE)), torch.tensor(label, dtype=torch.float32), filepath

class BalancedBatchSampler(torch.utils.data.Sampler):
    def __init__(self, dataset, batch_size):
        self.dataset = dataset
        self.batch_size = batch_size
        self.indices_c0 = [i for i, (_, label) in enumerate(dataset.samples) if label == 0]
        self.indices_c1 = [i for i, (_, label) in enumerate(dataset.samples) if label == 1]
        
        assert batch_size % 2 == 0, "Batch size must be even."
        
        self.max_len = max(len(self.indices_c0), len(self.indices_c1))
        self.n_batches = self.max_len // (self.batch_size // 2)
        
    def __iter__(self):
        idx0 = np.array(self.indices_c0)
        idx1 = np.array(self.indices_c1)
        np.random.shuffle(idx0)
        np.random.shuffle(idx1)
        
        target_len = self.n_batches * (self.batch_size // 2)
        idx0 = np.resize(idx0, target_len)
        idx1 = np.resize(idx1, target_len)
        
        for i in range(self.n_batches):
            start = i * (self.batch_size // 2)
            end = start + (self.batch_size // 2)
            batch_indices = np.concatenate([idx0[start:end], idx1[start:end]])
            np.random.shuffle(batch_indices)
            yield batch_indices.astype(int)

    def __len__(self):
        return self.n_batches

def get_data_loaders(batch_size):
    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.webp', '.mp4')
    
    # === [경로 확인 필요] 사용자의 실제 데이터 경로로 수정하세요 ===
    path_class1 = 'processed_dataset_384_padded/fake' 
    path_class0_list = ['processed_dataset_384_padded/real']
    # ========================================================
    
    train_c1 = glob.glob(os.path.join(path_class1, '**', '*'), recursive=True)
    train_c1 = [f for f in train_c1 if f.lower().endswith(valid_extensions)]
    
    train_c0 = []
    for p in path_class0_list:
        temp = glob.glob(os.path.join(p, '**', '*'), recursive=True)
        valid_files = [f for f in temp if f.lower().endswith(valid_extensions)]
        # 데이터 밸런스를 위해 샘플링 (필요시 조정)
        train_c0 += valid_files 
    
    print(f"Train Real: {len(train_c0)}, Train Fake: {len(train_c1)}")

    # Validation 경로
    val_c1 = glob.glob('valid/fake/*')
    val_c0 = glob.glob('valid/real/*')
    val_c1 = [f for f in val_c1 if f.lower().endswith(valid_extensions)]
    val_c0 = [f for f in val_c0 if f.lower().endswith(valid_extensions)]

    transform = transforms.Compose([
        transforms.Resize((CROP_SIZE, CROP_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    train_dataset = MixedContentDataset(train_c1, train_c0, transform)
    val_dataset = MixedContentDataset(val_c1, val_c0, transform)
    
    train_sampler = BalancedBatchSampler(train_dataset, batch_size=batch_size)
    
    train_loader = DataLoader(
        train_dataset, batch_sampler=train_sampler, num_workers=4, pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True
    )
    
    return train_loader, val_loader

# --- [Training Functions] ---
def train_one_epoch(model, loader, criterion, optimizer, scaler, device, epoch):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    loop = tqdm(loader, desc=f"Epoch {epoch}", leave=True)
    
    for images, labels, _ in loop:
        images = images.to(device)
        labels = labels.to(device).unsqueeze(1) # (B) -> (B, 1)

        with torch.cuda.amp.autocast():
            outputs = model(images)
            loss = criterion(outputs, labels)

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item()
        
        probs = torch.sigmoid(outputs)
        preds = (probs > 0.5).float()
        correct += (preds == labels).sum().item()
        total += labels.size(0)

        loop.set_postfix(loss=loss.item(), acc=(correct/total)*100)

    return running_loss / len(loader), (correct / total) * 100

def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for images, labels, _ in tqdm(loader, desc="Validating", leave=False):
            images = images.to(device)
            labels = labels.to(device).unsqueeze(1)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            
            probs = torch.sigmoid(outputs)
            preds = (probs > 0.5).float()
            
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    accuracy = (correct / total) * 100
    try:
        auroc = roc_auc_score(np.array(all_labels), np.array(all_probs))
        f1 = f1_score(np.array(all_labels), (np.array(all_probs) > 0.5).astype(int))
    except:
        auroc = 0.0
        f1 = 0.0
    
    return total_loss / len(loader), accuracy, f1, auroc

# --- [Main Execution] ---
if __name__ == "__main__":
    # 1. 설정
    BATCH_SIZE = 32
    NUM_EPOCHS = 20
    LEARNING_RATE = 1e-4
    SEED = 8746
    
    # 로그 설정 (Logger 클래스 필요)
    if not os.path.exists("logs"): os.makedirs("logs")
    sys.stdout = Logger("logs/training_log.txt")
    
    seed_torch(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # 2. 데이터
    print("Loading Data...")
    try:
        train_loader, val_loader = get_data_loaders(BATCH_SIZE)
    except Exception as e:
        print(f"데이터 로드 중 에러 발생: {e}")
        print("경로 설정을 확인해주세요.")
        sys.exit(1)

    # 3. 모델 (Effort Model)
    # num_classes=1 (Binary), rank_ratio=0.75
    model = DeepfakeEffortModel(model_name='swin_base_patch4_window12_384', num_classes=1, rank_ratio=0.75)
    model.to(device)

    # 4. 학습 준비
    criterion = nn.BCEWithLogitsLoss()
    
    # Trainable Parameter만 Optimizer에 전달 (효율성)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=LEARNING_RATE, weight_decay=1e-4)
    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)
    scaler = torch.cuda.amp.GradScaler()

    # 5. Loop
    best_f1 = 0.0
    
    for epoch in range(1, NUM_EPOCHS + 1):
        print(f"\nEpoch {epoch}/{NUM_EPOCHS}")
        
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, scaler, device, epoch)
        val_loss, val_acc, val_f1, val_auc = validate(model, val_loader, criterion, device)
        
        print(f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.2f}%")
        print(f"Val   Loss: {val_loss:.4f}, Acc: {val_acc:.2f}%, F1: {val_f1:.4f}, AUC: {val_auc:.4f}")
        
        scheduler.step(val_loss)
        
        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save(model.state_dict(), "best_model_effort.pth")
            print("--> Best Model Saved.")