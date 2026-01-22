import torch
import torch.nn as nn

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import cv2
import os
import glob, sys
import random
import numpy as np

from efficientnet import efficientnetb2_custom

from sklearn.metrics import f1_score
 
from tqdm import tqdm 
import shutil

def seed_torch(seed=8746):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.

seed = 8746
seed_torch(seed=seed)
CROP_SIZE = 512

class Logger(object):
    def __init__(self, filename="training_log.txt"):
        self.terminal = sys.stdout
        self.log = open(filename, "w") # 'w' (덮어쓰기) 또는 'a' (이어쓰기)

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        # 이 flush 메소드는 end='\r' 같은 출력이
        # 실시간으로 반영되도록 보장하는 데 중요합니다.
        self.terminal.flush()
        self.log.flush()

# RandomCropAvoidArea 클래스 대신 아래 클래스를 사용합니다.
class PathAwareRandomCrop(object):
    def __init__(self, output_size, avoid_br_size, avoid_tr_size):
        self.output_size = (output_size, output_size) # (h, w)
        self.avoid_br_size = (avoid_br_size, avoid_br_size) # (h, w)
        self.avoid_tr_size = (avoid_tr_size, avoid_tr_size) # (h, w)
        self.avoid_tl_size = (avoid_tr_size, avoid_tr_size) # (h, w)

    def __call__(self, sample):
        img, filepath = sample 
        
        # if 'avoid_bottom_right' in filepath:
        if 'gemini' in filepath or 'hailuo' in filepath:
            img = self._perform_br_avoid_crop(img)
        elif 'avoid_top_right' in filepath:
            img = self._perform_tr_avoid_crop(img)
        elif 'avoid_left_top_right_bottom' in filepath:
            img = self._perform_br_tl_avoid_crop(img)
        elif 'center_crop' in filepath:
            img = transforms.CenterCrop(self.output_size)(img)
        else:
            w, h = img.size
            th, tw = self.output_size
            
            if w < tw or h < th:
                img = self._resize_and_centercrop(img)
            else:
                img = transforms.RandomCrop(self.output_size)(img)

        return img

    def _resize_and_centercrop(self, img):
        """크롭 크기보다 이미지가 작을 때 공통 처리"""
        img = transforms.Resize(self.output_size)(img)
        return transforms.CenterCrop(self.output_size)(img)

    def _perform_br_avoid_crop(self, img):
        """기존 로직: 우측 하단 회피"""
        w, h = img.size
        th, tw = self.output_size
        ah, aw = self.avoid_br_size

        if w < tw or h < th:
            return self._resize_and_centercrop(img)

        avoid_x_start = w - aw
        avoid_y_start = h - ah

        for _ in range(10):
            i = random.randint(0, h - th) # top
            j = random.randint(0, w - tw) # left
            
            # 겹치지 않는 조건
            if (j + tw <= avoid_x_start) or (i + th <= avoid_y_start):
                return transforms.functional.crop(img, i, j, th, tw)
        
        return transforms.RandomCrop(self.output_size)(img) # 10번 실패 시
    
    def _perform_tr_avoid_crop(self, img):
        """새 로직: 우측 상단 회피"""
        w, h = img.size
        th, tw = self.output_size
        ah, aw = self.avoid_tr_size

        if w < tw or h < th:
            return self._resize_and_centercrop(img)

        avoid_x_start = w - aw
        avoid_y_start = 0 # Top

        for _ in range(10):
            i = random.randint(0, h - th) # top
            j = random.randint(0, w - tw) # left

            if (j + tw <= avoid_x_start) or (i >= (avoid_y_start + ah)):
                return transforms.functional.crop(img, i, j, th, tw)

        return transforms.RandomCrop(self.output_size)(img) # 10번 실패 시
    
    def _perform_br_tl_avoid_crop(self, img):
            """추가된 로직: 우측 하단(BR) 및 좌측 상단(TL) 동시 회피"""
            w, h = img.size
            th, tw = self.output_size
            br_ah, br_aw = self.avoid_br_size
            tl_ah, tl_aw = self.avoid_tl_size

            if w < tw or h < th:
                return self._resize_and_centercrop(img)

            # BR 회피 영역 (시작 x, y)
            br_avoid_x_start = w - br_aw
            br_avoid_y_start = h - br_ah
            
            # TL 회피 영역 (끝 x, y)
            tl_avoid_x_end = tl_aw
            tl_avoid_y_end = tl_ah

            for _ in range(10):
                i = random.randint(0, h - th) # top
                j = random.randint(0, w - tw) # left

                # 1. BR과 겹치는지 검사
                overlaps_with_br = (j + tw > br_avoid_x_start) and (i + th > br_avoid_y_start)
                
                # 2. TL과 겹치는지 검사
                overlaps_with_tl = (j < tl_avoid_x_end) and (i < tl_avoid_y_end)

                # 3. 둘 다 겹치지 않는 경우에만 크롭 반환
                if not overlaps_with_br and not overlaps_with_tl:
                    return transforms.functional.crop(img, i, j, th, tw)

            return transforms.RandomCrop(self.output_size)(img) # 10번 실패 시

# --- 2. 커스텀 Dataset ---
class MixedContentDataset(Dataset):
    def __init__(self, class1_files, class0_files, crop_transform, transform):
        self.crop_transform = crop_transform
        self.transform = transform
        self.samples = []
        
        for f in class1_files:
            self.samples.append((f, 1))
        for f in class0_files:
            self.samples.append((f, 0))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        filepath, label = self.samples[idx]

        # if filepath.endswith(('.mp4', '.avi', '.mkv')):
        #     cap = cv2.VideoCapture(filepath)
        #     frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
        #     if frame_count > 0:
        #         random_idx = random.randint(0, frame_count - 1)
        #         cap.set(cv2.CAP_PROP_POS_FRAMES, random_idx)
            
        #     ret, frame = cap.read()
        #     cap.release()
            
        #     if not ret:
        #         raise Exception(f"Failed to read frame at {random_idx}: {filepath}")
            
        #     img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        #     img = Image.fromarray(img)
            
        # else:
        img = Image.open(filepath).convert('RGB')
        
        w, h = img.size
        if w % 2 == 1: w += 1
        if h % 2 == 1: h += 1
            
        img = img.resize((w, h))
        # img = self.crop_transform((img, filepath))
        img = self.transform(img)
        
        return img, torch.tensor(label, dtype=torch.float32), filepath

# --- [추가] 커스텀 Batch Sampler ---
class BalancedBatchSampler(torch.utils.data.Sampler):
    def __init__(self, dataset, batch_size):
        self.dataset = dataset
        self.batch_size = batch_size
        
        # 데이터셋에서 레이블 정보 추출
        # MixedContentDataset.samples는 (filepath, label) 튜플 리스트라고 가정
        self.indices_c0 = [i for i, (_, label) in enumerate(dataset.samples) if label == 0]
        self.indices_c1 = [i for i, (_, label) in enumerate(dataset.samples) if label == 1]
        
        # 배치 사이즈는 반드시 짝수여야 함
        assert batch_size % 2 == 0, "Batch size must be even for 50:50 split."
        
        # 한 에폭당 배치의 수 결정 (데이터가 많은 클래스 기준으로 설정하여 오버샘플링 효과)
        # 만약 적은 클래스 기준으로 하려면 min()을 사용하세요.
        self.max_len = max(len(self.indices_c0), len(self.indices_c1))
        self.n_batches = self.max_len // (self.batch_size // 2)
        
    def __iter__(self):
        # 매 에폭마다 인덱스 섞기
        idx0 = np.array(self.indices_c0)
        idx1 = np.array(self.indices_c1)
        np.random.shuffle(idx0)
        np.random.shuffle(idx1)
        
        # 반반씩 가져오기 위해 필요한 총 개수
        target_len = self.n_batches * (self.batch_size // 2)
        
        # 데이터가 부족한 쪽은 반복(Cycle/Resize)해서 채움
        idx0 = np.resize(idx0, target_len)
        idx1 = np.resize(idx1, target_len)
        
        for i in range(self.n_batches):
            start = i * (self.batch_size // 2)
            end = start + (self.batch_size // 2)
            
            # 0번 클래스 절반 + 1번 클래스 절반
            batch_indices = np.concatenate([idx0[start:end], idx1[start:end]])
            
            # 배치 내부에서 순서 섞기 (선택 사항, 모델 학습에 더 좋음)
            np.random.shuffle(batch_indices)
            
            yield batch_indices.astype(int)

    def __len__(self):
        return self.n_batches

# --- 3. 데이터 준비 및 분할 ---
def get_data_loaders(batch_size):
    # 이미지/비디오 확장자 필터
    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', 'webp', 'jfif')
    
    # --- 클래스 1 파일 로드 ---
    path_class1 = 'train/fake'
    all_files_c1 = glob.glob(os.path.join(path_class1, '**', '*'), recursive=True)
    all_files_c1 = [f for f in all_files_c1 if f.lower().endswith(valid_extensions)]
    
    # --- 클래스 0 파일 로드 ---
    path_class0_list = []
    path_class0_list.append(os.path.expanduser('~/.cache/kagglehub/datasets/sautkin/imagenet1k1/versions/2'))
    path_class0_list.append('train/real/videezy')
    path_class0_list.append('train/real/youtube')
    
    all_files_c0 = []
    val_c0 = []
    
    # ... (기존 파일 로딩 로직 유지) ...
    for i in path_class0_list:
        temp_files = glob.glob(os.path.join(i, '**', '*'), recursive=True)
        valid_files = [f for f in temp_files if f.lower().endswith(valid_extensions)]
        sample_count = min(1000, len(valid_files))
        sampled_files = random.sample(valid_files, sample_count)
        all_files_c0 += sampled_files
        if len(valid_files) < 1000:
            val_c0 += sampled_files
            
    print(f"학습 클래스 1 파일 수: {len(all_files_c1)}")
    print(f"학습 클래스 0 파일 수: {len(all_files_c0)}")
            
    val_files_c1 = all_files_c1 # (참고: 검증셋 로직은 사용자 의도에 맞게 유지)
    val_files_c0 = val_c0

    crop_transform = PathAwareRandomCrop(output_size=200, avoid_br_size=200, avoid_tr_size=200)

    transform = transforms.Compose([
        transforms.CenterCrop(CROP_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_dataset = MixedContentDataset(all_files_c1, all_files_c0, crop_transform, transform)
    val_dataset = MixedContentDataset(val_files_c1, val_files_c0, crop_transform, transform)
    
    # [핵심 변경] BalancedBatchSampler 생성
    train_batch_sampler = BalancedBatchSampler(train_dataset, batch_size=batch_size)
    
    train_loader = DataLoader(
        train_dataset,
        # batch_size와 shuffle은 batch_sampler와 함께 쓸 수 없으므로 제거
        batch_sampler=train_batch_sampler, 
        num_workers=8,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=4
    )
    
    # Validation은 굳이 5:5일 필요 없으므로 기존 방식 유지 (평가 목적)
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
        persistent_workers=True,
    )
    
    return train_loader, val_loader
def validate(model, loader, criterion, device, save_dir="wrong_samples"):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    correct_c0 = 0
    total_c0 = 0
    correct_c1 = 0
    total_c1 = 0
    all_labels = []
    all_preds = []

    # 오답 저장 폴더 생성
    # if not os.path.exists(save_dir):
    #     os.makedirs(save_dir)
    #     print(f"Created directory for wrong samples: {save_dir}")
    
    with torch.no_grad():
        for images, labels, names in loader:
            images = images.to(device)
            labels = labels.to(device).unsqueeze(1)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            
            probs = torch.sigmoid(outputs)
            preds = (probs > 0.5).float()
            
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            # 오답 처리 및 저장
            # for i in range(labels.size(0)): 
            #     is_correct = (preds[i] == labels[i]).item()
                # if not is_correct:
                #     pred_score = probs[i].item()
                #     true_label = labels[i].item()
                #     full_path = names[i]
                #     filename = os.path.basename(full_path)
                    
                    # 예: Pred_0.91_Label_0_image.jpg
                    # save_name = f"Pred_{pred_score:.4f}_Label_{int(true_label)}_{filename}"
                    # save_path = os.path.join(save_dir, save_name)
                    
                    # try:
                    #     shutil.copy(full_path, save_path)
                    #     print(f"  [WRONG - SAVED] {save_name}")
                    # except Exception as e:
                    #     print(f"  [WRONG - SAVE FAILED] {filename}: {e}")

            # 클래스별 정확도
            c0_mask = (labels == 0)
            total_c0 += c0_mask.sum().item()
            correct_c0 += ((preds == 0) & c0_mask).sum().item()

            c1_mask = (labels == 1)
            total_c1 += c1_mask.sum().item()
            correct_c1 += ((preds == 1) & c1_mask).sum().item()

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

    avg_loss = total_loss / len(loader)
    accuracy = (correct / total) * 100
    acc_c0 = (correct_c0 / total_c0) * 100 if total_c0 > 0 else 0
    acc_c1 = (correct_c1 / total_c1) * 100 if total_c1 > 0 else 0
    
    macro_f1 = f1_score(
        np.array(all_labels).squeeze(),
        np.array(all_preds).squeeze(),
        average='macro', 
        zero_division=0
    )
    
    print(f"  Val Avg Loss: {avg_loss:.4f}, Val Acc: {accuracy:.2f}%, C0: {acc_c0:.2f}%, C1: {acc_c1:.2f}%, F1: {macro_f1:.4f}")
    
    return avg_loss, accuracy, macro_f1, acc_c0, acc_c1


# --- [추가됨] 4. 학습 함수 (AMP 적용) ---
def train_one_epoch(model, loader, criterion, optimizer, scaler, device, epoch):
    model.train()
    
    running_loss = 0.0
    correct = 0
    total = 0
    
    # Progress Bar 설정
    loop = tqdm(loader, desc=f"Epoch [{epoch}] Train", leave=True)
    
    for images, labels, _ in loop: # 파일 경로는 학습 때 필요 없으므로 _ 처리
        images = images.to(device)
        labels = labels.to(device).unsqueeze(1) # (B) -> (B, 1)

        # 1. Forward (Mixed Precision)
        with torch.amp.autocast('cuda'):
            outputs = model(images)
            loss = criterion(outputs, labels)

        # 2. Backward & Optimize
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # 3. 통계 계산
        running_loss += loss.item()
        
        probs = torch.sigmoid(outputs)
        preds = (probs > 0.5).float()
        
        correct += (preds == labels).sum().item()
        total += labels.size(0)

        # Tqdm 업데이트
        loop.set_postfix(loss=loss.item(), acc=(correct/total)*100)

    avg_loss = running_loss / len(loader)
    accuracy = (correct / total) * 100
    
    return avg_loss, accuracy

# --- [수정됨] 5. 메인 실행 블록 ---
if __name__ == "__main__":
    # 1. 하이퍼파라미터 설정
    BATCH_SIZE = 32 # 메모리 상황에 따라 조절 (64 -> 32 권장, AMP 사용 시 64도 가능할 수 있음)
    NUM_EPOCHS = 50
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-4
    SEED = 8746
    
    # 로그 파일 설정
    sys.stdout = Logger("training_log_new.txt")
    
    seed_torch(SEED)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Current Device: {device}")
    
    # 2. 데이터 로드
    print("\n[Loading Data...]")
    # get_data_loaders 함수 내부의 BATCH_SIZE 변수를 전역 변수나 인자로 받도록 수정하는 것이 좋으나,
    # 현재 코드 구조상 get_data_loaders 내부에서 BATCH_SIZE를 참조하므로 
    # 위에서 정의한 BATCH_SIZE가 get_data_loaders 호출 시 반영되도록 주의해주세요.
    # (제공해주신 코드의 get_data_loaders는 전역 변수 BATCH_SIZE를 참조합니다.)
    train_loader, val_loader = get_data_loaders(BATCH_SIZE)
    print("Data loading complete.")

    # 3. 모델 초기화
    print("\n[Initializing Model...]")
    model = efficientnetb2_custom()
    
    model.to(device)

    # 4. Optimizer, Scheduler, Loss, Scaler 정의
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    
    # 학습률 스케줄러 (Validation Loss가 개선되지 않으면 LR 감소)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3
    )
    
    # Mixed Precision Scaler
    scaler = torch.amp.GradScaler('cuda')

    # 5. Training Loop
    best_f1 = 0.0
    best_loss = float('inf')
    
    print(f"\n[Start Training] Epochs: {NUM_EPOCHS}, Batch: {BATCH_SIZE}, LR: {LEARNING_RATE}")
    print("-" * 60)

    for epoch in range(1, NUM_EPOCHS + 1):
        # --- Train ---
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler, device, epoch
        )
        
        print(f"Epoch [{epoch}/{NUM_EPOCHS}] Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")

        # --- Validate ---
        # validate 함수가 오답 이미지를 저장하므로, 학습 중에는 save_dir를 매번 덮어쓰거나
        # epoch 별로 분리하는 것이 좋습니다. 여기서는 epoch 별 폴더 생성은 하지 않고 기본으로 둡니다.
        print(f"Validating Epoch [{epoch}]...")
        val_loss, val_acc, val_f1, val_c0, val_c1 = validate(
            model, val_loader, criterion, device, save_dir=f"wrong_samples_epoch_{epoch}"
        )
        
        # --- Scheduling ---
        scheduler.step(val_loss)
        
        # --- Save Model ---
        # 1) Best F1 Score 기준 저장
        if val_f1 > best_f1:
            print(f"--> Best Model Updated (F1: {best_f1:.4f} -> {val_f1:.4f})")
            best_f1 = val_f1
            torch.save(model.state_dict(), "best_model_f1.pth")
            
        # 2) Best Loss 기준 저장 (선택 사항)
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), "best_model_loss.pth")

        # 3) Last Epoch 저장
        torch.save(model.state_dict(), "last_model.pth")
        
        print("-" * 60)

    print("Training Finished.")
    # BATCH_SIZE = 64
    # criterion = nn.BCEWithLogitsLoss() 
    
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # print(f"Using device: {device}")

    # print("Loading data...")
    # train_loader, val_loader = get_data_loaders(BATCH_SIZE)
    # print("Data loading complete.")

    # print("Loading model...")

    # model = efficientnetb2_custom()

    # model_path = 'model_epoch_1.pth'
    # model.load_state_dict(torch.load(model_path, map_location='cpu'), strict=True)
    # model.to(device)
    

    # val_loss, val_acc, val_f1, val_c0, val_c1 = validate(model, val_loader, criterion, device)