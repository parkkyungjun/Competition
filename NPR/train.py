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
from sklearn.model_selection import train_test_split

from efficientnet import efficientnetb2_custom

from sklearn.metrics import f1_score
 
from tqdm import tqdm 
import shutil

def seed_torch(seed=1029):
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
        
        if 'avoid_bottom_right' in filepath:
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
    def __init__(self, class1_files, class0_files, transform_c1, transform_c0):
        self.transform_c1 = transform_c1
        self.transform_c0 = transform_c0
        self.samples = []
        
        for f in class1_files:
            self.samples.append((f, 1))
        for f in class0_files:
            self.samples.append((f, 0))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        filepath, label = self.samples[idx]

        if filepath.endswith(('.mp4', '.avi')):
            cap = cv2.VideoCapture(filepath)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            mid = frame_count // 2
            cap.set(cv2.CAP_PROP_POS_FRAMES, mid)
            
            ret, frame = cap.read()
            cap.release()
            
            if not ret:
                # 읽기 실패 시 예외 처리보다 빈 이미지 등을 반환하거나 로깅하는 게 안전할 수 있음
                raise Exception(f"Failed to read frame: {filepath}")
            
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(img)
            
        else:
            img = Image.open(filepath).convert('RGB')
        
        w, h = img.size
        if w % 2 == 1: w += 1
        if h % 2 == 1: h += 1
            
        img = img.resize((w, h))
        if label == 1:
            img = self.transform_c1(img)
        else:
            img = self.transform_c0(img)
        
        # [변경됨] 파일 복사를 위해 basename 대신 전체 filepath 반환
        return img, torch.tensor(label, dtype=torch.float32), filepath


# --- 3. 데이터 준비 및 분할 ---
def get_data_loaders(batch_size):
    # 이미지/비디오 확장자 필터
    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.mp4', '.avi', 'webp')
    
    # --- 클래스 1 파일 로드 ---
    path_class1 = 'gen_dataset'
    all_files_c1 = glob.glob(os.path.join(path_class1, '**', '*'), recursive=True)
    all_files_c1 = [f for f in all_files_c1 if f.lower().endswith(valid_extensions)]
    
    # --- 클래스 0 파일 로드 ---
    path_class0_list = []
    path_class0_list.append(os.path.expanduser('~/.cache/kagglehub/datasets/sautkin/imagenet1k1/versions/2'))
    # path_class0_list.append(os.path.expanduser('~/.cache/kagglehub/datasets/subrahmanya090/face-images-high-quality-scraped-from-unsplash/versions/1'))
    
    all_files_c0 = []
    
    for i in path_class0_list:
        all_files_c0_file = glob.glob(os.path.join(i, '**', '*'), recursive=True)
        all_files_c0 += [f for f in all_files_c0_file if f.lower().endswith(valid_extensions) and not f.lower().endswith('.mp4')] # 혹시 모르니 mp4 제외

    all_files_c0_sampling = random.sample(all_files_c0, 8000) # 너무 많으면 5k로 제한    
        
    # print(f"발견된 클래스 1 파일 수: {len(all_files_c1)}")
    # print(f"발견된 클래스 0 파일 수: {len(all_files_c0_sampling)}")

    # print(set(sorted([os.path.basename(i).split('_')[0] for i in all_files_c1])))

    real = "./real"
    real = glob.glob(real+'/**', recursive=True)
    real = [f for f in real if f.lower().endswith(valid_extensions)]
    
    # fake = "./fake2"
    # fake = glob.glob(fake+'/**', recursive=True)
    # fake = [f for f in fake if f.lower().endswith(valid_extensions)]
    
    # fake = "/home/pkj/.cache/kagglehub/datasets/ayushmandatta1/deepdetect-2025/versions/1/ddata/train/fake"
    # fake = glob.glob(fake+'/**', recursive=True)
    # fake = [f for f in fake if f.lower().endswith(valid_extensions)]
    
    # unsplash = "/home/pkj/.cache/kagglehub/datasets/subrahmanya090/face-images-high-quality-scraped-from-unsplash/versions/1/face_images_scraped_from_unsplash_high_quality_with_noise"
    # unsplash = glob.glob(unsplash+'/**', recursive=True) /home/pkj/.cache/kagglehub/datasets/ayushmandatta1/deepdetect-2025/versions/1
    # unsplash = [f for f in unsplash if f.lower().endswith(valid_extensions)]

    # coco = "/home/pkj/.cache/kagglehub/datasets/awsaf49/coco-2017-dataset/versions/2/coco2017"
    # coco = glob.glob(coco+'/**', recursive=True)
    # coco = [f for f in coco if f.lower().endswith(valid_extensions)]

    # deepdetect = "/home/pkj/.cache/kagglehub/datasets/ayushmandatta1/deepdetect-2025/versions/1/ddata"
    # deepdetect = glob.glob(deepdetect+'/**', recursive=True)
    # deepdetect = [f for f in deepdetect if (f.lower().endswith(valid_extensions) and f not in 'real')]
    
    # temp = "gen_dataset/avoid_bottom_right_cropped/Hailuo"
    # temp = glob.glob(temp+'/**', recursive=True)
    # temp = [f for f in temp if f.lower().endswith(valid_extensions)]
    
    # real = "/home/pkj/.cache/kagglehub/datasets/malaxmels/ai-detection-dataset/versions/1/dataset/real"
    # real = glob.glob(real+'/**', recursive=True)
    # real = [f for f in real if f.lower().endswith(valid_extensions)]
    
    # ai = "/home/pkj/.cache/kagglehub/datasets/malaxmels/ai-detection-dataset/versions/1/dataset/ai"
    # ai = glob.glob(ai+'/**', recursive=True)
    # ai = [f for f in ai if f.lower().endswith(valid_extensions)]
    
    # forensynths_real = "/home/pkj/.cache/kagglehub/datasets/chandlertimm/forensynths/versions/4/Wang_CVPR2020/training/person/0_real"
    # forensynths_real = glob.glob(forensynths_real+'/**', recursive=True)
    # forensynths_real = [f for f in forensynths_real if f.lower().endswith(valid_extensions)]
    # random.shuffle(forensynths_real)
    
    # forensynths_fake = "/home/pkj/.cache/kagglehub/datasets/chandlertimm/forensynths/versions/4/Wang_CVPR2020/training/person/1_fake"
    # forensynths_fake = glob.glob(forensynths_fake+'/**', recursive=True)
    # forensynths_fake = [f for f in forensynths_fake if f.lower().endswith(valid_extensions)]
    # random.shuffle(forensynths_fake)
    train_files_c0 = []
    for filepath in tqdm(all_files_c0, desc="이미지 처리 중"):
        with Image.open(filepath) as img:
            w, h = img.size
            if w < 256 or h < 256:
                continue
            train_files_c0.append(filepath) # (width, height) 튜플
            
    val_files_c1 = [] # all_files_c1
    val_files_c0 = train_files_c0 # unsplash[:val_size] + all_files_c0_sampling[:val_size]

    print(f"검증셋: 클래스1 {len(val_files_c1)}, 클래스0 {len(val_files_c0)}")

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    transform_val = transforms.Compose([
        # transforms.CenterCrop((CROP_SIZE, CROP_SIZE)),
        transforms.ToTensor(),
        normalize
    ])

    val_dataset = MixedContentDataset(val_files_c1, val_files_c0, transform_val, transform_val) # 검증셋은 일반 크롭
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False, # 검증 시에는 셔플 X
        num_workers=8,
        pin_memory=True
    )
    
    return val_loader

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
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        print(f"Created directory for wrong samples: {save_dir}")
    
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
            for i in range(labels.size(0)): 
                is_correct = (preds[i] == labels[i]).item()
                if not is_correct:
                    pred_score = probs[i].item()
                    true_label = labels[i].item()
                    full_path = names[i]
                    filename = os.path.basename(full_path)
                    
                    # 보기 쉽게 파일명에 예측값_라벨_원본이름 형식으로 저장
                    # 예: Pred_0.91_Label_0_image.jpg
                    save_name = f"Pred_{pred_score:.4f}_Label_{int(true_label)}_{filename}"
                    save_path = os.path.join(save_dir, save_name)
                    
                    try:
                        shutil.copy(full_path, save_path)
                        print(f"  [WRONG - SAVED] {save_name}")
                    except Exception as e:
                        print(f"  [WRONG - SAVE FAILED] {filename}: {e}")

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

if __name__ == "__main__":
    BATCH_SIZE = 64
    criterion = nn.BCEWithLogitsLoss() 
    
    # 장치 설정
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 데이터 로더 준비
    print("Loading data...")
    val_loader = get_data_loaders(BATCH_SIZE)
    print("Data loading complete.")

    # 모델 로드
    print("Loading model...")

    # model = resnet50(num_classes=1)
    model = efficientnetb2_custom()

    model_path = 'model_epoch_1.pth'
    model.load_state_dict(torch.load(model_path, map_location='cpu'), strict=True)
    model.to(device)
    

    val_loss, val_acc, val_f1, val_c0, val_c1 = validate(model, val_loader, criterion, device)