# [코드 기능 요약]
# 1. CLIP 기반 멀티모달(이미지+텍스트) 분류 및 임베딩 학습 시스템
# 2. 주요 기능:
#   - 색상/물건/조합 데이터 분리 학습 및 평가
#   - Focal Loss로 클래스 불균형 완화
#   - FAISS를 통한 고속 텍스트→이미지 검색
#   - t-SNE 기반 3D 임베딩 시각화 (이미지/텍스트 동시)
# 3. 데이터 처리:
#   - 동의어 정규화(색상/물건)
#   - 강화된 이미지 증강
# 4. 학습 전략:
#   - 혼합 정밀도(Mixed Precision) 학습
#   - 조기 종료(Early Stopping), 학습률 스케줄링

import os  
import sys  
import io  
import random  # 랜덤 샘플링 등 사용
import numpy as np  # 수치 연산
from collections import Counter  # 데이터 카운팅
from PIL import Image  # 이미지 처리
import torch  # PyTorch 메인 모듈
import torch.nn as nn  # 신경망 모듈
import torch.nn.functional as F  # 신경망 함수
import torch.optim as optim  # 최적화 함수
from torch.utils.data import Dataset, DataLoader, random_split, WeightedRandomSampler, Subset  # 데이터 관련
from torchvision import transforms  # 이미지 변환
from tqdm import tqdm  # 진행률 표시
import matplotlib.pyplot as plt  # 시각화
from sklearn.manifold import TSNE  # 차원 축소(t-SNE)
from torch.optim.lr_scheduler import ReduceLROnPlateau  # 학습률 스케줄러
import faiss  # 고속 벡터 검색 라이브러리
import clip  # OpenAI CLIP 모델

# 한글 출력 설정 (윈도우 환경 대응)
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # 라이브러리 충돌 방지
sys.stdout = io.TextIOWrapper(sys.stdout.detach(), encoding='utf-8')  # 한글 출력

# ----- 동의어 및 정규화 -----
def build_synonym_map(synonym_groups):
    # 동의어 그룹을 대표값-동의어 맵으로 변환
    class_list = []
    synonym_map = {}
    for group in synonym_groups:
        rep = group[0]  # 대표값
        class_list.append(rep)
        for word in group:
            synonym_map[word] = rep  # 동의어 모두 대표값으로 매핑
    return class_list, synonym_map

# 색상 동의어 그룹 정의
color_synonyms = [
    ["흰색", "화이트", "하얀색"],
    ["검정", "블랙", "검은색", "까만색", "검정색", "다크"],
    ["빨강", "레드", "빨간색"],
    ["파랑", "블루", "파란색", "스카이 블루"],
    ["하늘색", "아이스 블루"],
    ["노랑", "옐로우", "노란색","금색", "골드"],
    ["초록", "그린", "녹색"],
    ["남색", "네이비"],
    ["분홍", "핑크", "연분홍"],
    ["주황", "오렌지", "주황색"],
    ["보라", "퍼플", "바이올렛", "보라색"],
    ["갈색", "브라운", "황토색"],
    ["은색", "실버", "회색", "그레이"],
    ["베이지", "크림"],
    ["청록", "티파니", "청록색"],
    ["자주", "마젠타", "자주색"],
    ["연두", "라임", "연두색", "민트"],
    ["연노랑", "파스텔옐로우"],
    ["연분홍", "파스텔핑크"]
]
color_classes, COLOR_SYNONYM_MAP = build_synonym_map(color_synonyms)  # 색상 클래스 및 맵 생성

# 물건 동의어 그룹 정의
object_synonyms = [
    ["갤럭시", "휴대폰", "스마트폰"],
    ["아이폰"],
    ["가방", "백팩"],
    ["버즈", "갤럭시 버즈"],
    ["겉옷", "자켓","재킷"],
    ["아이패드", "패드"],
    ["맥북"],
    ["카드", "신용카드", "체크카드"],
    ["지갑", "카드지갑"],
    ["모자", "캡", "캡모자"],
    ["마우스", "무선마우스", "노트북 마우스"],
    ["태블릿", "태블렛"],
    ["무선 이어폰", "이어폰"],
    ["샌달", "샌들"],
    ["스마트워치", "갤럭시워치"],
    ["헤드셋", "헤드폰"]
]
object_classes, OBJECT_SYNONYM_MAP = build_synonym_map(object_synonyms)  # 물건 클래스 및 맵 생성

def normalize_color(label):
    # 색상 라벨을 대표값으로 정규화
    return COLOR_SYNONYM_MAP.get(label.strip(), label.strip())

def normalize_object(label):
    # 물건 라벨을 대표값으로 정규화
    return OBJECT_SYNONYM_MAP.get(label.strip(), label.strip())

# ----- Focal Loss -----
class FocalLoss(nn.Module):
    # 클래스 불균형 완화를 위한 Focal Loss 정의
    def __init__(self, gamma=1.5, weight=None, ignore_index=-100):
        super().__init__()
        self.gamma = gamma
        self.weight = weight
        self.ignore_index = ignore_index

    def forward(self, input, target):
        # CrossEntropy 기반 Focal Loss 계산
        ce_loss = F.cross_entropy(input, target, weight=self.weight, 
                                  ignore_index=self.ignore_index, reduction='none')
        pt = torch.exp(-ce_loss)
        return ((1 - pt)**self.gamma * ce_loss).mean()

# ----- 데이터셋 -----
class AugmentedCLIPDataset(Dataset):
    # 이미지와 텍스트를 동시에 다루는 커스텀 데이터셋
    def __init__(self, color_dir, object_dir, combined_dir, preprocess, train=True, oversample_min_count=50):
        # 클래스 리스트 생성
        self.color_classes = sorted(set([normalize_color(label.strip()) for label in os.listdir(color_dir)]))
        self.object_classes = sorted(set([normalize_object(label.strip()) for label in os.listdir(object_dir)]))
        self.combined_classes = sorted(set([
            normalize_color(label.strip().split('_')[0]) + '_' + normalize_object(label.strip().split('_')[1])
            for label in os.listdir(combined_dir)
        ])) if os.path.exists(combined_dir) else []
        self.preprocess = preprocess  # CLIP 전처리 함수
        self.train = train  # 학습/검증 구분

        # 학습용 이미지 증강 파이프라인
        self.train_transform = transforms.Compose([
            transforms.RandomRotation(40),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.3),
            transforms.RandomResizedCrop(224, scale=(0.9, 1.0)),
            transforms.RandomAffine(degrees=20, translate=(0.1, 0.1), scale=(0.8, 1.2)),
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
            preprocess
        ])
        self.val_transform = preprocess  # 검증용 변환(증강 없음)

        self.data = []  # 전체 데이터 저장 리스트
        self._load_data(color_dir, "color")  # 색상 데이터 로드
        self._load_data(object_dir, "object")  # 물건 데이터 로드
        self._load_combined_data(combined_dir)  # 조합 데이터 로드
        if train:
            self._oversample(oversample_min_count)  # 소수 클래스 오버샘플링
        self.class_weights = self._calculate_class_weights()  # 클래스별 가중치 계산

    def _oversample(self, min_count):
        # 소수 클래스 샘플을 min_count까지 증식
        label_list = [item['label'] for item in self.data]
        label_counts = Counter(label_list)
        for label, count in label_counts.items():
            if count < min_count:
                add_count = min_count - count
                samples = [item for item in self.data if item['label'] == label]
                self.data.extend(random.choices(samples, k=add_count))

    def _load_data(self, root_dir, data_type):
        # 색상/물건 데이터 로드
        for label in os.listdir(root_dir):
            norm_label = normalize_color(label.strip()) if data_type == "color" else normalize_object(label.strip())
            label_dir = os.path.join(root_dir, label)
            for img_file in os.listdir(label_dir):
                self.data.append({
                    "path": os.path.join(label_dir, img_file),
                    "type": data_type,
                    "label": norm_label
                })

    def _get_combined_prompts(self, color, obj):
        # 조합 데이터의 텍스트 프롬프트 생성
        prompts = [
            f"{color} {obj}",
            f"분실된 {color} {obj} 사진",
            f"이 사진은 {color} {obj}입니다",
            f"{color} 색상의 {obj}",
            f"{color} {obj}를 찾습니다"
        ]
        return random.choice(prompts)

    def _load_combined_data(self, combined_dir):
        # 색상+물건 조합 데이터 로드
        if not combined_dir:
            return
        for combined_label in os.listdir(combined_dir):
            parts = combined_label.split('_')
            if len(parts) < 2:
                continue
            color = normalize_color(parts[0])
            obj = normalize_object(parts[1])
            label_dir = os.path.join(combined_dir, combined_label)
            for img_file in os.listdir(label_dir):
                self.data.append({
                    "path": os.path.join(label_dir, img_file),
                    "type": "combined",
                    "color": color,
                    "object": obj,
                    "text": self._get_combined_prompts(color, obj),
                    "label": f"{color}_{obj}"
                })

    def _calculate_class_weights(self):
        # 클래스별 샘플 수에 따른 가중치 계산
        class_counts = {}
        for item in self.data:
            label = f"{item['color']}_{item['object']}" if item["type"] == "combined" else item["label"]
            class_counts[label] = class_counts.get(label, 0) + 1
        weights = [
            len(self.data) / (len(class_counts) * class_counts[
                f"{item['color']}_{item['object']}" if item["type"] == "combined" else item["label"]
            ])
            for item in self.data
        ]
        return torch.DoubleTensor(weights)

    def __len__(self):
        # 전체 데이터 개수 반환
        return len(self.data)

    def __getitem__(self, idx):
        # 인덱스에 해당하는 데이터 반환
        item = self.data[idx]
        image = Image.open(item["path"]).convert("RGB")
        transform = self.train_transform if self.train else self.val_transform
        if item["type"] == "combined":
            # 조합 데이터
            return {
                "image": transform(image),
                "text": item["text"],
                "type": "combined",
                "color": normalize_color(item["color"]),
                "object": normalize_object(item["object"]),
                "label": f"{normalize_color(item['color'])}_{normalize_object(item['object'])}"
            }
        else:
            # 색상/물건 단일 데이터
            return {
                "image": transform(image),
                "text": f"{item['label']} 색상" if item["type"] == "color" else f"{item['label']} 물건",
                "type": item["type"],
                "label": item["label"],
                "color": normalize_color(item["label"]) if item["type"] == "color" else "",
                "object": normalize_object(item["label"]) if item["type"] == "object" else ""
            }

# ----- 모델 -----
class EnhancedCLIPModel(nn.Module):
    # CLIP 기반 멀티모달 분류 모델
    def __init__(self, clip_model, color_classes, object_classes, combined_classes):
        super().__init__()
        self.clip = clip_model  # CLIP 백본
        # 이미지 임베딩 어댑터
        self.image_adapter = nn.Sequential(
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Linear(256, 512)
        )
        # 텍스트 임베딩 어댑터
        self.text_adapter = nn.Sequential(
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Linear(256, 512)
        )
        # 색상 분류 헤드
        self.color_head = nn.Linear(512, len(color_classes))
        # 물건 분류 헤드
        self.object_head = nn.Linear(512, len(object_classes))
        # 조합 분류 헤드
        self.combined_head = nn.Linear(512, len(combined_classes))
        # CLIP의 logit scaling 파라미터
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1/0.07))

    def forward(self, images, texts):
        # 이미지/텍스트 임베딩 추출 및 어댑터 적용
        image_features = self.clip.encode_image(images)
        text_features = self.clip.encode_text(texts)
        adapted_img = image_features + 0.05 * self.image_adapter(image_features)
        adapted_txt = text_features + 0.05 * self.text_adapter(text_features)
        adapted_img = adapted_img / adapted_img.norm(dim=-1, keepdim=True)
        adapted_txt = adapted_txt / adapted_txt.norm(dim=-1, keepdim=True)
        return (
            self.color_head(adapted_img),  # 색상 분류 결과
            self.object_head(adapted_img),  # 물건 분류 결과
            self.combined_head(adapted_img), # 조합 분류 결과
            adapted_img,  # 이미지 임베딩
            adapted_txt   # 텍스트 임베딩
        )

# ----- Trainer -----
class CLIPTrainer:
    # 전체 학습/평가/시각화/검색 파이프라인 관리 클래스
    def __init__(self, color_dir, object_dir, combined_dir, device, batch_size, sample_ratio, patience):
        self.device = device  # 학습 디바이스
        self.clip_model, preprocess = clip.load("ViT-B/32", device=device, jit=False)  # CLIP 모델 로드
        self.clip_model.float()
        # 데이터셋 생성
        self.dataset = AugmentedCLIPDataset(color_dir, object_dir, combined_dir, preprocess, train=True, oversample_min_count=50)
        if sample_ratio < 1.0:
            # 데이터 일부만 샘플링
            n = int(len(self.dataset) * sample_ratio)
            indices = torch.randperm(len(self.dataset))[:n]
            self.dataset.data = [self.dataset.data[i] for i in indices]
            self.dataset.class_weights = self.dataset._calculate_class_weights()
        # 학습/검증 데이터 분할
        train_size = int(0.9 * len(self.dataset))
        self.train_dataset, self.val_dataset = random_split(self.dataset, [train_size, len(self.dataset)-train_size])
        train_weights = self.dataset.class_weights[self.train_dataset.indices]
        sampler = WeightedRandomSampler(train_weights, len(train_weights))

        # 전체 데이터로부터 DataLoader 생성
        self.train_loader = DataLoader(self.train_dataset, batch_size=batch_size, sampler=sampler, num_workers=4, pin_memory=False)
        self.val_loader = DataLoader(self.val_dataset, batch_size=batch_size, num_workers=4, pin_memory=False)

        # 타입별 데이터 분리 함수
        def split_by_type(dataset, type_name):
            indices = [i for i, item in enumerate(dataset) if item['type'] == type_name]
            return Subset(dataset, indices)

        # 타입별 DataLoader 생성
        self.train_color_loader = DataLoader(split_by_type(self.train_dataset, "color"), batch_size=batch_size, shuffle=True, num_workers=2)
        self.train_object_loader = DataLoader(split_by_type(self.train_dataset, "object"), batch_size=batch_size, shuffle=True, num_workers=2)
        self.train_combined_loader = DataLoader(split_by_type(self.train_dataset, "combined"), batch_size=batch_size, shuffle=True, num_workers=2)
        self.val_color_loader = DataLoader(split_by_type(self.val_dataset, "color"), batch_size=batch_size, shuffle=False, num_workers=2)
        self.val_object_loader = DataLoader(split_by_type(self.val_dataset, "object"), batch_size=batch_size, shuffle=False, num_workers=2)
        self.val_combined_loader = DataLoader(split_by_type(self.val_dataset, "combined"), batch_size=batch_size, shuffle=False, num_workers=2)

        # 모델, 옵티마이저, 스케줄러, 손실함수, 기타 설정
        self.model = EnhancedCLIPModel(
            self.clip_model,
            self.dataset.color_classes,
            self.dataset.object_classes,
            self.dataset.combined_classes  # combined 클래스 전달
        ).to(device)
        self.optimizer = optim.AdamW(self.model.parameters(), lr=1.5e-5, weight_decay=0.001)
        self.scheduler = ReduceLROnPlateau(self.optimizer, mode='min', factor=0.5, patience=2)
        self.criterion = FocalLoss(ignore_index=self.model.color_head.out_features - 1)
        self.scaler = torch.amp.GradScaler() if self.device == "cuda" else None
        self.patience = patience  # 조기 종료 patience

    def train_epoch(self, loader, use_combined=True, combined_loss_weight=0.5):
        # 한 에폭 학습
        self.model.train()
        total_loss, total_correct, total_count = 0, 0, 0
        for batch in tqdm(loader, desc="Train", leave=False):
                images = batch["image"].to(self.device, non_blocking=True)
                texts = batch["text"]
                text_tokens = clip.tokenize(texts).to(self.device, non_blocking=True)
                color_labels, object_labels, combined_labels = [], [], []
                for i in range(len(batch["type"])):
                    if batch["type"][i] == "combined":
                        if use_combined:
                            color_labels.append(self.model.color_head.out_features - 1)
                            object_labels.append(self.model.object_head.out_features - 1)
                            combined_labels.append(self.dataset.combined_classes.index(batch["label"][i]))
                        else:
                            # combined 데이터 무시 (loss/acc 계산에서 제외)
                            color_labels.append(self.model.color_head.out_features - 1)
                            object_labels.append(self.model.object_head.out_features - 1)
                            combined_labels.append(self.model.combined_head.out_features - 1)
                    elif batch["type"][i] == "color":
                        color_labels.append(self.dataset.color_classes.index(batch["label"][i]))
                        object_labels.append(self.model.object_head.out_features - 1)
                        combined_labels.append(self.model.combined_head.out_features - 1)
                    else:
                        color_labels.append(self.model.color_head.out_features - 1)
                        object_labels.append(self.dataset.object_classes.index(batch["label"][i]))
                        combined_labels.append(self.model.combined_head.out_features - 1)
                color_labels = torch.tensor(color_labels, device=self.device)
                object_labels = torch.tensor(object_labels, device=self.device)
                combined_labels = torch.tensor(combined_labels, device=self.device)

                if self.device == "cuda":
                    with torch.amp.autocast(self.device):
                        color_logits, object_logits, combined_logits, img_emb, txt_emb = self.model(images, text_tokens)
                        color_loss = self.criterion(color_logits, color_labels)
                        object_loss = self.criterion(object_logits, object_labels)
                        combined_loss = self.criterion(combined_logits, combined_labels)
                        logit_scale = self.model.logit_scale.exp()
                        logits_per_image = logit_scale * img_emb @ txt_emb.T
                        logits_per_text = logits_per_image.T
                        ground_truth = torch.arange(len(images), device=self.device)
                        contrastive_loss = (self.criterion(logits_per_image, ground_truth) + self.criterion(logits_per_text, ground_truth)) / 2
                        # combined_loss_weight로 combined loss 비중 조정
                        loss = 0.5 * (color_loss + object_loss) + combined_loss_weight * combined_loss + (1 - 0.5 - combined_loss_weight) * contrastive_loss

                        self.optimizer.zero_grad()
                        self.scaler.scale(loss).backward()
                        self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                else:
                    color_logits, object_logits, combined_logits, img_emb, txt_emb = self.model(images, text_tokens)
                    color_loss = self.criterion(color_logits, color_labels)
                    object_loss = self.criterion(object_logits, object_labels)
                    combined_loss = self.criterion(combined_logits, combined_labels)
                    logit_scale = self.model.logit_scale.exp()
                    logits_per_image = logit_scale * img_emb @ txt_emb.T
                    logits_per_text = logits_per_image.T
                    ground_truth = torch.arange(len(images), device=self.device)
                    contrastive_loss = (self.criterion(logits_per_image, ground_truth) + self.criterion(logits_per_text, ground_truth)) / 2
                    loss = 0.1 * color_loss + 0.3 * object_loss + combined_loss_weight * combined_loss + 0.2 * contrastive_loss

                    self.optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.optimizer.step()
            

                color_pred = color_logits.argmax(dim=1)
                object_pred = object_logits.argmax(dim=1)
                combined_pred = combined_logits.argmax(dim=1)
                color_mask = color_labels != (self.model.color_head.out_features - 1)
                object_mask = object_labels != (self.model.object_head.out_features - 1)
                combined_mask = combined_labels != (self.model.combined_head.out_features - 1)
                total_correct += (color_pred[color_mask] == color_labels[color_mask]).sum().item()
                total_correct += (object_pred[object_mask] == object_labels[object_mask]).sum().item()
                total_correct += (combined_pred[combined_mask] == combined_labels[combined_mask]).sum().item()
                total_count += color_mask.sum().item() + object_mask.sum().item() + combined_mask.sum().item()
                total_loss += loss.item()
        avg_loss = total_loss / len(loader)
        avg_acc = total_correct / total_count if total_count > 0 else 0
        return avg_loss, avg_acc

    def validate_epoch(self, loader, use_combined=True, combined_loss_weight=0.5):
        self.model.eval()
        total_loss, total_correct, total_count = 0, 0, 0
        with torch.no_grad():
            for batch in tqdm(loader, desc="Val", leave=False):
                images = batch["image"].to(self.device, non_blocking=True)
                texts = batch["text"]
                text_tokens = clip.tokenize(texts).to(self.device, non_blocking=True)
                color_labels, object_labels, combined_labels = [], [], []
                for i in range(len(batch["type"])):
                    if batch["type"][i] == "combined":
                        if use_combined:
                            color_labels.append(self.model.color_head.out_features - 1)
                            object_labels.append(self.model.object_head.out_features - 1)
                            combined_labels.append(self.dataset.combined_classes.index(batch["label"][i]))
                        else:
                            color_labels.append(self.model.color_head.out_features - 1)
                            object_labels.append(self.model.object_head.out_features - 1)
                            combined_labels.append(self.model.combined_head.out_features - 1)
                    elif batch["type"][i] == "color":
                        color_labels.append(self.dataset.color_classes.index(batch["label"][i]))
                        object_labels.append(self.model.object_head.out_features - 1)
                        combined_labels.append(self.model.combined_head.out_features - 1)
                    else:
                        color_labels.append(self.model.color_head.out_features - 1)
                        object_labels.append(self.dataset.object_classes.index(batch["label"][i]))
                        combined_labels.append(self.model.combined_head.out_features - 1)
                color_labels = torch.tensor(color_labels, device=self.device)
                object_labels = torch.tensor(object_labels, device=self.device)
                combined_labels = torch.tensor(combined_labels, device=self.device)

                color_logits, object_logits, combined_logits, img_emb, txt_emb = self.model(images, text_tokens)
                color_loss = self.criterion(color_logits, color_labels)
                object_loss = self.criterion(object_logits, object_labels)
                combined_loss = self.criterion(combined_logits, combined_labels)
                logit_scale = self.model.logit_scale.exp()
                logits_per_image = logit_scale * img_emb @ txt_emb.T
                logits_per_text = logits_per_image.T
                ground_truth = torch.arange(len(images), device=self.device)
                contrastive_loss = (self.criterion(logits_per_image, ground_truth) + self.criterion(logits_per_text, ground_truth)) / 2
                loss = 0.1 * color_loss + 0.3 * object_loss + combined_loss_weight * combined_loss + 0.2 * contrastive_loss

                color_pred = color_logits.argmax(dim=1)
                object_pred = object_logits.argmax(dim=1)
                combined_pred = combined_logits.argmax(dim=1)
                color_mask = color_labels != (self.model.color_head.out_features - 1)
                object_mask = object_labels != (self.model.object_head.out_features - 1)
                combined_mask = combined_labels != (self.model.combined_head.out_features - 1)
                total_correct += (color_pred[color_mask] == color_labels[color_mask]).sum().item()
                total_correct += (object_pred[object_mask] == object_labels[object_mask]).sum().item()
                total_correct += (combined_pred[combined_mask] == combined_labels[combined_mask]).sum().item()
                total_count += color_mask.sum().item() + object_mask.sum().item() + combined_mask.sum().item()
                total_loss += loss.item()
        avg_loss = total_loss / len(loader)
        avg_acc = total_correct / total_count if total_count > 0 else 0
        return avg_loss, avg_acc

    def train(self, epochs, warmup_epochs=12, combined_loss_weight=0.5):
        # warmup_epochs: color/object만 학습하는 epoch 수
        best_loss = float('inf')
        patience_counter = 0
        for epoch in range(epochs):
            # warmup 단계: combined 데이터 제외
            if epoch < warmup_epochs:
                use_combined = False
                comb_weight = 0.0
            else:
                use_combined = True
                comb_weight = combined_loss_weight

            train_results = {}
            val_results = {}
            for name, train_loader, val_loader in [
                ("color", self.train_color_loader, self.val_color_loader),
                ("object", self.train_object_loader, self.val_object_loader),
                ("combined", self.train_combined_loader, self.val_combined_loader)
            ]:
                train_loss, train_acc = self.train_epoch(
                    train_loader,
                    use_combined=(use_combined if name == "combined" else True),
                    combined_loss_weight=(comb_weight if name == "combined" else 0.2)
                )
                val_loss, val_acc = self.validate_epoch(
                    val_loader,
                    use_combined=(use_combined if name == "combined" else True)
                )
                train_results[name] = (train_loss, train_acc)
                val_results[name] = (val_loss, val_acc)
            print(
                f"Epoch {epoch+1}/{epochs} | "
                f"color: TLoss={train_results['color'][0]:.4f}, TAcc={train_results['color'][1]:.4f}, VLoss={val_results['color'][0]:.4f}, VAcc={val_results['color'][1]:.4f} | "
                f"object: TLoss={train_results['object'][0]:.4f}, TAcc={train_results['object'][1]:.4f}, VLoss={val_results['object'][0]:.4f}, VAcc={val_results['object'][1]:.4f} | "
                f"combined: TLoss={train_results['combined'][0]:.4f}, TAcc={train_results['combined'][1]:.4f}, VLoss={val_results['combined'][0]:.4f}, VAcc={val_results['combined'][1]:.4f}"
            )
            mean_val_loss = sum([v[0] for v in val_results.values()]) / 3
            self.scheduler.step(mean_val_loss)
            if mean_val_loss < best_loss:
                best_loss = mean_val_loss
                patience_counter = 0
                torch.save(self.model.state_dict(), "best_clip_model.pt")
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break

    def extract_features(self, dataloader, mode="image"):
        # 이미지/텍스트 임베딩 추출
        self.model.eval()
        features, labels, texts = [], [], []
        with torch.no_grad():
            for batch in tqdm(dataloader, leave=False):
                images = batch["image"].to(self.device, non_blocking=True)
                text = batch["text"]
                if mode == "image":
                    img_emb = self.model.clip.encode_image(images)
                    img_emb = img_emb / img_emb.norm(dim=-1, keepdim=True)
                    features.append(img_emb.cpu().numpy())
                    labels.extend(batch.get("label", ["combined"]*len(images)))
                    texts.extend(text)
                else:
                    text_tokens = clip.tokenize(text).to(self.device, non_blocking=True)
                    txt_emb = self.model.clip.encode_text(text_tokens)
                    txt_emb = txt_emb / txt_emb.norm(dim=-1, keepdim=True)
                    features.append(txt_emb.cpu().numpy())
                    labels.extend(batch.get("label", ["combined"]*len(text)))
                    texts.extend(text)
        return np.concatenate(features), labels, texts

    def visualize_features_3d(self, dataloader, title=""):
        # 이미지/텍스트 임베딩을 3D t-SNE로 시각화
        img_features, img_labels, _ = self.extract_features(dataloader, mode="image")
        txt_features, txt_labels, txt_texts = self.extract_features(dataloader, mode="text")
        features = np.concatenate([img_features, txt_features], axis=0)
        labels = np.array(list(img_labels) + list(txt_labels))
        types = np.array(['image'] * len(img_labels) + ['text'] * len(txt_labels))
        tsne = TSNE(n_components=3, random_state=42)
        reduced = tsne.fit_transform(features)
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        img_idx = types == 'image'
        ax.scatter(reduced[img_idx, 0], reduced[img_idx, 1], reduced[img_idx, 2],
                   c='b', label='Image', alpha=0.6, s=40)
        txt_idx = types == 'text'
        ax.scatter(reduced[txt_idx, 0], reduced[txt_idx, 1], reduced[txt_idx, 2],
                   c='r', label='Text', alpha=0.6, s=40, marker='^')
        ax.set_title(f"3D t-SNE: {title}")
        ax.set_xlabel("Dim 1")
        ax.set_ylabel("Dim 2")
        ax.set_zlabel("Dim 3")
        ax.legend()
        plt.show()

    def faiss_search(self, image_dataloader, text_dataloader, k=5):
        # 텍스트 임베딩으로 이미지 임베딩을 FAISS로 검색
        image_features, image_labels, _ = self.extract_features(image_dataloader, mode="image")
        text_features, text_labels, text_strs = self.extract_features(text_dataloader, mode="text")
        index = faiss.IndexFlatIP(image_features.shape[1])
        faiss.normalize_L2(image_features)
        faiss.normalize_L2(text_features)
        index.add(image_features.astype(np.float32))
        D, I = index.search(text_features.astype(np.float32), k)
        for i, (text, idxs) in enumerate(zip(text_strs, I)):
            print(f"\n[텍스트: {text}]")
            for rank, idx in enumerate(idxs):
                print(f"  {rank+1}위: 이미지 라벨={image_labels[idx]} (유사도={D[i][rank]:.3f})")

# ----- 메인 실행 -----
if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # 현재 파일 경로
    device = "cuda" if torch.cuda.is_available() else "cpu"  # GPU/CPU 자동 선택
    if device == "cpu":
        torch.set_num_threads(os.cpu_count())  # CPU 멀티스레드 설정
    trainer = CLIPTrainer(
        color_dir=os.path.join(BASE_DIR, "dataset", "colors"),
        object_dir=os.path.join(BASE_DIR, "dataset", "objects"),
        combined_dir=os.path.join(BASE_DIR, "dataset", "combined"),
        device=device,
        batch_size=64,
        sample_ratio=1.0,
        patience=5
    )
    trainer.train(epochs=30)  # 전체 학습

    # 3D t-SNE 시각화 (color/object/combined)
    print("3D t-SNE 시각화 (color)")
    trainer.visualize_features_3d(trainer.val_color_loader, title="Color")
    print("3D t-SNE 시각화 (object)")
    trainer.visualize_features_3d(trainer.val_object_loader, title="Object")
    print("3D t-SNE 시각화 (combined)")
    trainer.visualize_features_3d(trainer.val_combined_loader, title="Combined")

    # FAISS 기반 텍스트→이미지 검색 (color/object/combined)
    print("FAISS 기반 텍스트→이미지 검색 (color)")
    trainer.faiss_search(trainer.val_color_loader, trainer.val_color_loader, k=3)
    print("FAISS 기반 텍스트→이미지 검색 (object)")
    trainer.faiss_search(trainer.val_object_loader, trainer.val_object_loader, k=3)
    print("FAISS 기반 텍스트→이미지 검색 (combined)")
    trainer.faiss_search(trainer.val_combined_loader, trainer.val_combined_loader, k=3)