# [코드 기능 요약]
# 1. CLIP 기반 멀티모달(이미지+텍스트) 분류 및 임베딩 학습 시스템 
# 2. 주요 기능:
#   - 색상/물건 데이터 분리 학습 및 평가 (각 분류 헤드만 지원)
#   - Focal Loss로 클래스 불균형 완화
#   - 클래스별 오버샘플링 및 동의어 정규화(색상/물건)
#   - 강화된 이미지 증강 파이프라인
#   - contrastive loss(이미지-텍스트 임베딩 정렬) 동시 학습
#   - t-SNE 기반 3D 임베딩 시각화 (이미지/텍스트 동시)
#   - FAISS를 통한 고속 텍스트→이미지 검색
# 3. 학습 전략:
#   - 혼합 정밀도(Mixed Precision) 학습(GPU)
#   - 조기 종료(Early Stopping), ReduceLROnPlateau 학습률 스케줄링

import os  # 운영체제 기능 사용
import sys  # 시스템 관련 기능 사용
import io  # 입출력 관련 기능
import json  # JSON 파일 입출력
import random  # 랜덤 샘플링
import numpy as np  # 수치 연산
from collections import Counter  # 데이터 카운팅
from PIL import Image  # 이미지 파일 처리
import torch  # PyTorch 메인 모듈
import torch.nn as nn  # 신경망 모듈
import torch.nn.functional as F  # 신경망 함수
import torch.optim as optim  # 최적화 함수
from torch.utils.data import Dataset, DataLoader, random_split, WeightedRandomSampler  # 데이터 관련
from torchvision import transforms  # 이미지 변환
from tqdm import tqdm  # 진행률 표시
import matplotlib.pyplot as plt  # 시각화
from sklearn.manifold import TSNE  # 차원 축소(t-SNE)
from torch.optim.lr_scheduler import ReduceLROnPlateau  # 학습률 스케줄러
import faiss  # 고속 벡터 검색 라이브러리
import clip  # OpenAI CLIP 모델
import matplotlib

matplotlib.rc('font', family='Malgun Gothic')  # Windows의 경우
matplotlib.rcParams['axes.unicode_minus'] = False  # 마이너스 기호 출력 설정
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # 라이브러리 충돌 방지
sys.stdout = io.TextIOWrapper(sys.stdout.detach(), encoding='utf-8')  # 한글 출력

BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # 현재 파일의 디렉토리 경로를 BASE_DIR로 저장
color_dir = os.path.join(BASE_DIR, "dataset", "colors")  # 색상 데이터 경로
object_dir = os.path.join(BASE_DIR, "dataset", "objects")  # 물건 데이터 경로
combined_dir = os.path.join(BASE_DIR, "dataset", "combined")  # 조합 데이터 경로

# 동의어 그룹을 대표값-동의어 맵으로 변환하는 함수
def build_synonym_map(synonym_groups):
    class_list = []  # 대표값 리스트
    synonym_map = {}  # 동의어 맵
    for group in synonym_groups:
        rep = group[0]  # 대표값
        class_list.append(rep)
        for word in group:
            synonym_map[word] = rep  # 동의어 모두 대표값으로 매핑
    return class_list, synonym_map

# 색상 동의어 그룹 정의
color_synonyms = [
    ["흰색", "화이트", "하얀색"],
    ["검은색","검정", "블랙", "까만색", "검정색", "다크"],
    ["빨간색", "레드", "빨강강"],
    ["파란색", "블루", "파랑", "스카이 블루"],
    ["하늘색", "아이스 블루"],
    ["노란색", "옐로우", "노랑", "금색", "골드"],
    ["초록색", "그린", "녹색", "초록"],
    ["남색", "네이비"],
    ["분홍색", "핑크", "연분홍", "분홍"],
    ["주황색", "오렌지", "주황"],
    ["보라색", "퍼플", "바이올렛", "보라"],
    ["갈색", "브라운", "황토색"],
    ["회색", "실버", "은색", "그레이"],
    ["베이지", "크림"],
    ["청녹색", "티파니", "청록색", "청록", "청녹"],
    ["자주색", "마젠타", "자주"],
    ["민트", "연두색", "라임", "연두"],
]
color_classes, COLOR_SYNONYM_MAP = build_synonym_map(color_synonyms)  # 색상 클래스 및 맵 생성

# 물건 동의어 그룹 정의
object_synonyms = [
    ["갤럭시", "휴대폰", "스마트폰", "폰"],
    ["아이폰", "휴대폰", "폰"],
    ["가방", "백팩"],
    ["백팩", "가방"],
    ["갤럭시 버즈", "버즈"],
    ["자켓", "겉옷", "재킷"],
    ["아이패드", "패드"],
    ["맥북", "노트북"],
    ["노트북"],
    ["카드", "신용카드", "체크카드"],
    ["지갑", "카드지갑"],
    ["카드지갑", "지갑"]
    ["모자", "캡", "캡모자"],
    ["노트북 마우스", "마우스", "무선마우스"],
    ["태블릿", "태블렛"],
    ["이어폰", "무선 이어폰"],
    ["샌들", "샌달"],
    ["스마트워치", "갤럭시워치"],
    ["헤드셋", "헤드폰"]
]
object_classes, OBJECT_SYNONYM_MAP = build_synonym_map(object_synonyms)  # 물건 클래스 및 맵 생성

def normalize_color(label):
    # 색상 라벨을 동의어 맵을 이용해 대표값으로 변환
    return COLOR_SYNONYM_MAP.get(label.strip(), label.strip())

def normalize_object(label):
    # 물건 라벨을 동의어 맵을 이용해 대표값으로 변환
    return OBJECT_SYNONYM_MAP.get(label.strip(), label.strip())

# Focal Loss 정의 (클래스 불균형 완화)
class FocalLoss(nn.Module):
    def __init__(self, gamma=1.5, weight=None, ignore_index=-100):
        super().__init__()
        self.gamma = gamma
        self.weight = weight
        self.ignore_index = ignore_index

    def forward(self, input, target):
        ce_loss = F.cross_entropy(input, target, weight=self.weight,
                                  ignore_index=self.ignore_index, reduction='none')
        pt = torch.exp(-ce_loss)
        return ((1 - pt) ** self.gamma * ce_loss).mean()

# 데이터셋 클래스 정의
class AugmentedCLIPDataset(Dataset):
    def __init__(self, color_dir, object_dir, combined_dir, preprocess, train=True, oversample_min_count=50):
        self.color_classes = sorted(set([normalize_color(label.strip()) for label in os.listdir(color_dir)]))  # 색상 클래스
        self.object_classes = sorted(set([normalize_object(label.strip()) for label in os.listdir(object_dir)]))  # 물건 클래스
        self.combined_classes = []
        if os.path.exists(combined_dir):
            self.combined_classes = sorted(set([
                normalize_color(label.strip().split('_')[0]) + '_' + normalize_object(label.strip().split('_')[1])
                for label in os.listdir(combined_dir)
            ]))
        self.preprocess = preprocess  # CLIP 전처리
        self.train = train  # 학습/검증 구분

        # 이미지 증강 파이프라인
        self.train_transform = transforms.Compose([
            transforms.RandomRotation(40),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.3),
            transforms.RandomResizedCrop(224, scale=(0.9, 1.0)),
            transforms.RandomAffine(degrees=20, translate=(0.1, 0.1), scale=(0.8, 1.2)),
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
            preprocess
        ])
        self.val_transform = preprocess  # 검증용 변환

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
        image = Image.open(item["path"]).convert("RGB")  # 이미지 로드
        transform = self.train_transform if self.train else self.val_transform  # 증강/검증 변환 선택
        if item["type"] == "combined":
            return {
                "image": transform(image),
                "text": item["text"],
                "type": "combined",
                "color": normalize_color(item["color"]),
                "object": normalize_object(item["object"]),
                "label": f"{normalize_color(item['color'])}_{normalize_object(item['object'])}"
            }
        else:
            return {
                "image": transform(image),
                "text": f"{item['label']} 색상" if item["type"] == "color" else f"{item['label']} 물건",
                "type": item["type"],
                "label": item["label"],
                "color": normalize_color(item["label"]) if item["type"] == "color" else "",
                "object": normalize_object(item["label"]) if item["type"] == "object" else ""
            }

# CLIP 기반 멀티모달 분류 모델 정의
class EnhancedCLIPModel(nn.Module):
    def __init__(self, clip_model, color_classes, object_classes):
        super().__init__()
        self.clip = clip_model  # CLIP 백본
        self.image_adapter = nn.Sequential(
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Linear(256, 512)
        )
        self.text_adapter = nn.Sequential(
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Linear(256, 512)
        )
        self.color_head = nn.Linear(512, len(color_classes))  # 색상 분류 헤드
        self.object_head = nn.Linear(512, len(object_classes))  # 물건 분류 헤드
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))  # CLIP logit scaling 파라미터

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
            adapted_img,  # 이미지 임베딩
            adapted_txt   # 텍스트 임베딩
        )

# Trainer 클래스
class CLIPTrainer:
    def __init__(self, color_dir, object_dir, combined_dir, device, batch_size, sample_ratio, patience):
        self.device = device  # 학습 디바이스
        self.clip_model, preprocess = clip.load("ViT-B/32", device=device, jit=False)  # CLIP 모델 로드
        self.clip_model.float()  # float32로 변환
        self.dataset = AugmentedCLIPDataset(
            color_dir, object_dir, combined_dir, preprocess, train=True, oversample_min_count=50
        )
        if sample_ratio < 1.0:
            n = int(len(self.dataset) * sample_ratio)
            indices = torch.randperm(len(self.dataset))[:n]
            self.dataset.data = [self.dataset.data[i] for i in indices]
            self.dataset.class_weights = self.dataset._calculate_class_weights()
        train_size = int(0.9 * len(self.dataset))
        self.train_dataset, self.val_dataset = random_split(
            self.dataset, [train_size, len(self.dataset) - train_size]
        )
        train_weights = self.dataset.class_weights[self.train_dataset.indices]
        sampler = WeightedRandomSampler(train_weights, len(train_weights))
        self.train_loader = DataLoader(
            self.train_dataset, batch_size=batch_size, sampler=sampler, num_workers=4, pin_memory=False
        )
        self.val_loader = DataLoader(
            self.val_dataset, batch_size=batch_size, num_workers=4, pin_memory=False
        )
        self.model = EnhancedCLIPModel(
            self.clip_model,
            self.dataset.color_classes,
            self.dataset.object_classes
        ).to(device)
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=1e-6,
            weight_decay=0.005
        )
        self.scheduler = ReduceLROnPlateau(self.optimizer, mode='min', factor=0.5, patience=2)
        self.criterion = FocalLoss(ignore_index=self.model.color_head.out_features - 1)
        self.scaler = torch.amp.GradScaler() if self.device == "cuda" else None
        self.patience = patience

    def train_epoch(self):
        self.model.train()
        total_loss, total_correct, total_count = 0, 0, 0
        for batch in tqdm(self.train_loader, desc="Train", leave=False):
            images = batch["image"].to(self.device, non_blocking=True)  # 이미지 텐서
            texts = batch["text"]  # 텍스트 리스트
            text_tokens = clip.tokenize(texts).to(self.device, non_blocking=True)  # 텍스트 토큰화
            color_labels, object_labels = [], []
            for i, t in enumerate(batch["type"]):
                if t == "combined":
                    color_labels.append(self.model.color_head.out_features - 1)
                    object_labels.append(self.model.object_head.out_features - 1)
                elif t == "color":
                    color_labels.append(self.dataset.color_classes.index(batch["label"][i]))
                    object_labels.append(self.model.object_head.out_features - 1)
                else:
                    color_labels.append(self.model.color_head.out_features - 1)
                    object_labels.append(self.dataset.object_classes.index(batch["label"][i]))
            color_labels = torch.tensor(color_labels, device=self.device)
            object_labels = torch.tensor(object_labels, device=self.device)

            if self.device == "cuda":
                with torch.amp.autocast(self.device):
                    color_logits, object_logits, img_emb, txt_emb = self.model(images, text_tokens)
                    color_loss = self.criterion(color_logits, color_labels)
                    object_loss = self.criterion(object_logits, object_labels)
                    logit_scale = self.model.logit_scale.exp()
                    logits_per_image = logit_scale * img_emb @ txt_emb.T
                    logits_per_text = logits_per_image.T
                    ground_truth = torch.arange(len(images), device=self.device)
                    contrastive_loss = (self.criterion(logits_per_image, ground_truth) + self.criterion(logits_per_text, ground_truth)) / 2
                    loss = 0.6 * (color_loss + object_loss) + 0.4 * contrastive_loss
                    self.optimizer.zero_grad()
                    self.scaler.scale(loss).backward()
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
            else:
                color_logits, object_logits, img_emb, txt_emb = self.model(images, text_tokens)
                color_loss = self.criterion(color_logits, color_labels)
                object_loss = self.criterion(object_logits, object_labels)
                logit_scale = self.model.logit_scale.exp()
                logits_per_image = logit_scale * img_emb @ txt_emb.T
                logits_per_text = logits_per_image.T
                ground_truth = torch.arange(len(images), device=self.device)
                contrastive_loss = (self.criterion(logits_per_image, ground_truth) + self.criterion(logits_per_text, ground_truth)) / 2
                loss = 0.6 * (color_loss + object_loss) + 0.4 * contrastive_loss
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()

            color_mask = color_labels != (self.model.color_head.out_features - 1)
            object_mask = object_labels != (self.model.object_head.out_features - 1)
            total_count += color_mask.sum().item() + object_mask.sum().item()
            total_loss += loss.item()

            # Top-3(2개 이상) 정확도
            if color_mask.any():
                color_top3 = color_logits[color_mask].topk(3, dim=1).indices
                color_label_expand = color_labels[color_mask].unsqueeze(1).expand(-1, 3)
                color_match = (color_top3 == color_label_expand).sum(dim=1)
                total_correct += (color_match >= 2).sum().item()
            if object_mask.any():
                object_top3 = object_logits[object_mask].topk(3, dim=1).indices
                object_label_expand = object_labels[object_mask].unsqueeze(1).expand(-1, 3)
                object_match = (object_top3 == object_label_expand).sum(dim=1)
                total_correct += (object_match >= 2).sum().item()

        avg_loss = total_loss / len(self.train_loader)
        avg_acc = total_correct / total_count if total_count > 0 else 0
        return avg_loss, avg_acc

    def validate_epoch(self):
        self.model.eval()
        total_loss, total_correct, total_count = 0, 0, 0
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Val", leave=False):
                images = batch["image"].to(self.device, non_blocking=True)
                texts = batch["text"]
                text_tokens = clip.tokenize(texts).to(self.device, non_blocking=True)
                color_labels, object_labels = [], []
                for i, t in enumerate(batch["type"]):
                    if t == "combined":
                        color_labels.append(self.model.color_head.out_features - 1)
                        object_labels.append(self.model.object_head.out_features - 1)
                    elif t == "color":
                        color_labels.append(self.dataset.color_classes.index(batch["label"][i]))
                        object_labels.append(self.model.object_head.out_features - 1)
                    else:
                        color_labels.append(self.model.color_head.out_features - 1)
                        object_labels.append(self.dataset.object_classes.index(batch["label"][i]))
                color_labels = torch.tensor(color_labels, device=self.device)
                object_labels = torch.tensor(object_labels, device=self.device)

                color_logits, object_logits, img_emb, txt_emb = self.model(images, text_tokens)
                color_loss = self.criterion(color_logits, color_labels)
                object_loss = self.criterion(object_logits, object_labels)
                logit_scale = self.model.logit_scale.exp()
                logits_per_image = logit_scale * img_emb @ txt_emb.T
                logits_per_text = logits_per_image.T
                ground_truth = torch.arange(len(images), device=self.device)
                contrastive_loss = (self.criterion(logits_per_image, ground_truth) + self.criterion(logits_per_text, ground_truth)) / 2
                loss = 0.6 * (color_loss + object_loss) + 0.4 * contrastive_loss

                color_mask = color_labels != (self.model.color_head.out_features - 1)
                object_mask = object_labels != (self.model.object_head.out_features - 1)
                total_count += color_mask.sum().item() + object_mask.sum().item()
                total_loss += loss.item()

                if color_mask.any():
                    color_top3 = color_logits[color_mask].topk(3, dim=1).indices
                    color_label_expand = color_labels[color_mask].unsqueeze(1).expand(-1, 3)
                    color_match = (color_top3 == color_label_expand).sum(dim=1)
                    total_correct += (color_match >= 2).sum().item()
                if object_mask.any():
                    object_top3 = object_logits[object_mask].topk(3, dim=1).indices
                    object_label_expand = object_labels[object_mask].unsqueeze(1).expand(-1, 3)
                    object_match = (object_top3 == object_label_expand).sum(dim=1)
                    total_correct += (object_match >= 2).sum().item()
        avg_loss = total_loss / len(self.val_loader)
        avg_acc = total_correct / total_count if total_count > 0 else 0
        return avg_loss, avg_acc

    # 전체 학습 루프
    def train(self, epochs):
        best_loss = float('inf')  # 최적 검증 손실 초기화
        patience_counter = 0  # 조기 종료 카운터
        for epoch in range(epochs):
            train_loss, train_acc = self.train_epoch()  # 학습 1에폭
            val_loss, val_acc = self.validate_epoch()  # 검증 1에폭
            print(
                f"Epoch {epoch+1}/{epochs} | "
                f"Train Loss={train_loss:.4f}, Train Acc={train_acc:.4f} | "
                f"Val Loss={val_loss:.4f}, Val Acc={val_acc:.4f}"
            )
            self.scheduler.step(val_loss)  # 학습률 스케줄러
            if val_loss < best_loss:
                best_loss = val_loss
                patience_counter = 0
                torch.save(self.model.state_dict(), "best_clip_model.pt")  # 모델 저장
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
                    labels.extend(batch.get("label", ["unknown"] * len(images)))
                    texts.extend(text)
                else:
                    text_tokens = clip.tokenize(text).to(self.device, non_blocking=True)
                    txt_emb = self.model.clip.encode_text(text_tokens)
                    txt_emb = txt_emb / txt_emb.norm(dim=-1, keepdim=True)
                    features.append(txt_emb.cpu().numpy())
                    labels.extend(batch.get("label", ["unknown"] * len(text)))
                    texts.extend(text)
        return np.concatenate(features), labels, texts

    # 3D t-SNE 시각화 메소드
    def visualize_features_3d(self, dataloader, title="", mode="type"):
        img_features, _, _ = self.extract_features(dataloader, mode="image")
        txt_features, _, _ = self.extract_features(dataloader, mode="text")
        features = np.concatenate([img_features, txt_features], axis=0)

        # Subset일 경우 원본 dataset 접근
        dataset = dataloader.dataset
        if hasattr(dataset, 'dataset'):
            dataset = dataset.dataset

        # 실제 데이터 개수와 맞추기
        n_img = img_features.shape[0]
        n_txt = txt_features.shape[0]

        color_labels = [item.get("color", "") for item in dataset.data]
        object_labels = [item.get("object", "") for item in dataset.data]
        type_labels = [item.get("type", "") for item in dataset.data]

        # 길이 맞추기
        color_labels = color_labels[:n_img]
        object_labels = object_labels[:n_img]
        type_labels = type_labels[:n_img]

        if mode == "color":
            labels = color_labels + color_labels
        elif mode == "object":
            labels = object_labels + object_labels
        else:
            labels = type_labels + type_labels

        labels = labels[:features.shape[0]]  # features와 labels 길이 맞추기

        unique = sorted(set(labels))
        color_map = {u: plt.cm.tab20(i % 20) for i, u in enumerate(unique)}
        point_colors = [color_map[l] for l in labels]

        # perplexity는 n_samples보다 작아야 함
        n_samples = features.shape[0]
        perplexity = min(30, max(2, (n_samples - 1) // 3))

        tsne = TSNE(n_components=3, random_state=42, perplexity=perplexity)
        reduced = tsne.fit_transform(features)
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(reduced[:, 0], reduced[:, 1], reduced[:, 2],
                c=point_colors, alpha=0.5, s=40)
        for item in unique:
            ax.scatter([], [], [], c=[color_map[item]], label=item)
        ax.set_title(f"3D t-SNE: {title} ({mode})")
        ax.set_xlabel("Dim 1")
        ax.set_ylabel("Dim 2")
        ax.set_zlabel("Dim 3")
        ax.legend()
        plt.show()

    def faiss_search(self, image_dataloader, text_dataloader, k=3):
        # 텍스트→이미지 검색 및 Top-3(2개 이상) 정확도 평가
        image_features, image_labels, _ = self.extract_features(image_dataloader, mode="image")
        text_features, text_labels, text_strs = self.extract_features(text_dataloader, mode="text")
        index = faiss.IndexFlatIP(image_features.shape[1])
        faiss.normalize_L2(image_features)
        faiss.normalize_L2(text_features)
        index.add(image_features.astype(np.float32))
        D, I = index.search(text_features.astype(np.float32), k)

        correct = 0
        total = len(text_labels)
        for i, (text, idxs) in enumerate(zip(text_strs, I)):
            gt_label = text_labels[i]
            retrieved_labels = [image_labels[idx] for idx in idxs]
            gt_label_count = retrieved_labels.count(gt_label)
            is_correct = gt_label_count >= 2
            if is_correct:
                correct += 1
            print(f"\n[텍스트: {text}]")
            for rank, idx in enumerate(idxs):
                print(f"  {rank+1}위: 이미지 라벨={image_labels[idx]} (유사도={D[i][rank]:.3f})")

        topk_acc = correct / total if total > 0 else 0
        print(f"\nFAISS Accuracy: {topk_acc:.4f}")

# 메인 실행부
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"  # GPU가 있으면 'cuda', 없으면 'cpu' 사용
    if device == "cpu":
        torch.set_num_threads(os.cpu_count())  # CPU 코어 수만큼 PyTorch가 멀티스레드로 동작하도록 설정

    trainer = CLIPTrainer(
        color_dir=os.path.join(BASE_DIR, "dataset", "colors"),
        object_dir=os.path.join(BASE_DIR, "dataset", "objects"),
        combined_dir=os.path.join(BASE_DIR, "dataset", "combined"),
        device=device,
        batch_size=32,  # 미니배치 크기 지정
        sample_ratio=1.0,  # 전체 데이터 중 사용할 비율(1.0이면 전체 사용)
        patience=5  # 검증 손실이 개선되지 않을 때 조기 종료
    )
    trainer.train(epochs=25)  # 25회 반복

    class_info = {
        "color_classes": trainer.dataset.color_classes,
        "object_classes": trainer.dataset.object_classes,
    }
    os.makedirs(os.path.join(BASE_DIR, "data"), exist_ok=True)
    with open(os.path.join(BASE_DIR, "data", "clip_classes.json"), "w", encoding="utf-8") as f:
        json.dump(class_info, f, ensure_ascii=False, indent=2)  # 클래스 정보 저장

    print("3D t-SNE 시각화 (색상별)")
    trainer.visualize_features_3d(trainer.val_loader, title="Color", mode="color")

    print("3D t-SNE 시각화 (물건별)")
    trainer.visualize_features_3d(trainer.val_loader, title="Object", mode="object")
    
    print("3D t-SNE 시각화 (전체)")
    trainer.visualize_features_3d(trainer.val_loader, title="All", mode="type")

    print("FAISS 기반 텍스트→이미지 검색")
    trainer.faiss_search(trainer.val_loader, trainer.val_loader, k=3)
    
    