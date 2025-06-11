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

import os  # 운영체제 관련 기능을 사용하기 위한 모듈
import sys  # 시스템 관련 기능을 사용하기 위한 모듈
import io  # 입출력 관련 기능을 사용하기 위한 모듈
import json  # JSON 파일 입출력용 모듈
import random  # 랜덤 샘플링을 위한 모듈
import math  # 수학 함수 사용을 위한 모듈
import numpy as np  # 수치 연산을 위한 모듈
from collections import Counter  # 데이터 카운팅을 위한 모듈
from PIL import Image  # 이미지 파일 처리를 위한 모듈
import torch  # PyTorch 메인 모듈
import torch.nn as nn  # 신경망 모듈
import torch.nn.functional as F  # 신경망 함수
import torch.optim as optim  # 최적화 함수
from torch.utils.data import Dataset, DataLoader, random_split, WeightedRandomSampler  # 데이터 관련 클래스
from torchvision import transforms  # 이미지 변환 함수
from tqdm import tqdm  # 진행률 표시 바
import matplotlib.pyplot as plt  # 시각화 라이브러리
from collections import Counter  # 데이터 카운팅을 위한 모듈
from sklearn.manifold import TSNE  # 차원 축소(t-SNE) 라이브러리
from torch.optim.lr_scheduler import ReduceLROnPlateau  # 학습률 스케줄러
import faiss  # 고속 벡터 검색 라이브러리
import clip  # OpenAI CLIP 모델
import matplotlib  # matplotlib 설정용

matplotlib.rc('font', family='Malgun Gothic')  # 한글 폰트 설정(Windows)
matplotlib.rcParams['axes.unicode_minus'] = False  # 마이너스 기호 깨짐 방지
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # 라이브러리 충돌 방지
sys.stdout = io.TextIOWrapper(sys.stdout.detach(), encoding='utf-8')  # 한글 출력 설정

BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # 현재 파일의 디렉토리 경로를 BASE_DIR로 저장
color_dir = os.path.join(BASE_DIR, "dataset", "colors")  # 색상 데이터 경로
object_dir = os.path.join(BASE_DIR, "dataset", "objects")  # 물건 데이터 경로
combined_dir = os.path.join(BASE_DIR, "dataset", "combined")  # 조합 데이터 경로

# 동의어 그룹을 대표값-동의어 맵으로 변환하는 함수
def build_synonym_map(synonym_groups):
    class_list = []  # 대표값 리스트
    synonym_map = {}  # 동의어 맵
    for group in synonym_groups:  # 각 동의어 그룹에 대해
        rep = group[0]  # 첫 번째 단어를 대표값으로 지정
        class_list.append(rep)  # 대표값 리스트에 추가
        for word in group:  # 그룹 내 모든 단어에 대해
            synonym_map[word] = rep  # 동의어를 대표값으로 매핑
    return class_list, synonym_map  # 대표값 리스트와 동의어 맵 반환

# 색상 동의어 그룹 정의
color_synonyms = [
    ["흰색", "화이트", "하얀색"],
    ["검은색","검정", "블랙", "까만색", "검정색", "다크"],
    ["빨간색", "레드", "빨강"],
    ["파란색", "블루", "파랑", "스카이블루"],
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
    ["민트", "연두색", "라임", "연두"]
]
color_classes, COLOR_SYNONYM_MAP = build_synonym_map(color_synonyms)  # 색상 클래스 및 동의어 맵 생성

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
    ["카드지갑", "지갑"],
    ["모자", "캡", "캡모자"],
    ["노트북 마우스", "마우스", "무선마우스"],
    ["태블릿", "태블렛"],
    ["이어폰", "무선 이어폰"],
    ["샌들", "샌달"],
    ["스마트워치", "갤럭시워치"],
    ["헤드셋", "헤드폰"]
]
object_classes, OBJECT_SYNONYM_MAP = build_synonym_map(object_synonyms)  # 물건 클래스 및 동의어 맵 생성

def normalize_color(label):
    # 색상 라벨을 동의어 맵을 이용해 대표값으로 변환
    return COLOR_SYNONYM_MAP.get(label.strip(), label.strip())

def normalize_object(label):
    # 물건 라벨을 동의어 맵을 이용해 대표값으로 변환
    return OBJECT_SYNONYM_MAP.get(label.strip(), label.strip())

# Focal Loss 정의 (클래스 불균형 완화)
class FocalLoss(nn.Module):
    def __init__(self, gamma=1.5, weight=None, ignore_index=-100):
        super().__init__()  # 부모 클래스 초기화
        self.gamma = gamma  # Focal Loss의 감쇠 계수
        self.weight = weight  # 클래스별 가중치
        self.ignore_index = ignore_index  # 무시할 인덱스

    def forward(self, input, target):
        # 입력과 타겟을 받아 Focal Loss 계산
        ce_loss = F.cross_entropy(input, target, weight=self.weight,
                                  ignore_index=self.ignore_index, reduction='none')  # Cross Entropy Loss
        pt = torch.exp(-ce_loss)  # 예측 확률
        return ((1 - pt) ** self.gamma * ce_loss).mean()  # Focal Loss 공식 적용

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
        self.preprocess = preprocess  # CLIP 전처리 함수
        self.train = train  # 학습/검증 구분

        # 이미지 증강 파이프라인 정의
        self.train_transform = transforms.Compose([
            transforms.RandomRotation(40),  # 40도 이내 랜덤 회전
            transforms.RandomHorizontalFlip(p=0.5),  # 50% 확률로 좌우 반전
            transforms.RandomVerticalFlip(p=0.3),  # 30% 확률로 상하 반전
            transforms.RandomResizedCrop(224, scale=(0.9, 1.0)),  # 랜덤 크롭
            transforms.RandomAffine(degrees=20, translate=(0.1, 0.1), scale=(0.8, 1.2)),  # 랜덤 어파인 변환
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),  # 가우시안 블러
            preprocess  # CLIP 기본 전처리
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
        label_list = [item['label'] for item in self.data]  # 모든 라벨 리스트
        label_counts = Counter(label_list)  # 라벨별 개수 카운트
        for label, count in label_counts.items():
            if count < min_count:  # min_count보다 적으면
                add_count = min_count - count  # 추가할 개수 계산
                samples = [item for item in self.data if item['label'] == label]  # 해당 라벨 샘플 추출
                self.data.extend(random.choices(samples, k=add_count))  # 랜덤 복제하여 추가

    def _load_data(self, root_dir, data_type):
        # 색상/물건 데이터 로드
        for label in os.listdir(root_dir):  # 각 라벨 폴더에 대해
            norm_label = normalize_color(label.strip()) if data_type == "color" else normalize_object(label.strip())  # 동의어 정규화
            label_dir = os.path.join(root_dir, label)  # 폴더 경로
            for img_file in os.listdir(label_dir):  # 폴더 내 이미지 파일에 대해
                self.data.append({
                    "path": os.path.join(label_dir, img_file),  # 이미지 경로
                    "type": data_type,  # 데이터 타입(color/object)
                    "label": norm_label  # 정규화된 라벨
                })

    def _get_combined_prompts(self, color, obj):
        # 조합 데이터의 텍스트 프롬프트 생성
        prompts = [
            f"{color} {obj}",
            f"분실된 {color} {obj} 사진",
            f"이 사진은 {color} {obj}입니다",
            f"{color} 색상의 {obj}"
        ]
        return random.choice(prompts)  # 랜덤 프롬프트 반환

    def _load_combined_data(self, combined_dir):
        # 색상+물건 조합 데이터 로드
        if not combined_dir:
            return
        for combined_label in os.listdir(combined_dir):  # 조합 라벨 폴더에 대해
            parts = combined_label.split('_')
            if len(parts) < 2:
                continue  # 잘못된 폴더명은 건너뜀
            color = normalize_color(parts[0])  # 색상 정규화
            obj = normalize_object(parts[1])  # 물건 정규화
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
        return torch.DoubleTensor(weights)  # 각 샘플별 가중치 반환

    def __len__(self):
        # 전체 데이터 개수 반환
        return len(self.data)

    def __getitem__(self, idx):
        # 인덱스에 해당하는 데이터 반환
        item = self.data[idx]
        image = Image.open(item["path"]).convert("RGB")  # 이미지 로드 및 RGB 변환
        transform = self.train_transform if self.train else self.val_transform  # 증강/검증 변환 선택
        if item["type"] == "combined":
            # 조합 데이터인 경우
            return {
                "image": transform(image),
                "text": item["text"],
                "type": "combined",
                "color": normalize_color(item["color"]),
                "object": normalize_object(item["object"]),
                "label": f"{normalize_color(item['color'])}_{normalize_object(item['object'])}"
            }
        else:
            # 색상/물건 단일 데이터인 경우
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
        super().__init__()  # 부모 클래스 초기화
        self.clip = clip_model  # CLIP 백본
        self.image_adapter = nn.Sequential(
            nn.Linear(512, 256),  # 512차원 → 256차원
            nn.GELU(),  # GELU 활성화 함수
            nn.Linear(256, 512)  # 256차원 → 512차원
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
        image_features = self.clip.encode_image(images)  # 이미지 임베딩 추출
        text_features = self.clip.encode_text(texts)  # 텍스트 임베딩 추출
        adapted_img = image_features + 0.05 * self.image_adapter(image_features)  # 이미지 어댑터 적용
        adapted_txt = text_features + 0.05 * self.text_adapter(text_features)  # 텍스트 어댑터 적용
        adapted_img = adapted_img / adapted_img.norm(dim=-1, keepdim=True)  # 정규화
        adapted_txt = adapted_txt / adapted_txt.norm(dim=-1, keepdim=True)  # 정규화
        return (
            self.color_head(adapted_img),  # 색상 분류 결과
            self.object_head(adapted_img),  # 물건 분류 결과
            adapted_img,  # 이미지 임베딩
            adapted_txt   # 텍스트 임베딩
        )

def get_random_color():
    # 임의의 16진수 색상 코드 반환
    return "#%06x" % random.randint(0, 0xFFFFFF)

# Trainer 클래스
class CLIPTrainer:
    def __init__(self, color_dir, object_dir, combined_dir, device, batch_size, sample_ratio, patience):
        self.device = device  # 학습 디바이스(cpu/gpu)
        self.clip_model, preprocess = clip.load("ViT-B/32", device=device, jit=False)  # CLIP 모델 로드
        self.clip_model.float()  # float32로 변환
        self.dataset = AugmentedCLIPDataset(
            color_dir, object_dir, combined_dir, preprocess, train=True, oversample_min_count=50
        )
        if sample_ratio < 1.0:
            n = int(len(self.dataset) * sample_ratio)  # 사용할 데이터 개수
            indices = torch.randperm(len(self.dataset))[:n]  # 무작위 인덱스 선택
            self.dataset.data = [self.dataset.data[i] for i in indices]  # 데이터 샘플링
            self.dataset.class_weights = self.dataset._calculate_class_weights()  # 가중치 재계산
        train_size = int(0.9 * len(self.dataset))  # 학습 데이터 비율(90%)
        self.train_dataset, self.val_dataset = random_split(
            self.dataset, [train_size, len(self.dataset) - train_size]
        )
        train_weights = self.dataset.class_weights[self.train_dataset.indices]  # 학습 데이터 가중치
        sampler = WeightedRandomSampler(train_weights, len(train_weights))  # 오버샘플링 샘플러
        self.train_loader = DataLoader(
            self.train_dataset, batch_size=batch_size, sampler=sampler, num_workers=2, pin_memory=False
        )
        self.val_loader = DataLoader(
            self.val_dataset, batch_size=batch_size, num_workers=2, pin_memory=False
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
        self.scheduler = ReduceLROnPlateau(self.optimizer, mode='min', factor=0.5, patience=2)  # 학습률 스케줄러
        self.criterion = FocalLoss(ignore_index=self.model.color_head.out_features - 1)  # Focal Loss
        self.scaler = torch.amp.GradScaler() if self.device == "cuda" else None  # 혼합정밀도 스케일러
        self.patience = patience  # 조기 종료 patience

    def train_epoch(self):
        # 1 에폭 학습
        self.model.train()
        total_loss, total_count = 0, 0
        topk_correct = {1: 0, 3: 0, 5: 0}
        for batch in tqdm(self.train_loader, desc="Train", leave=False):
            images = batch["image"].to(self.device, non_blocking=True)  # 이미지 텐서
            texts = batch["text"]  # 텍스트 리스트
            text_tokens = clip.tokenize(texts).to(self.device, non_blocking=True)  # 토큰화
            color_labels, object_labels = [], []
            for i in range(len(batch["type"])):
                if batch["type"][i] == "combined":
                    color_labels.append(self.model.color_head.out_features - 1)  # 조합 데이터는 무시 인덱스
                    object_labels.append(self.model.object_head.out_features - 1)
                elif batch["type"][i] == "color":
                    color_labels.append(self.dataset.color_classes.index(batch["label"][i]))
                    object_labels.append(self.model.object_head.out_features - 1)
                else:
                    color_labels.append(self.model.color_head.out_features - 1)
                    object_labels.append(self.dataset.object_classes.index(batch["label"][i]))
            color_labels = torch.tensor(color_labels, device=self.device)
            object_labels = torch.tensor(object_labels, device=self.device)

            if self.device == "cuda":
                with torch.amp.autocast(self.device):  # 혼합정밀도
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

            # color 평가
            color_mask = color_labels != (self.model.color_head.out_features - 1)
            if color_mask.any():
                color_logits_valid = color_logits[color_mask]
                color_labels_valid = color_labels[color_mask]
                total_count += len(color_labels_valid)
                for k in [1, 3, 5]:
                    topk = min(k, color_logits_valid.shape[1])
                    color_topk = color_logits_valid.topk(topk, dim=1).indices
                    color_label_expand = color_labels_valid.unsqueeze(1).expand(-1, topk)
                    color_match = (color_topk == color_label_expand).any(dim=1)
                    topk_correct[k] += (color_match >= 1).sum().item()

            # object 평가
            object_mask = object_labels != (self.model.object_head.out_features - 1)
            if object_mask.any():
                object_logits_valid = object_logits[object_mask]
                object_labels_valid = object_labels[object_mask]
                total_count += len(object_labels_valid)
                for k in [1, 3, 5]:
                    topk = min(k, object_logits_valid.shape[1])
                    object_topk = object_logits_valid.topk(topk, dim=1).indices
                    object_label_expand = object_labels_valid.unsqueeze(1).expand(-1, topk)
                    object_match = (object_topk == object_label_expand).any(dim=1)
                    topk_correct[k] += (object_match >= 1).sum().item()
                total_loss += loss.item()

        avg_loss = total_loss / len(self.train_loader)
        topk_acc = {k: (topk_correct[k] / total_count if total_count > 0 else 0) for k in [1, 3, 5]}
        return avg_loss, topk_acc

    def validate_epoch(self):
        # 1 에폭 검증
        self.model.eval()
        total_loss, total_count = 0, 0
        topk_correct = {1: 0, 3: 0, 5: 0}
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Val", leave=False):
                images = batch["image"].to(self.device, non_blocking=True)
                texts = batch["text"]
                text_tokens = clip.tokenize(texts).to(self.device, non_blocking=True)
                color_labels, object_labels = [], []
                for i in range(len(batch["type"])):
                    if batch["type"][i] == "combined":
                        color_labels.append(self.model.color_head.out_features - 1)
                        object_labels.append(self.model.object_head.out_features - 1)
                    elif batch["type"][i] == "color":
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

                # color 평가
                color_mask = color_labels != (self.model.color_head.out_features - 1)
                if color_mask.any():
                    color_logits_valid = color_logits[color_mask]
                    color_labels_valid = color_labels[color_mask]
                    total_count += len(color_labels_valid)
                    for k in [1, 3, 5]:
                        topk = min(k, color_logits_valid.shape[1])
                        color_topk = color_logits_valid.topk(topk, dim=1).indices
                        color_label_expand = color_labels_valid.unsqueeze(1).expand(-1, topk)
                        color_match = (color_topk == color_label_expand).any(dim=1)
                        topk_correct[k] += (color_match == 1).sum().item()

                # object 평가
                object_mask = object_labels != (self.model.object_head.out_features - 1)
                if object_mask.any():
                    object_logits_valid = object_logits[object_mask]
                    object_labels_valid = object_labels[object_mask]
                    total_count += len(object_labels_valid)
                    for k in [1, 3, 5]:
                        topk = min(k, object_logits_valid.shape[1])
                        object_topk = object_logits_valid.topk(topk, dim=1).indices
                        object_label_expand = object_labels_valid.unsqueeze(1).expand(-1, topk)
                        object_match = (object_topk == object_label_expand).any(dim=1)
                        topk_correct[k] += (object_match == 1).sum().item()
                    total_loss += loss.item()

            avg_loss = total_loss / len(self.val_loader)
            topk_acc = {k: (topk_correct[k] / total_count if total_count > 0 else 0) for k in [1, 3, 5]}
            return avg_loss, topk_acc

    def train(self, epochs):
        # 전체 학습 루프
        best_loss = float('inf')
        patience_counter = 0
        train_losses, train_accs = [], []
        val_losses, val_accs = [], []

        for epoch in range(epochs):
            train_loss, train_topk = self.train_epoch()
            val_loss, val_topk = self.validate_epoch()
            print(
                f"Epoch {epoch+1}/{epochs} | "
                f"Train Loss={train_loss:.4f}, Top1={train_topk[1]:.4f}, Top3={train_topk[3]:.4f}, Top5={train_topk[5]:.4f} | "
                f"Val Loss={val_loss:.4f}, Top1={val_topk[1]:.4f}, Top3={val_topk[3]:.4f}, Top5={val_topk[5]:.4f}"
            )
            self.scheduler.step(val_loss)
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            train_accs.append(train_topk[3])  # Top-3 accuracy만 저장
            val_accs.append(val_topk[3])
            if val_loss < best_loss:
                best_loss = val_loss
                save_path = os.path.join(BASE_DIR, "data", "best_clip_model_example.pt")
                os.makedirs(os.path.dirname(save_path), exist_ok=True)  # 폴더가 없으면 생성
                torch.save(self.model.state_dict(), save_path)
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break

        # 학습/검증 Loss, Accuracy 그래프 그리기
        epochs_range = range(1, epochs + 1)
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(epochs_range, train_losses, label='Train Loss')
        plt.plot(epochs_range, val_losses, label='Val Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Loss Curve')
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(epochs_range, train_accs, label='Train Acc')
        plt.plot(epochs_range, val_accs, label='Val Acc')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title('Accuracy Curve')
        plt.legend()
        plt.tight_layout()
        plt.show()

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
    # t-SNE 기반 임베딩 시각화
    def visualize_features_2d(self, dataloader, title=""):
        self.model.eval()
        
        # 각 분류 작업별 특성 및 라벨 저장
        color_features = []
        object_features = []
        combined_features = []
        all_color_labels = []
        all_object_labels = []
        all_combined_labels = []
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Extracting features", leave=False):
                images = batch["image"].to(self.device, non_blocking=True)
                texts = batch["text"]
                text_tokens = clip.tokenize(texts).to(self.device, non_blocking=True)
                
                # 모델 출력 (분류기 logits와 임베딩)
                color_logits, object_logits, img_emb, txt_emb = self.model(images, text_tokens)
                
                # 각 분류 작업의 특성 저장
                color_features.append(color_logits.cpu().numpy())    # 색상 분류기 출력
                object_features.append(object_logits.cpu().numpy())  # 객체 분류기 출력
                combined_features.append(img_emb.cpu().numpy())      # 이미지 임베딩 (combined용)
                
                # 라벨 저장
                for i in range(len(batch["type"])):
                    if batch["type"][i] == "color":
                        all_color_labels.append(batch["label"][i])
                        all_object_labels.append("")
                        all_combined_labels.append("")
                    elif batch["type"][i] == "object":
                        all_color_labels.append("")
                        all_object_labels.append(batch["label"][i])
                        all_combined_labels.append("")
                    elif batch["type"][i] == "combined":
                        all_color_labels.append(batch.get("color", [""])[i])
                        all_object_labels.append(batch.get("object", [""])[i])
                        all_combined_labels.append(batch["label"][i])
                    else:
                        all_color_labels.append("")
                        all_object_labels.append("")
                        all_combined_labels.append("")
        
        # 특성 합치기
        color_features = np.concatenate(color_features)
        object_features = np.concatenate(object_features)
        combined_features = np.concatenate(combined_features)
        
        # 라벨을 numpy 배열로 변환
        all_color_labels = np.array(all_color_labels)
        all_object_labels = np.array(all_object_labels)
        all_combined_labels = np.array(all_combined_labels)
        
        def plot_classification_tsne(features, labels, label_name, exclude_labels=None):
            if exclude_labels is None:
                exclude_labels = set(["", "기타", None, "nan"])
            
            # 유효한 라벨만 필터링
            valid_mask = np.array([l not in exclude_labels for l in labels])
            if np.sum(valid_mask) == 0:
                print(f"No valid labels for {label_name}")
                return
                
            filtered_features = features[valid_mask]
            filtered_labels = labels[valid_mask]
            unique_labels = sorted(set(filtered_labels))
            
            if len(unique_labels) == 0 or len(filtered_features) < 5:
                print(f"Not enough data for {label_name} t-SNE")
                return
            
            print(f"[{label_name}] 유효한 샘플: {len(filtered_features)}개, 라벨 종류: {len(unique_labels)}개")
            print(f"[{label_name}] 라벨: {unique_labels}")
            
            # t-SNE 수행 (각 분류별로 다른 특성 공간에서)
            perplexity = min(30, len(filtered_features) // 4)
            tsne = TSNE(n_components=2, random_state=42, perplexity=max(5, perplexity))
            reduced = tsne.fit_transform(filtered_features)
            
            # 시각화
            cmap = plt.get_cmap('tab20', max(20, len(unique_labels)))
            plt.figure(figsize=(12, 10))
            
            for i, label in enumerate(unique_labels):
                idx = filtered_labels == label
                if np.sum(idx) > 0:
                    plt.scatter(reduced[idx, 0], reduced[idx, 1], 
                            c=[cmap(i)], label=label, alpha=0.7, s=50)
            
            plt.title(f"2D t-SNE: {label_name} 분류 결과 ({title})")
            plt.xlabel("t-SNE Dimension 1")
            plt.ylabel("t-SNE Dimension 2")
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
            plt.tight_layout()
            plt.show()
    
        # 각 분류 작업별로 서로 다른 특성 공간에서 t-SNE 수행
        print("Color 분류 결과 t-SNE 시각화")
        plot_classification_tsne(color_features, all_color_labels, "Color")
        
        print("Object 분류 결과 t-SNE 시각화")
        plot_classification_tsne(object_features, all_object_labels, "Object")
        
        print("Combined 분류 결과 t-SNE 시각화")
        plot_classification_tsne(combined_features, all_combined_labels, "Combined")

    def faiss_search(self, image_dataloader, text_dataloader, k=3, batch_per_fig=3):
        # 텍스트→이미지 검색 및 Top-3(2개 이상) 정확도 평가
        image_features, image_labels, _ = self.extract_features(image_dataloader, mode="image")
        text_features, text_labels, text_strs = self.extract_features(text_dataloader, mode="text")
        dataset = image_dataloader.dataset
        if hasattr(dataset, "dataset") and hasattr(dataset, "indices"):
            image_paths = [dataset.dataset.data[i]["path"] for i in dataset.indices]
            image_labels = [dataset.dataset.data[i]["label"] for i in dataset.indices]
        else:
            image_paths = [item["path"] for item in dataset.data]
            image_labels = [item["label"] for item in dataset.data]
        index = faiss.IndexFlatIP(image_features.shape[1])
        faiss.normalize_L2(image_features)
        faiss.normalize_L2(text_features)
        index.add(image_features.astype(np.float32))
        D, I = index.search(text_features.astype(np.float32), k)

        total = len(text_labels)
        sample_size = total
        indices = np.random.choice(total, sample_size, replace=False)
        indices = sorted(indices)

        correct = 0
        num_batches = math.ceil(sample_size / batch_per_fig)
        for batch_idx in range(num_batches):
            start = batch_idx * batch_per_fig
            end = min((batch_idx + 1) * batch_per_fig, sample_size)
            batch_indices = indices[start:end]
            fig, axes = plt.subplots(len(batch_indices), k, figsize=(3*k, 3*len(batch_indices)))
            if len(batch_indices) == 1:
                axes = np.expand_dims(axes, 0)
            if k == 1:
                axes = np.expand_dims(axes, 1)
            for row_idx, idx in enumerate(batch_indices):
                text = text_strs[idx]
                gt_label = text_labels[idx]
                idxs = I[idx]
                retrieved_labels = [image_labels[i] for i in idxs]
                gt_label_count = retrieved_labels.count(gt_label)
                is_correct = gt_label_count >= 2
                if is_correct:
                    correct += 1
                for col_idx, i in enumerate(idxs):
                    img = Image.open(image_paths[i]).convert("RGB")
                    ax = axes[row_idx, col_idx]
                    ax.imshow(img)
                    ax.set_title(f"{text}\n{col_idx+1}위: {image_labels[i]}\n유사도={D[idx][col_idx]:.4f}")
                    ax.axis("off")
                axes[row_idx, 0].set_ylabel(f"Q{row_idx+1+start}:\n{text}", fontsize=10)
            plt.tight_layout()
            plt.show()

        topk_acc = correct / sample_size if sample_size > 0 else 0
        print(f"\nFAISS Accuracy ({sample_size}개): {topk_acc:.4f}")

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
    trainer.train(epochs=30)  # 30회 반복

    class_info = {
        "color_classes": trainer.dataset.color_classes,
        "object_classes": trainer.dataset.object_classes,
    }
    os.makedirs(os.path.join(BASE_DIR, "data"), exist_ok=True)
    with open(os.path.join(BASE_DIR, "data", "clip_classes.json"), "w", encoding="utf-8") as f:
        json.dump(class_info, f, ensure_ascii=False, indent=2)  # 클래스 정보 저장

    print("2D t-SNE 시각화 (color)")
    trainer.visualize_features_2d(trainer.val_loader)


    print("FAISS 기반 텍스트→이미지 검색")
    trainer.faiss_search(trainer.val_loader, trainer.val_loader, k=3)
    
    