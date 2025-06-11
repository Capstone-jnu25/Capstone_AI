import os
import sys
import torch
import numpy as np
import json
import logging
from PIL import Image
from typing import List, Optional

import open_clip  # 추가
from .keyword_extractor import StopwordAwareTFIDF
from .clip_model import CLIPTrainer
import faiss

# ----- 설정 및 모델 로드 -----
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_PATH = os.path.join(DATA_DIR, "best_clip_model.pt")
TFIDF_MODEL_PATH = os.path.join(DATA_DIR, "tfidf_model.joblib")
DATASET_DIR = os.path.join(DATA_DIR, "dataset")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

keyword_extractor = StopwordAwareTFIDF()
keyword_extractor.load_model(TFIDF_MODEL_PATH)

trainer = CLIPTrainer(
    color_dir=os.path.join(DATASET_DIR, "colors"),
    object_dir=os.path.join(DATASET_DIR, "objects"),
    combined_dir=os.path.join(DATASET_DIR, "combined"),
    device=DEVICE,
    batch_size=32,
    sample_ratio=1.0,
    patience=5
)
trainer.model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
trainer.model.eval()

# --- open_clip 토크나이저 준비 ---
tokenizer = open_clip.get_tokenizer('ViT-B-32')

# ----- 임베딩 유틸리티 -----
def process_image(image_path: str):
    image = Image.open(image_path).convert("RGB")
    image_tensor = trainer.dataset.preprocess(image).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        img_emb = trainer.model.clip.encode_image(image_tensor)
        img_emb = img_emb / img_emb.norm(dim=-1, keepdim=True)
    return img_emb.cpu().numpy()[0]

def process_text(text: str):
    with torch.no_grad():
        text_token = tokenizer([text])  # open_clip 토크나이저 사용
        text_token = text_token.to(DEVICE)
        text_emb = trainer.model.clip.encode_text(text_token)
        text_emb = text_emb / text_emb.norm(dim=-1, keepdim=True)
    return text_emb.cpu().numpy()[0]

def create_faiss_index(embeddings):
    if not embeddings or len(embeddings) == 0:
        return None
    embs = np.stack(embeddings).astype('float32')
    index = faiss.IndexFlatIP(embs.shape[1])
    index.add(embs)
    return index

def search_similar(query_emb, index, ids, top_k=3):
    if index is None or query_emb is None:
        return []
    query_emb = query_emb.reshape(1, -1).astype('float32')
    k = min(top_k, index.ntotal)
    if k == 0:
        return []
    D, I = index.search(query_emb, k)
    return [ids[i] for i in I[0]]

# ----- 테스트 데이터 불러오기 -----
def load_test_data():
    # (이하 기존 코드와 동일)
    new_posts = []
    new_images = []
    existing_posts = None
    existing_images = []

    image_dir = r"C:\Users\answn\.conda\capstone_project\final_test_data2"
    existing_image_files = [os.path.join(image_dir, f"{i}.jpg") for i in range(1, 16)]

    # 1. 텍스트+이미지
    new_posts.append({
        "postId": 21,
        "title": "검은색 카드지갑",
        "contents": "제가 AI융합관에서 검은색 카드지갑을 잃어버렸습니다. 혹시 보신 분 있으시면 연락주세요!"
    })
    new_images.append(os.path.join(r"C:\Users\answn\.conda\capstone_project\final_test_data1", "1.jpg"))

    # 2. 텍스트만
    new_posts.append({
        "postId": 22,
        "title": "카드지갑",
        "contents": "상대쪽에서 카드지갑을 잃어버렸어요. 검은색이에요"
    })
    new_images.append(None)

    # 3. 이미지만
    new_posts.append(None)
    new_images.append(os.path.join(r"C:\Users\answn\.conda\capstone_project\final_test_data1", "2.jpg"))

    with open(r"C:\Users\answn\.conda\capstone_project\final_test_data2.txt", "r", encoding="utf-8") as f:
        existing_posts = json.load(f)

    return new_posts, new_images, existing_posts, existing_image_files

# ----- 테스트 실행 -----
def run_recommendation_test():
    new_posts, new_images, existing_posts, existing_image_files = load_test_data()

    existing_post_ids = [post["postId"] for post in existing_posts]
    existing_keywords = []
    for post in existing_posts:
        text = f"{post.get('title', '')} {post.get('contents', '')}"
        keywords = keyword_extractor.extract_keywords(text)
        existing_keywords.append(keywords)

    existing_img_embeddings = []
    for img_path in existing_image_files:
        emb = process_image(img_path)
        existing_img_embeddings.append(emb)

    existing_text_embeddings = []
    for keywords in existing_keywords:
        text = " ".join(keywords)
        emb = process_text(text)
        existing_text_embeddings.append(emb)

    img_index = create_faiss_index(existing_img_embeddings)
    text_index = create_faiss_index(existing_text_embeddings)

    for idx, (new_post, new_image) in enumerate(zip(new_posts, new_images)):
        print(f"\n===== 테스트 케이스 {idx+1} =====")
        recommended_ids = []
        extracted_keywords = []

        # 1. 텍스트+이미지 케이스
        if new_post and new_image:
            text = f"{new_post.get('title', '')} {new_post.get('contents', '')}"
            extracted_keywords = keyword_extractor.extract_keywords(text)
            
            # 텍스트 임베딩 생성
            text_emb = process_text(" ".join(extracted_keywords))
            
            # 이미지 임베딩 생성
            img_emb = process_image(new_image)

            # 이미지-이미지 유사도 검색
            img_img_ids = search_similar(img_emb, img_index, existing_post_ids)
            
            # 텍스트-이미지 유사도 검색
            text_img_ids = search_similar(text_emb, img_index, existing_post_ids)

            # 결과 통합 (중복 제거)
            all_ids = []
            for id_list in [img_img_ids, text_img_ids]:
                for pid in id_list:
                    if pid not in all_ids:
                        all_ids.append(pid)
            recommended_ids = all_ids[:3]

        # 2. 텍스트만 케이스
        elif new_post and not new_image:
            text = f"{new_post.get('title', '')} {new_post.get('contents', '')}"
            extracted_keywords = keyword_extractor.extract_keywords(text)
            new_text_emb = process_text(" ".join(extracted_keywords))

            # 텍스트-이미지 유사도 검색
            if img_index is not None:
                recommended_ids = search_similar(new_text_emb, img_index, existing_post_ids)
            else:
                recommended_ids = []

        # 3. 이미지만 케이스
        elif not new_post and new_image:
            img_emb = process_image(new_image)

            # 이미지-이미지 유사도 검색
            img_img_ids = search_similar(img_emb, img_index, existing_post_ids)
            
            # 이미지-텍스트 유사도 검색
            img_text_ids = search_similar(img_emb, text_index, existing_post_ids)

            # 결과 통합 (중복 제거)
            all_ids = []
            for id_list in [img_img_ids, img_text_ids]:
                for pid in id_list:
                    if pid not in all_ids:
                        all_ids.append(pid)
            recommended_ids = all_ids[:3]

        else:
            print("잘못된 테스트 케이스입니다.")
            continue

        print("추천 결과:", recommended_ids)
        print("추출 키워드:", extracted_keywords)

if __name__ == "__main__":
    run_recommendation_test()
