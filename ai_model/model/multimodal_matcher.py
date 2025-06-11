import os
import sys
import torch
import numpy as np
import json
import logging
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from typing import List, Optional
from PIL import Image

import open_clip  # 추가
from .keyword_extractor import StopwordAwareTFIDF
from .clip_model import CLIPTrainer
import faiss

app = FastAPI()

# ----- 설정 및 모델 로드 -----
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "data", "best_clip_model.pt")
TFIDF_MODEL_PATH = os.path.join(BASE_DIR, "data", "tfidf_model.joblib")
TEMP_DIR = "/tmp"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

keyword_extractor = StopwordAwareTFIDF()
keyword_extractor.load_model(TFIDF_MODEL_PATH)

trainer = CLIPTrainer(
    color_dir=os.path.join(BASE_DIR, "dataset", "colors"),
    object_dir=os.path.join(BASE_DIR, "dataset", "objects"),
    combined_dir=os.path.join(BASE_DIR, "dataset", "combined"),
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

async def save_temp_file(file: UploadFile) -> str:
    temp_path = os.path.join(TEMP_DIR, file.filename)
    with open(temp_path, "wb") as f:
        content = await file.read()
        f.write(content)
    return temp_path

def cleanup_temp_files(file_paths):
    for path in file_paths:
        if path and os.path.exists(path):
            try:
                os.remove(path)
            except Exception as e:
                logger.error(f"임시 파일 삭제 오류: {str(e)}")

def search_similar(query_emb, index, ids, top_k=3):
    if index is None or query_emb is None:
        return []
    query_emb = query_emb.reshape(1, -1).astype('float32')
    k = min(top_k, index.ntotal)
    if k == 0:
        return []
    D, I = index.search(query_emb, k)
    return [ids[i] for i in I[0]]

# ----- API -----
@app.post("/ai/recommend")
async def multimodal_recommend(
    newPost: Optional[str] = Form(None),
    newImage: Optional[UploadFile] = File(None),
    existingPosts: str = Form(...),
    existingImages: List[UploadFile] = File(...)
):
    temp_files = []
    try:
        # 기존 게시글 정보 준비
        existing_posts = json.loads(existingPosts)
        existing_post_ids = [post["postId"] for post in existing_posts]
        existing_keywords = [post["keywords"] for post in existing_posts]

        # 기존 이미지 임베딩
        existing_img_embeddings, existing_img_paths = [], []
        for img_file in existingImages:
            temp_path = await save_temp_file(img_file)
            temp_files.append(temp_path)
            existing_img_paths.append(temp_path)
            emb = process_image(temp_path)
            existing_img_embeddings.append(emb)

        # 기존 텍스트 임베딩
        existing_text_embeddings = []
        for keywords in existing_keywords:
            text = " ".join(keywords)
            emb = process_text(text)
            existing_text_embeddings.append(emb)

        # FAISS 인덱스 생성
        img_index = create_faiss_index(existing_img_embeddings)
        text_index = create_faiss_index(existing_text_embeddings)

        recommended_ids = []
        extracted_keywords = []

        # 1. 새 게시글 + 이미지 (텍스트+이미지)
        if newPost and newImage:
            post_data = json.loads(newPost)
            text = f"{post_data.get('title', '')} {post_data.get('contents', '')}"
            extracted_keywords = keyword_extractor.extract_keywords(text)
            text_query = " ".join(extracted_keywords) if extracted_keywords else text
            text_emb = process_text(text_query)

            new_img_path = await save_temp_file(newImage)
            temp_files.append(new_img_path)
            img_emb = process_image(new_img_path)

            # 1) 새 이미지 ↔ 기존 이미지
            img_img_ids = search_similar(img_emb, img_index, existing_post_ids)
            # 2) 새 텍스트 ↔ 기존 이미지
            text_img_ids = search_similar(text_emb, img_index, existing_post_ids)

            # 결과 통합 (중복 제거, 우선순위: img_img > text_img)
            all_ids = []
            for id_list in [img_img_ids, text_img_ids]:
                for pid in id_list:
                    if pid not in all_ids:
                        all_ids.append(pid)
            recommended_ids = all_ids[:3]

        # 2. 새 게시글(텍스트)만
        elif newPost and not newImage:
            post_data = json.loads(newPost)
            text = f"{post_data.get('title', '')} {post_data.get('contents', '')}"
            extracted_keywords = keyword_extractor.extract_keywords(text)
            new_text_emb = process_text(" ".join(extracted_keywords))

            # 텍스트 → 기존 이미지 임베딩 유사도만 사용
            recommended_ids = search_similar(new_text_emb, img_index, existing_post_ids)

        # 3. 새 이미지만
        elif not newPost and newImage:
            new_img_path = await save_temp_file(newImage)
            temp_files.append(new_img_path)
            img_emb = process_image(new_img_path)

            # 1) 새 이미지 ↔ 기존 이미지
            img_img_ids = search_similar(img_emb, img_index, existing_post_ids)
            # 2) 새 이미지 ↔ 기존 텍스트
            img_text_ids = search_similar(img_emb, text_index, existing_post_ids)

            # 결과 통합 (중복 제거, 우선순위: img_img > img_text)
            all_ids = []
            for id_list in [img_img_ids, img_text_ids]:
                for pid in id_list:
                    if pid not in all_ids:
                        all_ids.append(pid)
            recommended_ids = all_ids[:3]

        else:
            raise HTTPException(400, "텍스트나 이미지 중 하나는 필수입니다.")

        return {
            "recommended_ids": recommended_ids,
            "keywords": extracted_keywords
        }
    except Exception as e:
        logger.error(f"추천 처리 실패: {str(e)}")
        raise HTTPException(500, f"추천 처리 중 오류 발생: {str(e)}")
    finally:
        cleanup_temp_files(temp_files)
