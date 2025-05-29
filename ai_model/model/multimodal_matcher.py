import os
import torch
import numpy as np
import requests
import logging
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional
from contextlib import asynccontextmanager
from PIL import Image
from .keyword_extractor import StopwordAwareTFIDF
from .clip_model import CLIPTrainer
import faiss
from sklearn.feature_extraction.text import TfidfVectorizer

app = FastAPI()

# ----- 설정 -----
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "data", "best_clip_model.pt")
TFIDF_MODEL_PATH = os.path.join(BASE_DIR, "data", "tfidf_model.joblib")
BACKEND_API_URL = "http://13.124.71.212:8080/api/lost-board"  # 실제 백엔드 API 주소

# ----- 키워드 추출기 및 모델 로드 -----
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

# ----- 입력/출력 데이터 모델 -----
class PostRequest(BaseModel):
    id: int
    title: str = ""
    content: str = ""
    image_path: Optional[str] = None

class RecommendResponse(BaseModel):
    recommended_ids: List[int]
    keywords: List[str]

# ----- 게시글 데이터 API에서 받아오기 -----
def fetch_posts_from_api(api_url):
    try:
        response = requests.get(api_url, timeout=5)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        print(f"API 요청 실패: {e}")
        return []

def build_post_embeddings(posts, text_query_emb=None):
    """
    posts: 게시글 리스트
    text_query_emb: 텍스트 쿼리 임베딩(numpy array), 텍스트로 이미지 검색 시 사용
    """
    embeddings = []
    ids = []
    keywords_list = []
    text_corpus = []
    for post in posts:
        text = (post.get("title", "") or "") + " " + (post.get("content", "") or "")
        keywords = keyword_extractor.extract_keywords(text)
        keywords_list.append(keywords)
        text_corpus.append(text)
        image_path = post.get("image_path", "")
        if image_path and not os.path.isabs(image_path):
            image_path = os.path.join(BASE_DIR, image_path)
        if image_path and os.path.exists(image_path):
            image = Image.open(image_path).convert("RGB")
            image_tensor = trainer.dataset.preprocess(image).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                img_emb = trainer.model.clip.encode_image(image_tensor)
                img_emb = img_emb / img_emb.norm(dim=-1, keepdim=True)
            img_emb_np = img_emb.cpu().numpy()[0]
            embeddings.append(img_emb_np)
            ids.append(post["id"])
        else:
            embeddings.append(None)
            ids.append(post["id"])
    return embeddings, ids, keywords_list, text_corpus

def create_faiss_index(embeddings):
    if not embeddings:
        return None
    embs = np.stack(embeddings).astype('float32')
    index = faiss.IndexFlatIP(embs.shape[1])
    index.add(embs)
    return index

# ----- 서버 시작 시 FAISS 인덱스 사전 계산 -----
faiss_index = None
db_ids = []

@asynccontextmanager
async def lifespan(app: FastAPI):
    global faiss_index, db_ids, keyword_extractor, trainer
    # Startup: 서버 시작 시 FAISS 인덱스 미리 빌드
    posts = fetch_posts_from_api(BACKEND_API_URL)
    embeddings, ids, keyword_extractor, trainer = build_post_embeddings(posts)
    faiss_index = create_faiss_index(embeddings)
    db_ids = ids
    logging.info(f"FAISS 인덱스 빌드 완료: {len(db_ids)}개 게시글")
    yield
    # (필요하다면 종료시 리소스 정리 코드 추가)

app = FastAPI(lifespan=lifespan)

# ----- 추천 API -----
@app.post("/ai/recommend", response_model=RecommendResponse)
def ai_recommend(post: PostRequest):
    # 1. 백엔드 API에서 게시글 데이터 받아오기
    posts_db = fetch_posts_from_api(BACKEND_API_URL)

    # 2. 키워드 추출
    text = (post.title or "") + " " + (post.content or "")
    keywords = keyword_extractor.extract_keywords(text)

    # 3. 전체 게시글 임베딩/키워드 준비
    embeddings, ids, keywords_list, text_corpus = build_post_embeddings(posts_db)

    # 4. 입력 이미지 임베딩 추출
    if post.image_path:
        image_path = post.image_path
        if not os.path.isabs(image_path):
            image_path = os.path.join(BASE_DIR, image_path)
        if os.path.exists(image_path):
            image = Image.open(image_path).convert("RGB")
            image_tensor = trainer.dataset.preprocess(image).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                query_emb = trainer.model.clip.encode_image(image_tensor)
                query_emb = query_emb / query_emb.norm(dim=-1, keepdim=True)
            query_emb_np = query_emb.cpu().numpy()[0].astype('float32').reshape(1, -1)

            # 5. FAISS 인덱스 생성 (임베딩이 있는 게시글만)
            valid_indices = [i for i, emb in enumerate(embeddings) if emb is not None]
            if not valid_indices:
                return RecommendResponse(recommended_ids=[], keywords=keywords)
            db_embs = np.stack([embeddings[i] for i in valid_indices]).astype('float32')
            db_ids = [ids[i] for i in valid_indices]

            index = faiss.IndexFlatIP(db_embs.shape[1])
            index.add(db_embs)
            D, I = index.search(query_emb_np, min(4, len(db_ids)))
            top_idx = I[0][1:4] if len(db_ids) > 1 else I[0][:1]
            recommended_ids = [db_ids[i] for i in top_idx]
            return RecommendResponse(recommended_ids=recommended_ids, keywords=keywords)
        else:
            # 이미지 경로가 있지만 파일이 없으면 텍스트로 이미지 검색 -> 6번으로 이동
            pass

    # 6. 이미지가 없거나 임베딩 불가 시: 텍스트로 이미지 검색 (CLIP 텍스트 인코더 활용)
    # 텍스트 쿼리 임베딩 추출
    if keywords:
        # 키워드들을 언더스코어로 연결 (예: 빨간_지갑)
        keyword_query = " ".join(keywords)
    else:
        keyword_query = text  # 키워드가 없으면 전체 텍스트 사용

    with torch.no_grad():
        text_token = trainer.model.clip.tokenize([keyword_query]).to(DEVICE)
        text_emb = trainer.model.clip.encode_text(text_token)
        text_emb = text_emb / text_emb.norm(dim=-1, keepdim=True)
        text_emb_np = text_emb.cpu().numpy().astype('float32')

    # 이미지 임베딩이 있는 게시글만 대상으로 FAISS 검색
    valid_indices = [i for i, emb in enumerate(embeddings) if emb is not None]
    if not valid_indices:
        return RecommendResponse(recommended_ids=[], keywords=keywords)
    db_embs = np.stack([embeddings[i] for i in valid_indices]).astype('float32')
    db_ids = [ids[i] for i in valid_indices]

    index = faiss.IndexFlatIP(db_embs.shape[1])
    index.add(db_embs)
    D, I = index.search(text_emb_np, min(4, len(db_ids)))
    top_idx = I[0][:3]
    recommended_ids = [db_ids[i] for i in top_idx]
    return RecommendResponse(recommended_ids=recommended_ids, keywords=keywords)

