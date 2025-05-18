import os
import torch
import clip
import json
from PIL import Image
from keyword_extractor import StopwordAwareTFIDF
from clip_model import EnhancedCLIPModel
import numpy as np

# ----- 설정 -----
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "best_clip_model.pt")
CLASSES_PATH = os.path.join(BASE_DIR, "data", "clip_classes.json")
TFIDF_MODEL_PATH = os.path.join(BASE_DIR, "data", "tfidf_model.joblib")

# ----- 키워드 추출기 로드 -----
keyword_extractor = StopwordAwareTFIDF()
keyword_extractor.load_model(TFIDF_MODEL_PATH) # TF-IDF 모델 로드

# ----- CLIP 모델 및 클래스 정보 로드 -----
clip_model, preprocess = clip.load("ViT-B/32", device=DEVICE, jit=False)
with open(CLASSES_PATH, "r", encoding="utf-8") as f:
    class_info = json.load(f)
color_classes = class_info["color_classes"]  # 색상 클래스 리스트
object_classes = class_info["object_classes"]  # 물건 클래스 리스트
model = EnhancedCLIPModel(clip_model, color_classes, object_classes).to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

# ----- 게시글 데이터 예시 -----
# DB에서 게시글 리스트를 불러와야 함
# ex) posts = db.query("SELECT id, title, content, image_path FROM posts")
posts = [
    {
        "id": 1,
        "title": "파란색 가방을 잃어버렸어요",
        "content": "지하철에서 파란색 백팩을 분실했습니다.",
        "image_path": "sample_images/bag1.jpg"
    },
    {
        "id": 2,
        "title": "검정색 지갑 찾습니다",
        "content": "작은 검정색 카드지갑을 분실했습니다.",
        "image_path": "sample_images/wallet1.jpg"
    },
    # ... (더 많은 게시글)
]

# ----- 게시글 임베딩 생성 -----
def get_post_embedding(post):
    # 키워드 추출
    text = post["title"] + " " + post["content"]
    keywords = keyword_extractor.extract_keywords(text, top_n=2)
    prompt = " ".join(keywords)
    # 이미지 임베딩
    if post.get("image_path") and os.path.exists(post["image_path"]):
        image = Image.open(post["image_path"]).convert("RGB")
        image_tensor = preprocess(image).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            img_emb = model.clip.encode_image(image_tensor)
            img_emb = img_emb / img_emb.norm(dim=-1, keepdim=True)
        return {"type": "image+text", "embedding": img_emb.cpu().numpy(), "prompt": prompt}
    else:
        # 이미지가 없으면 텍스트 임베딩만 사용
        text_tokens = clip.tokenize([prompt]).to(DEVICE)
        with torch.no_grad():
            txt_emb = model.clip.encode_text(text_tokens)
            txt_emb = txt_emb / txt_emb.norm(dim=-1, keepdim=True)
        return {"type": "text", "embedding": txt_emb.cpu().numpy(), "prompt": prompt}

# ----- 전체 게시글 임베딩 사전 구축 -----
post_embeddings = []
for post in posts:
    emb_info = get_post_embedding(post)
    post_embeddings.append({
        "id": post["id"],
        "type": emb_info["type"],
        "embedding": emb_info["embedding"],
        "prompt": emb_info["prompt"]
    })

# ----- 유사 게시글 추천 함수 -----
def recommend_similar_posts(query_title, query_content, query_image_path=None, top_k=5):
    # 쿼리 키워드 추출
    query_text = query_title + " " + query_content
    query_keywords = keyword_extractor.extract_keywords(query_text, top_n=2)
    query_prompt = " ".join(query_keywords)
    # 쿼리 임베딩 생성
    if query_image_path and os.path.exists(query_image_path):
        image = Image.open(query_image_path).convert("RGB")
        image_tensor = preprocess(image).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            query_emb = model.clip.encode_image(image_tensor)
            query_emb = query_emb / query_emb.norm(dim=-1, keepdim=True)
        query_emb = query_emb.cpu().numpy()
    else:
        text_tokens = clip.tokenize([query_prompt]).to(DEVICE)
        with torch.no_grad():
            query_emb = model.clip.encode_text(text_tokens)
            query_emb = query_emb / query_emb.norm(dim=-1, keepdim=True)
        query_emb = query_emb.cpu().numpy()

    # 코사인 유사도 계산
    sims = []
    for post in post_embeddings:
        emb = post["embedding"]
        sim = float(np.dot(query_emb, emb.T))
        sims.append((post["id"], sim))
    sims.sort(key=lambda x: x[1], reverse=True)
    top_ids = [pid for pid, _ in sims[:top_k]]

    # 실제로는 DB에서 id로 게시글을 다시 조회해야 함
    # ex) recommended_posts = db.query("SELECT * FROM posts WHERE id IN (...)", top_ids)
    recommended_posts = [p for p in posts if p["id"] in top_ids]
    return recommended_posts

if __name__ == "__main__":
    # 예시 쿼리
    query_title = "파란색 백팩을 찾습니다"
    query_content = "어제 분실한 파란색 가방을 찾고 있어요."
    query_image_path = "sample_images/query_bag.jpg"  # 없으면 None

    results = recommend_similar_posts(query_title, query_content, query_image_path, top_k=3)
    print("추천 게시글:")
    for post in results:
        print(f"- {post['title']} ({post['id']})")
        print(f"  내용: {post['content']}")
        print(f"  이미지: {post.get('image_path', '없음')}")
        print()