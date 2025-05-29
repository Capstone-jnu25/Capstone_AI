import requests
import logging
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

# 세션에 리트라이(재시도) 정책 적용
def get_requests_session():
    session = requests.Session()
    retries = Retry(
        total=5,
        backoff_factor=0.3,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET", "POST"]
    )
    adapter = HTTPAdapter(max_retries=retries)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session

session = get_requests_session()

def fetch_posts_from_api(get_api_url):
    try:
        response = session.get(get_api_url, timeout=5)
        response.raise_for_status()
        posts = response.json()
        logging.info(f"게시글 {len(posts)}건 API로부터 수신 성공")
        return posts
    except Exception as e:
        logging.error(f"API 요청 실패: {e}", exc_info=True)
        return []