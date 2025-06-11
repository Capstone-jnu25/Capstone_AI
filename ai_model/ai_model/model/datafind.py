import sys
import io
import os
import re
import time
import random
import requests
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from pathvalidate import sanitize_filename

sys.stdout = io.TextIOWrapper(sys.stdout.detach(), encoding='utf-8')

# 기본 폴더 구조 생성
os.makedirs("dataset/colors", exist_ok=True)
os.makedirs("dataset/objects", exist_ok=True)
os.makedirs("dataset/combined", exist_ok=True)  

def safe_folder_name(name):
    name = sanitize_filename(name)
    name = re.sub(r'_{2,}', '_', name)
    return name.strip('_')[:50]

def create_folder(keyword):
    try:
        if keyword.startswith('color_'):
            _, name = keyword.split('_', 1)
            folder_path = os.path.join("dataset/colors", safe_folder_name(name))
        elif keyword.startswith('object_'):
            _, name = keyword.split('_', 1)
            folder_path = os.path.join("dataset/objects", safe_folder_name(name))
        elif keyword.startswith('combined_'):
            _, color, obj = keyword.split('_', 2)
            combined_folder = f"{safe_folder_name(color)}_{safe_folder_name(obj)}"
            folder_path = os.path.join("dataset/combined", combined_folder)
        else:
            raise ValueError
    except ValueError:
        raise ValueError("키워드는 'color_색상명', 'object_물건명' 또는 'combined_색상_물건' 형식이어야 합니다.")
    os.makedirs(folder_path, exist_ok=True)
    return folder_path

def naver_image_crawler(keywords, max_images=100):
    seen = set()
    keywords = [k for k in keywords if not (k in seen or seen.add(k))]

    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Referer': 'https://www.naver.com/'
    }

    options = webdriver.ChromeOptions()
    options.add_argument("--headless")
    options.add_argument("--disable-gpu")
    options.add_argument("--no-sandbox")
    options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")

    try:
        driver = webdriver.Chrome(
            service=Service(ChromeDriverManager().install()),
            options=options
        )
    except Exception as e:
        print(f"드라이버 초기화 실패: {str(e)}")
        return

    for keyword in keywords:
        print(f"\n[{keyword}] 검색 시작...")
        try:
            folder = create_folder(keyword)
        except Exception as e:
            print(f"폴더 생성 실패: {str(e)}")
            continue

        # 검색어 추출 로직 개선
        try:
            if keyword.startswith('combined_'):
                _, color, obj = keyword.split('_', 2)
                search_query = f"{color} {obj}"
            else:
                search_query = keyword.split('_', 1)[1].strip()
            
            if not search_query:
                print("빈 검색어 → 건너뜀")
                continue
        except IndexError:
            print("잘못된 키워드 형식 → 건너뜀")
            continue

        # 1. requests + BeautifulSoup으로 첫 페이지 썸네일 수집
        url = f"https://search.naver.com/search.naver?where=image&query={search_query}"
        try:
            resp = requests.get(url, headers=headers, timeout=10)
            if resp.status_code != 200:
                print(f"페이지 요청 실패: {resp.status_code}")
                continue
        except Exception as e:
            print(f"페이지 요청 에러: {str(e)}")
            continue

        soup = BeautifulSoup(resp.text, 'html.parser')
        img_tags = soup.select('div.thumb img')
        if not img_tags:
            img_tags = soup.find_all('img')

        img_urls = []
        for img in img_tags:
            img_url = img.get('data-source') or img.get('data-lazy-src') or img.get('src')
            if img_url and img_url.startswith('http'):
                img_urls.append(img_url)
            if len(img_urls) >= max_images:
                break

        # 2. Selenium으로 스크롤 다운하며 추가 이미지 수집
        try:
            driver.get(url)
            time.sleep(random.uniform(1, 3))
            body = driver.find_element(By.TAG_NAME, 'body')
            for _ in range(15):
                body.send_keys(Keys.PAGE_DOWN)
                time.sleep(random.uniform(0.5, 1.5))
            soup_s = BeautifulSoup(driver.page_source, 'html.parser')
            sel_img_tags = soup_s.select('div.thumb img')
            for img in sel_img_tags:
                img_url = img.get('data-source') or img.get('data-lazy-src') or img.get('src')
                if img_url and img_url.startswith('http') and img_url not in img_urls:
                    img_urls.append(img_url)
                if len(img_urls) >= max_images:
                    break
        except Exception as e:
            print(f"Selenium 이미지 추가 수집 실패: {str(e)}")

        if not img_urls:
            print("검색 결과 없음 → 건너뜀")
            continue

        # 이미지 다운로드
        count = 0
        for idx, img_url in enumerate(img_urls):
            if count >= max_images:
                break
            ext_match = re.search(r'\.([a-zA-Z]+)(?=\?|$)', img_url)
            ext = ext_match.group(1).lower() if ext_match else 'jpg'
            if ext not in {'jpg', 'jpeg', 'png'}:
                ext = 'jpg'
            filename = os.path.join(folder, f"{safe_folder_name(search_query)}_{idx+1}.{ext}")
            try:
                img_resp = requests.get(img_url, headers=headers, timeout=10)
                if img_resp.status_code == 200:
                    with open(filename, 'wb') as f:
                        f.write(img_resp.content)
                    count += 1
                    print(f"✓ {keyword} - {count}/{max_images} 저장 완료")
                else:
                    print(f"✗ 이미지 다운로드 실패({img_resp.status_code}): {img_url}")
            except Exception as e:
                print(f"✗ 이미지 저장 에러: {str(e)}")
            time.sleep(random.uniform(0.5, 2))
        print(f"[{keyword}] 검색 완료! 총 {count}개 이미지 저장")

    driver.quit()

# 사용 예시
if __name__ == "__main__":
    search_keywords = ['color_흰색', 'color_골드',
        'color_블루', 'color_아이스 블루','color_하늘색', 
        'color_파란색', 'color_남색','color_빨간색', 'color_노란색', 
        'color_초록색', 'color_녹색','color_연두색','color_보라색', 
        'color_스카이 블루','color_주황색','color_검은색', 'color_회색', 
        'color_로즈골드','color_분홍색', 'color_베이지','color_네이비', 
        'color_청녹색','color_살구색','color_자홍색', 'color_황토색',
        'color_갈색', 'color_은색','color_자주색',
        'color_민트', 'color_라벤더', 
        'object_지갑','object_카드지갑','object_아이폰','object_갤럭시',
        'object_에어팟','object_갤럭시 버즈', 'object_태블릿',
        'object_스마트워치','object_노트북','object_맥북','object_이어폰',
        'object_아이패드','object_패드','object_팔찌', 'object_카드','object_민증',
        'object_목걸이','object_키링','object_노트북 마우스','object_마우스',
        'object_귀걸이','object_필통','object_반지','object_시계','object_헤드셋',
        'object_슬리퍼','object_파우치','object_이어폰 케이스','object_케이스',
        'object_USB','object_운전면허증', 'object_학생증','object_백팩',
        'object_에코백','object_가방',  'object_운동화','object_스니커즈',
        'object_구두','object_슬리퍼','object_샌들', 'object_부츠','object_우산',
        'object_모자','object_장갑', 'object_교통카드',
        'object_비니','object_자켓','object_겉옷','object_코트',
        'object_바람막이','object_과잠','object_후드집업',
        'object_후드티',
        'combined_화이트_갤럭시버즈', 'combined_화이트_에어팟','combined_검은색_백팩',
        'combined_화이트_맥북','combined_실버_헤드셋','combined_화이트_아이패드',
        'combined_민트_파우치','combined_블랙_태블릿','combined_베이지_에코백',
        'combined_하늘색_카드지갑','combined_그레이_태블릿','combined_핑크_마우스',
        'combined_블랙_지갑','combined_블랙_가방','combined_회색_노트북',
        'combined_블랙_스마트워치','combined_청녹색_파우치', 'combined_실버_노트북',
        'combined_검은색_비니','combined_오렌지_에코백','combined_퍼플_카드지갑',
        'combined_스카이블루_태블릿','combined_파란색_지갑', 'combined_민트_아이폰',
        'combined_검정_가방','combined_은색_노트북','combined_분홍색_헤드셋',
        'combined_남색_스마트워치','combined_연두색_파우치','combined_초록색_백팩',
        'combined_블랙_에어팟','combined_핑크_필통''combined_빨간색_우산', 
        'combined_실버_아이폰', 'combined_하얀색_슬리퍼', 'combined_파란색_캡모자',
        'combined_갈색_코트', 'combined_검은색_바람막이', 'combined_빨간색_마우스', 
        'combined_분홍색_장갑','combined_하얀색_스니커즈', 'combined_파란색_비니', 
        'combined_검은색_코트', 'combined_검은색_슬리퍼','combined_검은색_갤럭시버즈', 
        'combined_은색_팔찌', 'combined_은색_귀걸이', 'combined_핑크_파우치'
    ]
    naver_image_crawler(search_keywords, max_images=300)