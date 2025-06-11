import sys                                   # [전체 코드 흐름 개요]
import io                                    # 1. 프로젝트 초기 설정: 데이터 저장 디렉토리 생성 및 기존 데이터 임포트
import os                                    # 2. 경찰청 분실물/습득물 게시판 스크래핑: BeautifulSoup을 이용한 웹 크롤링                                                      
import re                                    # 3. 키워드 추출 및 정제: 한글 텍스트 처리 및 불필요 요소 제거                   
import json                                  # 4. 불용어 업데이트 시스템: 세션 데이터 분석을 통한 동적 불용어 관리
import shutil                                # 5. 데이터 배치 처리: 대량 데이터에 대한 효율적인 메모리 관리
import requests
from collections import defaultdict
from bs4 import BeautifulSoup
from konlpy.tag import Okt

# 윈도우 환경 한글 출력 설정(CMD/PowerShell 한글 깨짐 방지)
sys.stdout = io.TextIOWrapper(sys.stdout.detach(), encoding='utf-8')

# 프로젝트 기본 경로 설정
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # 현재 스크립트의 절대 경로
DATA_DIR = os.path.join(BASE_DIR, "data_example")      # 데이터 저장 디렉토리
SESSIONS_DIR = os.path.join(DATA_DIR, "sessions_example")  # 세션 데이터 저장 경로

"""프로젝트 실행에 필요한 디렉토리 구조 생성"""
def setup_project():
    os.makedirs(DATA_DIR, exist_ok=True)    # 데이터 디렉토리 생성(존재해도 오류 없음)
    os.makedirs(SESSIONS_DIR, exist_ok=True)  # 세션 저장 디렉토리 생성
    print(f"프로젝트 디렉토리 초기화 완료: {DATA_DIR}")

"""외부 세션 데이터를 프로젝트로 복사하는 함수"""
def import_session_files():
    # 원본 데이터가 저장된 경로 목록(불용어 파일을 만들기 위함)
    SOURCE_PATHS = [ 
        r"C:\Users\answn\Downloads\Sample\Sample\data2\session2",
        r"C:\Users\answn\Downloads\Sample\Sample\data2\session3",
        r"C:\Users\answn\Downloads\Sample\Sample\data2\session4",
        r"C:\Users\answn\Downloads\Sample2\Sample\data2\session2",
        r"C:\Users\answn\Downloads\Sample2\Sample\data2\session3",
        r"C:\Users\answn\Downloads\Sample2\Sample\data2\session4"
    ]
    copied_files = []
    for src_dir in SOURCE_PATHS:
        if not os.path.exists(src_dir):
            print(f"⚠️ 원본 경로 없음: {src_dir}")
            continue
        
        # os.walk로 디렉토리 트리 탐색
        for root, _, files in os.walk(src_dir):
            for file in files:
                if file.endswith('.json'):  # JSON 파일만 처리
                    src_path = os.path.join(root, file)
                    dst_path = os.path.join(SESSIONS_DIR, file)
                    
                    # 중복 파일 방지를 위한 복사본 생성
                    if os.path.exists(dst_path):
                        base, ext = os.path.splitext(file)
                        dst_path = os.path.join(SESSIONS_DIR, f"{base}_copy{ext}")
                    shutil.copy2(src_path, dst_path)  # 메타데이터 유지 복사
                    copied_files.append(dst_path)
    print(f"✅ 세션 파일 {len(copied_files)}개 복사 완료")

""" 경찰청 분실물, 습득물 목록 크롤링 클래스 """
class EnhancedLost112Scraper:
    BASE_URLS = {  # 크롤링 대상 URL 
        'lost': 'https://www.lost112.go.kr/lost/lostList.do',
        'found': 'https://www.lost112.go.kr/find/findList.do'
    }
    HEADERS = {  # 웹 크롤링 헤더 설정(보안 우회)
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36...',
        'Referer': 'https://www.lost112.go.kr/',
        'Accept-Language': 'ko-KR,ko;q=0.9,en-US;q=0.8,en;q=0.7',
    }

    def __init__(self, max_pages=100):
        self.session = requests.Session()  # 세션 유지를 위한 객체
        self.session.headers.update(self.HEADERS)  # 공통 헤더 설정
        self.max_pages = max_pages  # 최대 크롤링 페이지 수
        self.police_keywords = set()  # 추출 키워드 저장용
        self.batch_size = 50  # 한 번에 처리할 페이지 묶음 크기

    """HTML에서 게시물 제목 추출"""
    def _extract_titles(self, html: str) -> list:
        soup = BeautifulSoup(html, 'html.parser')
        return [
            item['title']
            for item in soup.select('td[scope="row"][class*="board_title1"][title]')
            if item.has_attr('title')  # title 속성 보유 요소만 필터링
        ]

    """50 페이지 범위 단위로 데이터 처리"""
    def _process_batch(self, page_type: str, start_page: int) -> int:
        batch_total = 0
        for page in range(start_page, start_page + self.batch_size):
            try:
                params = {  # GET 요청 파라미터
                    'pageIndex': page,
                    'searchMain': 'Y',
                    'sortOrder': 'LST'
                }
                response = self.session.get(
                    self.BASE_URLS[page_type],
                    params=params,
                    timeout=15  # 연결 타임아웃 15초
                )
                response.raise_for_status()  # HTTP 에러 검출
                titles = self._extract_titles(response.text)
                new_titles = [t for t in titles if t not in self.police_keywords]
                self.police_keywords.update(new_titles)
                batch_total += len(new_titles)
                if not titles:  # 더 이상 데이터 없는 경우
                    return batch_total
            except Exception as e:
                print(f"  ! {start_page}~{start_page+self.batch_size-1} 페이지 중 {page}번 실패")
                return batch_total
        return batch_total

    """실제 크롤링 실행 메인 메서드"""
    def scrape_titles(self):
        for page_type in ['lost', 'found']:  # 분실물/습득물 병렬 처리
            print(f"\n[{page_type.upper()} 게시판 스크래핑 시작]")
            total_keywords = 0
            # 배치 단위 페이지 처리(1~50, 51~100)
            for batch_start in range(1, self.max_pages + 1, self.batch_size):
                batch_end = min(batch_start + self.batch_size - 1, self.max_pages)
                batch_count = self._process_batch(page_type, batch_start)
                total_keywords += batch_count
                print(f"  → {batch_start}~{batch_end} 페이지: 페이지 {batch_count}개 추가")
                
                if batch_count < self.batch_size:  # 마지막 배치 검출
                    print(f"  ※ 총 {total_keywords} 페이지 처리 완료")
                    break
            print(f"\n  ※ {page_type.upper()} 총 {total_keywords}개 키워드 추출 완료")
            print(f"  ※ 누적 키워드 수: {len(self.police_keywords)}개")

    """크롤링 결과를 파일로 저장"""
    def save_keywords(self, output_path: str):
        exclude_patterns = [  # 제외할 패턴 목록
            r'\d{4}-\d{2}-\d{2}',  # 날짜
            r'\d{2,3}-\d{3,4}-\d{4}',  # 전화번호
            r'^\d+$'  # 순수 숫자
        ]
        unique_words = set()
        for keyword in self.police_keywords:
            if any(re.search(pattern, keyword) for pattern in exclude_patterns):
                continue  # 제외 패턴에 해당하면 스킵
            
            # 복합 명사 분해 처리(예: "빨간지갑" → "빨간", "지갑")
            for word in keyword.split():
                if word and not any(re.search(pattern, word) for pattern in exclude_patterns):
                    unique_words.add(word)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(sorted(unique_words)))

"""동적 불용어 관리 시스템"""
class AdvancedStopwordsUpdater:
    def __init__(self, config: dict):
        self.config = config  # 설정 정보 저장
        self.okt = Okt()  # 한국어 형태소 분석기
        self.phrase_counter = defaultdict(int)  # 어구 빈도 수집
        self.existing_stopwords = set()  # 기존 불용어 집합
        self.police_keywords = set()  # 경찰청 키워드 집합
        self.file_stats = defaultdict(int)  # 파일 처리 통계

    """파일 기반 리소스 로드"""
    def _load_resources(self):
        # 기존 불용어 로드
        if os.path.exists(self.config['stopwords_path']):
            with open(self.config['stopwords_path'], 'r', encoding='utf-8') as f:
                self.existing_stopwords = set(line.strip() for line in f)
        # 경찰청 키워드 로드
        if os.path.exists(self.config['police_keywords_path']):
            with open(self.config['police_keywords_path'], 'r', encoding='utf-8') as f:
                self.police_keywords = set(line.strip() for line in f)

    """텍스트 전처리 파이프라인"""
    def _process_text(self, text: str) -> list:
        cleaned = re.sub(r'[^\w\s]', '', text)  # 특수문자 제거
        tokens = self.okt.morphs(cleaned)  # 형태소 분석
        # 2-gram 어구 생성(예: ["분실", "신고"] → "분실 신고")
        return [' '.join(tokens[i:i+2]) for i in range(len(tokens)-1)]

    """로컬 세션 데이터 분석"""
    def _analyze_local_sessions(self):

        for path in self.config['session_paths']:  # 설정된 경로 순회
            for root, _, files in os.walk(path):
                for file in files:
                    if file.endswith('.json'):  # JSON 파일만 처리
                        file_path = os.path.join(root, file)
                        file_phrase_count = 0
                        with open(file_path, 'r', encoding='utf-8') as f:
                            data = json.load(f)  # JSON 파싱
                            
                            # 세션 내 대화 데이터 처리
                            for session in data.get('sessionInfo', []):
                                for dialog in session.get('dialog', []):
                                    utterance = dialog.get('utterance', '')
                                    phrases = self._process_text(utterance)
                                    file_phrase_count += len(phrases)
                                    
                                    # 어구 빈도 카운팅
                                    for phrase in phrases:
                                        self.phrase_counter[phrase] += 1
                        self.file_stats[file_path] = file_phrase_count

    """불용어 업데이트 전체 프로세스 실행"""
    def execute_pipeline(self):
        self._load_resources()  # 초기 리소스 로드
        self._analyze_local_sessions()  # 데이터 분석
        
        # 새로운 불용어 후보 선정
        new_candidates = {
            phrase
            for phrase, count in self.phrase_counter.items()
            if count >= self.config['threshold']  # 빈도 임계값 이상
            and phrase not in self.police_keywords  # 경찰 키워드 제외
            and phrase not in self.existing_stopwords  # 기존 불용어 제외
        }
        
        # 후보 정제(숫자 전용 단어 제거)
        filtered_candidates = set()
        for phrase in new_candidates:
            if re.match(r'^\d+$', phrase):  # 순수 숫자 패턴 검출
                continue
            # 복합 어구 분해(예: "분실 신고" → "분실", "신고")
            for word in phrase.split():
                filtered_candidates.add(word.strip())
        
        # 파일에 신규 불용어 추가
        with open(self.config['stopwords_path'], 'a', encoding='utf-8') as f:
            f.write('\n'.join(sorted(filtered_candidates)) + '\n')
        
        # 처리 결과 출력
        print(f"\n※ 불필요 용어 업데이트 완료")
        print(f"  - 신규 추가: {len(filtered_candidates)}개")
        print(f"  - 파일 처리량 배치 현황:")
        
        # 200개 파일 단위 배치 처리 통계 출력
        all_files = list(self.file_stats.items())
        batch_size = 200
        total_count = 0
        for i in range(0, len(all_files), batch_size):
            batch = all_files[i:i+batch_size]
            batch_file_count = len(batch)
            batch_word_count = sum(count for _, count in batch)
            total_count += batch_word_count
            print(f"    ▶ 배치 {i//batch_size + 1} ({i+1}~{i+len(batch)}번 파일)")
            print(f"      - 처리 파일: {batch_file_count}개")
            print(f"      - 추출 단어: {batch_word_count}개")
        print(f"\n  ※ 총 처리 현황")
        print(f"    - 전체 파일: {len(all_files)}개")
        print(f"    - 총 단어 수: {total_count}개")

if __name__ == "__main__":
    setup_project()  # 1. 프로젝트 초기화
    import_session_files()  # 2. 데이터 임포트(최초 1회만 실행해야 함!!!!!!!!!!!)
    
    print("\n" + "="*50)
    print("경찰청 데이터 수집 시작")
    scraper = EnhancedLost112Scraper(max_pages=100)
    scraper.scrape_titles()  # 3. 웹 스크래핑 실행
    police_keywords_path = os.path.join(DATA_DIR, "POLICE_KEYWORDS_EXAMPLE.txt")
    scraper.save_keywords(police_keywords_path)  # 4. 결과 저장
    
    print("="*50)
    print("\n불필요 용어 업데이트 시작")
    CONFIG = {  # 설정 파라미터
        "session_paths": [SESSIONS_DIR],
        "stopwords_path": os.path.join(DATA_DIR, "STOPWORDS_EXAMPLE.txt"),
        "police_keywords_path": police_keywords_path,
        "threshold": 3  # 불용어 선정 빈도 임계값
    }
    updater = AdvancedStopwordsUpdater(CONFIG)
    updater.execute_pipeline()  # 5. 불용어 업데이트 실행
    print("="*50)
    
    
