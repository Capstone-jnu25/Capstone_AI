
import sys                                          # [전체 코드 흐름 및 기능 개요]
import io                                           # 1. 주요 기능: 경찰청 분실물 데이터에 특화된 TF-IDF 키워드 추출 시스템
import os                                           # - 불용어(STOPWORDS) 완전 제외 + 경찰 키워드(POLICE_KEYWORDS) 강조
import re                                           # - 색상 정보 자동 감지 및 가중치 부여
import difflib  # 문자열 유사도 비교를 위한 라이브러리 # - 사용자 정의 라벨링 데이터 지원
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer                                                    
from sklearn.model_selection import train_test_split                                                    
from konlpy.tag import Okt                          # 2. 핵심 처리 단계:                                                
import joblib                                       # (1) 데이터 로드: 학습 데이터, 불용어, 경찰 키워드 파일 읽기
                                                    # (2) 텍스트 전처리: 형태소 분석 + 사용자 정의 토크나이징
                                                    # (3) 모델 학습: TF-IDF 가중치 조정 메커니즘 적용
                                                    # (4) 키워드 추출: 계층적 필터링 및 우선순위 정렬
                                                    # (5) 모델 지속화: 학습 결과 파일 저장/로드

                                                    # 3. 주요 특징:
                                                    # - 한국어 특화 처리(Okt 형태소 분석기)
                                                    # - 동적 불용어 관리 시스템
                                                    # - 유사 키워드 검출 기능(difflib 활용)
# CMD/PowerShell 환경에서 한글 출력 설정
sys.stdout = io.TextIOWrapper(sys.stdout.detach(), encoding='utf-8')

# 경찰청 분실물 데이터에 특화된 TF-IDF 처리 클래스
# 주요 기능: 사용자 정의 불용어 처리 ,경찰 키워드 가중치 부스팅, 색상 정보 자동 감지 ,모델 지속화 기능
class StopwordAwareTFIDF:
    def __init__(self, max_features=10000, min_df=2):
        # 모델 파라미터 초기화
        self.max_features = max_features  # 최대 추출 단어 수
        self.min_df = min_df  # 단어 최소 문서 빈도
        self.vectorizer = None  # 텍스트 벡터화 객체
        self.transformer = None  # TF-IDF 변환 객체
        self.stopwords = set()  # 불용어 저장 집합
        self.police_keywords = set()  # 경찰 키워드 저장 집합
        self.feature_names = []  # 특징 이름 목록
        self.okt = Okt()  # 한국어 형태소 분석기 인스턴스
        self.base_dir = os.path.dirname(os.path.abspath(__file__))  # 기본 경로
        self.data_dir = os.path.join(self.base_dir, "data")  # 데이터 저장 경로

    """라벨 파싱 메서드: 학습 데이터에서 라벨 정보 추출"""
    def _parse_labels(self, texts):
        parsed_texts = []
        self.labels = set()  # 라벨 저장 집합
        for line in texts:
            if '|' in line:
                # 라벨 정보가 있는 경우 분리 처리
                text, labels = line.split('|', 1)
                for label in labels.split(','):
                    self.labels.add(label.strip())  # 라벨 추가
                parsed_texts.append(text.strip())
            else:
                parsed_texts.append(line.strip())
        return parsed_texts

    """불용어 로드 메서드: 파일에서 불용어 읽기 + 유효성 검사"""
    def _load_stopwords(self):
        stopwords_path = os.path.join(self.data_dir, "STOPWORDS.txt")
        police_keywords = self._load_police_keywords()  # 경찰 키워드 로드
        token_pattern = re.compile(r'^[가-힣a-zA-Z0-9]{2,}$')  # 유효 토큰 패턴
        
        if os.path.exists(stopwords_path):
            with open(stopwords_path, 'r', encoding='utf-8') as f:
                return {
                    line.strip()
                    for line in f
                    if line.strip() and 
                    line.strip() not in police_keywords and  # 경찰 키워드 제외
                    token_pattern.match(line.strip())  # 유효 패턴 검증
                }
        else:
            print(f"⚠️ {stopwords_path} 파일이 존재하지 않습니다.")
            return set()

    """경찰 키워드 파일 로드 메서드"""
    def _load_police_keywords(self):
        police_path = os.path.join(self.data_dir, "POLICE_KEYWORDS.txt")
        if os.path.exists(police_path):
            with open(police_path, 'r', encoding='utf-8') as f:
                return {line.strip() for line in f if line.strip()}
        else:
            print(f"⚠️ {police_path} 파일이 존재하지 않습니다.")
            return set()

    """유사도 검사 메서드: 두 문자열의 유사도 계산"""
    def _is_similar(self, a, b, threshold=0.8):
        return difflib.SequenceMatcher(None, a, b).ratio() >= threshold

    """유사 키워드 검출 메서드: 슬라이딩 윈도우 기반 검색"""
    def _find_similar_keywords(self, text, keyword_set, threshold=0.8):
        found = set()
        words = text.split()
        # 8-gram부터 1-gram까지 점진적 검색
        for window in range(8, 0, -1):
            for i in range(len(words) - window + 1):
                candidate = ' '.join(words[i:i+window]) # 후보 키워드 생성
                for kw in keyword_set:
                    if self._is_similar(kw, candidate, threshold):
                        found.add(candidate)
        return list(found)

    """중첩 키워드 제거 메서드: 포함 관계 필터링"""
    def _remove_sub_keywords(self, keywords):
        keywords = sorted(keywords, key=len, reverse=True)  # 길이 기준 정렬
        result = []
        for i, kw in enumerate(keywords):
            # 더 긴 키워드에 포함되지 않는 경우만 선택
            if not any(kw != other and kw in other for other in keywords):
                result.append(kw)
        return result

    """사용자 정의 토크나이징 메서드"""
    def _tokenize_text(self, text):
        # 경찰 키워드 유사도 기반 추출
        police_items = self._find_similar_keywords(text, self.police_keywords, threshold=0.8)
        police_items = self._remove_sub_keywords(police_items)  # 중첩 제거
        
        # 형태소 분석 기반 명사 추출
        pos_tagged = self.okt.pos(text, norm=True, stem=True)
        nouns = [word for word, tag in pos_tagged if tag == 'Noun' and len(word) > 1]
        
        # 최종 토큰 생성 (중복 제거)
        tokens = list(dict.fromkeys(police_items + [n for n in nouns if n not in police_items]))
        return tokens

    """모델 학습 메인 메서드"""
    def fit(self, texts):
        self.police_keywords = self._load_police_keywords()
        self.stopwords = self._load_stopwords()
        parsed_texts = self._parse_labels(texts)  # 라벨 파싱
        print(f"라벨 {len(self.labels)}개 로드 완료")
        
        # 토크나이징 및 벡터화
        tokenized_texts = [' '.join(self._tokenize_text(t)) for t in parsed_texts]
        
        # 경찰 키워드 강제 추가 (누락 방지)
        for kw in self.police_keywords:
            if kw not in tokenized_texts:
                tokenized_texts.append(kw)
        
        # CountVectorizer 설정
        self.vectorizer = CountVectorizer(
            max_features=self.max_features,
            min_df=1,  # 최소 문서 빈도
            stop_words=list(self.stopwords),  # 불용어 적용
            token_pattern=r'(?u)\b[가-힣a-zA-Z0-9]{2,}\b'  # 토큰 패턴
        )
        X_counts = self.vectorizer.fit_transform(tokenized_texts)
        self.feature_names = self.vectorizer.get_feature_names_out()  # 특징명 저장
        
        # TF-IDF 변환기 설정
        self.transformer = TfidfTransformer()
        self.transformer.fit(X_counts)
        self.transformer.idf_ = self._adjust_idf(self.transformer.idf_)  # 가중치 조정

    """TF-IDF 가중치 조정 메서드"""
    def _adjust_idf(self, idf):
        """TF-IDF 가중치 조정 메서드"""
        adjusted_idf = idf.copy()
        for idx, term in enumerate(self.feature_names):
            if term in self.stopwords:
                adjusted_idf[idx] = -1  # 불용어 완전 제거
            elif term in self.police_keywords or re.match(r'([가-힣]{1,4}색)', term):
                adjusted_idf[idx] *= 5.0  # 경찰 키워드 & 색상 5배 가중
        return adjusted_idf

    """키워드 추출 메서드"""
    def extract_keywords(self, text, top_n=2):
        if not self.vectorizer or not self.transformer:
            print("⚠️ 모델이 학습되지 않았습니다.")
            return []
        
        # 텍스트 전처리
        tokenized = ' '.join(self._tokenize_text(text))
        # 벡터 변환
        X_counts = self.vectorizer.transform([tokenized])
        X_tfidf = self.transformer.transform(X_counts)
        
        # 상위 키워드 추출
        indices = np.argsort(X_tfidf.toarray()[0])[::-1]
        keywords = [self.feature_names[idx] for idx in indices if X_tfidf[0, idx] > 0]
        
        # 필터링 및 우선순위 정렬
        filtered = [kw for kw in keywords if kw not in self.stopwords]
        priority = [
            kw for kw in filtered
            if kw in self.police_keywords or re.match(r'([가-힣]{1,4}색)', kw)
        ]
        others = [kw for kw in filtered if kw not in priority]
        return (priority + others)[:top_n]

    """모델 저장 메서드"""
    def save_model(self, filename="tfidf_model.joblib"):
        model_path = os.path.join(self.data_dir, filename)
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        model_data = {
            #주어진 텍스트를 파싱해서 각 단어의 빈도수를 세는 알고리즘 저장
            'vectorizer': self.vectorizer, 
            
            #주어진 텍스트를 TF-IDF 가중치 벡터로 변환하는 알고리즘 저장
            'transformer': self.transformer,
            
            #바이너리 파일로 저장된 위의 것들이 어떤 단어에 해당하는지 알려주는 리스트
            'feature_names': self.feature_names,
            'stopwords': self.stopwords,
            'police_keywords': self.police_keywords
        }
        joblib.dump(model_data, model_path)
        print(f"모델이 {model_path}에 저장되었습니다.")

    """모델 로드 메서드"""
    def load_model(self, filename="tfidf_model.joblib"):
        model_path = os.path.join(self.data_dir, filename)
        if not os.path.exists(model_path):
            print(f"⚠️ {model_path} 파일이 존재하지 않습니다.")
            return False
        model_data = joblib.load(model_path)
        
        # 모델 파라미터 복원
        self.vectorizer = model_data['vectorizer']
        self.transformer = model_data['transformer']
        self.feature_names = model_data['feature_names']
        self.stopwords = model_data['stopwords']
        self.police_keywords = model_data['police_keywords']
        print("모델이 로드되었습니다.")
        return True

"""학습 데이터 로드 함수"""
def load_training_data(filename="train_data.txt"):
    DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
    filepath = os.path.join(DATA_DIR, filename)
    
    if not os.path.exists(filepath):
        print(f"⚠️ 학습 데이터 파일이 없습니다: {filepath}")
        return []
    with open(filepath, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f if line.strip()]

if __name__ == "__main__":
    # 학습 데이터 로드
    all_data = load_training_data()
    if not all_data:
        print("학습 데이터가 없습니다.")
        exit()
    
    # 데이터 분할 (96% 학습, 4% 테스트)
    train_texts, test_texts = train_test_split(
        all_data,
        test_size=0.04,
        random_state=42  # 랜덤 시드 고정
    )
    
    # 모델 생성 및 학습
    model = StopwordAwareTFIDF(max_features=5000)
    model.fit(train_texts)
    model.save_model()
    
    # 테스트셋 평가
    print("\n===== 테스트셋 평가 =====")
    correct = 0
    total = len(test_texts)
    for i, text in enumerate(test_texts):
        if '|' in text:
            input_text, true_labels = text.split('|', 1)
            true_keywords = [label.strip() for label in true_labels.split(',')]
        else:
            input_text = text
            true_keywords = []
        
        # 키워드 추출 실행
        pred_keywords = model.extract_keywords(input_text, top_n=2)
        
        # 정확도 계산
        matched = len(set(true_keywords) & set(pred_keywords))
        if matched == len(true_keywords):
            correct += 1
        
        # 결과 출력
        print(f"[예시 {i+1}]")
        print(f"원문: {input_text}")
        print(f"실제 키워드: {true_keywords}")
        print(f"추출 키워드: {pred_keywords}\n")
    
    # 최종 정확도 출력
    print(f"정확도: {correct/total:.2%}")


    