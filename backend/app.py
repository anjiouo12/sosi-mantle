import os
import glob
import json
import random
import pickle
import math
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from gensim.models import Word2Vec

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================
# 점수 스케일링 함수 (꼬맨틀 스타일 비선형 보정)
# =========================
def calculate_score(similarity: float) -> int:
    """
    코사인 유사도(-1.0 ~ 1.0)를 꼬맨틀 스타일의 점수(0 ~ 1000점)로 비선형 변환합니다.
    (상위 유사도 어휘가 700~900점대에 형성되도록 보정)
    """
    if similarity <= 0:
        return 0
    scaled = math.pow(similarity, 0.35) * 1000
    return min(1000, max(0, int(scaled)))

# =========================
# 경로 및 파일 로딩
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_DATA_DIR = os.path.join(BASE_DIR, "data", "raw")
ANSWERS_FILE = os.path.join(RAW_DATA_DIR, "answers.txt")
CUSTOM_WORDS_FILE = os.path.join(RAW_DATA_DIR, "custom_words.txt")
MODEL_PATH = os.path.join(BASE_DIR, "word2vec.model")

# 기본 커스텀 단어 목록 (custom_words.txt가 없을 때도 작동하도록 기본 제공)
DEFAULT_CUSTOM_WORDS = {
    "한국소방시설협회", "소방시설업", "소방공사업", "소방설계업", 
    "소방감리업", "소방시설관리업", "소방기술자", "소방시설", 
    "성능위주설계", "방염처리업", "소방시설공사", "소방공사"
}

def load_custom_words():
    """custom_words.txt 또는 기본 세트에서 기관 관련 단어를 로드합니다."""
    custom_set = set(DEFAULT_CUSTOM_WORDS)
    
    # 1. RAW_DATA_DIR 내부 확인
    if os.path.exists(CUSTOM_WORDS_FILE):
        with open(CUSTOM_WORDS_FILE, "r", encoding="utf-8") as f:
            words = [line.strip() for line in f if line.strip()]
            custom_set.update(words)
    # 2. 루트 BASE_DIR 내부 확인
    elif os.path.exists(os.path.join(BASE_DIR, "custom_words.txt")):
        with open(os.path.join(BASE_DIR, "custom_words.txt"), "r", encoding="utf-8") as f:
            words = [line.strip() for line in f if line.strip()]
            custom_set.update(words)
            
    return custom_set

CUSTOM_WORDS = load_custom_words()
print(f"🏢 등록된 커스텀/협회 단어 수: {len(CUSTOM_WORDS)}")

def load_answers():
    """answers.txt에서 소방 정답 단어 목록을 로드합니다."""
    if os.path.exists(ANSWERS_FILE):
        with open(ANSWERS_FILE, "r", encoding="utf-8") as f:
            answers = [line.strip() for line in f if line.strip() and " " not in line.strip()]
            if answers:
                return answers
    return ["소방관", "소방서", "화재", "진화", "안전", "물", "불", "한국소방시설협회"]

ANSWER_LIST = load_answers()

# =========================
# Word2Vec 모델 로드
# =========================
print("🧠 소방 전용 Word2Vec 모델 로딩 중...")
try:
    model_obj = Word2Vec.load(MODEL_PATH)
    wv_model = model_obj.wv
    print("⚡ Word2Vec 모델 로딩 완료!")
except Exception as e:
    print(f"⚠️ 모델 로드 실패: {e}")
    wv_model = None

# 전체 단어 집합 (모델 사전에 커스텀 단어까지 병합)
WORD_SET = set(wv_model.key_to_index.keys()) if wv_model else set()
WORD_SET.update(CUSTOM_WORDS)
print("로드된 전체 사전 단어 수(커스텀 포함):", len(WORD_SET))

# 불필요한 어미/조사 끝자리 필터링
INVALID_ENDINGS = ("한다", "이다", "했다", "이며", "에서", "으로", "로써")
def is_valid_word(word):
    if any(word.endswith(ending) for ending in INVALID_ENDINGS):
        return False
    return True

# =========================
# 모델 미등록 합성어(커스텀 단어)의 유사도 추론 함수
# =========================
def get_custom_word_similarity(target_word: str, custom_word: str) -> float:
    """
    Word2Vec 모델에 없는 복합 단어(예: 한국소방시설협회)를 
    부분 형태소/키워드 벡터 평균을 통해 정답 단어와의 코사인 유사도로 추론합니다.
    """
    if not wv_model or target_word not in wv_model:
        return 0.0

    target_vec = wv_model[target_word]

    # 모델에서 찾을 수 있는 키워드 부분 검색 (2글자 이상)
    matched_vectors = []
    for key in wv_model.key_to_index.keys():
        if len(key) >= 2 and key in custom_word:
            matched_vectors.append(wv_model[key])

    # 부분 키워드 매칭이 성공한 경우 평균 벡터 사용
    if matched_vectors:
        mean_vec = np.mean(matched_vectors, axis=0)
        norm_target = np.linalg.norm(target_vec)
        norm_mean = np.linalg.norm(mean_vec)
        if norm_target > 0 and norm_mean > 0:
            sim = np.dot(target_vec, mean_vec) / (norm_target * norm_mean)
            return float(sim)

    # 매칭 실패 시 기본 '소방' 키워드와 유사도 측정 시도
    if "소방" in wv_model:
        return float(wv_model.similarity(target_word, "소방") * 0.8)

    return 0.15

# =========================
# 오늘의 정답 & 랭킹 생성
# =========================
ANSWER = random.choice(ANSWER_LIST)

RANKS = {}
SCORES = {}
SORTED_RANK_LIST = []

def create_ranking():
    global RANKS, SCORES, SORTED_RANK_LIST, ANSWER
    
    RANKS = {}
    SCORES = {}
    SORTED_RANK_LIST = []

    if not wv_model:
        print("⚠️ 모델이 로드되지 않아 랭킹을 생성할 수 없습니다.")
        return

    # 정답 단어가 모델 사전 또는 커스텀 사전에 존재할 때까지 재선택
    while ANSWER not in WORD_SET:
        print(f"⚠️ 정답 '{ANSWER}'이(가) 사전에 없습니다. 다른 단어를 선택합니다.")
        valid_answers = [w for w in ANSWER_LIST if w in WORD_SET]
        if valid_answers:
            ANSWER = random.choice(valid_answers)
        else:
            ANSWER = random.choice(list(WORD_SET))

    print(f"🎯 최종 정답 설정 완료: {ANSWER}")

    # 정답 자체 등록
    RANKS[ANSWER] = 1
    SCORES[ANSWER] = 1000
    SORTED_RANK_LIST.append({"rank": 1, "word": ANSWER, "score": 1000})

    # 모델 내 유사 단어 상위 추출
    if ANSWER in wv_model:
        similar_words = wv_model.most_similar(ANSWER, topn=10000)
    else:
        # 정답 자체가 모델에 없는 커스텀 단어일 경우 추론으로 랭킹 산출
        temp_list = []
        for word in wv_model.key_to_index.keys():
            sim = get_custom_word_similarity(word, ANSWER)
            temp_list.append((word, sim))
        temp_list.sort(key=lambda x: x[1], reverse=True)
        similar_words = temp_list[:10000]

    current_rank = 2
    for word, sim_score in similar_words:
        if not is_valid_word(word) or word == ANSWER:
            continue
            
        score_val = calculate_score(sim_score)
        if word not in SCORES:
            SCORES[word] = score_val
            RANKS[word] = current_rank
            
            SORTED_RANK_LIST.append({
                "rank": current_rank,
                "word": word,
                "score": score_val
            })
            current_rank += 1

    # 커스텀 단어들도 랭킹 표에 동적 추가
    for c_word in CUSTOM_WORDS:
        if c_word not in SCORES and c_word != ANSWER:
            sim = get_custom_word_similarity(ANSWER, c_word)
            score_val = calculate_score(sim)
            SCORES[c_word] = score_val
            # 커스텀 단어는 기본적으로 상위 10,000위 외 배치 후 개별 검색 가능
            RANKS[c_word] = 9999

    print(f"✅ 의미 기반 랭킹 생성 완료 (상위 단어 수: {len(SORTED_RANK_LIST)})")

# 초기 랭킹 계산
create_ranking()

# =========================
# API 엔드포인트
# =========================
@app.get("/")
def home():
    return {
        "message": "소시맨틀 AI 서버 정상 작동",
        "words": len(WORD_SET)
    }

@app.get("/daily")
def daily():
    return {
        "length": len(ANSWER)
    }

@app.post("/guess")
def guess(data: dict):
    user_word = data.get("guess", "").strip()

    if not user_word:
        return {
            "word": "",
            "score": 0,
            "rank": None,
            "answer": False,
            "exists": False,
            "message": "단어를 입력해주세요."
        }

    # 사전에 아예 없는 경우 (모델에도 없고 커스텀 사전에도 없는 일반 없는 단어)
    if user_word not in WORD_SET:
        return {
            "word": user_word,
            "score": 0,
            "rank": None,
            "answer": False,
            "exists": False,
            "message": "사전에 등록되지 않은 단어입니다."
        }

    is_answer = (user_word == ANSWER)

    if is_answer:
        score = 1000
        rank = 1
    elif user_word in SCORES:
        score = SCORES[user_word]
        rank = RANKS[user_word]
    else:
        # 모델 사전 내 존재 단어 유사도 계산
        if wv_model and user_word in wv_model and ANSWER in wv_model:
            sim = wv_model.similarity(ANSWER, user_word)
        else:
            # 커스텀 단어 유사도 동적 추론
            sim = get_custom_word_similarity(ANSWER, user_word)
            
        score = calculate_score(sim)
        rank = 9999

    return {
        "word": user_word,
        "exists": True,
        "score": score,
        "rank": rank,
        "answer": is_answer
    }

@app.get("/top-ranks")
def get_top_ranks(limit: int = 100):
    if not ANSWER or not SORTED_RANK_LIST:
        return {"error": "정답 단어가 설정되지 않았습니다.", "top_ranks": []}

    return {
        "answer": ANSWER,
        "top_ranks": SORTED_RANK_LIST[:limit]
    }

@app.post("/reset-answer")
def reset_answer():
    global ANSWER
    
    valid_answers = [w for w in ANSWER_LIST if w in WORD_SET]
    ANSWER = random.choice(valid_answers) if valid_answers else random.choice(ANSWER_LIST)
    
    create_ranking()
    
    return {
        "message": "새로운 정답으로 변경되었습니다.",
        "success": True
    }