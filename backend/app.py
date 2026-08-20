import os
import random
import math
import numpy as np
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from gensim.models import Word2Vec

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def calculate_score(similarity: float) -> int:
    """코사인 유사도를 꼬맨틀 스타일 점수(0~1000점)로 변환"""
    if similarity <= 0:
        return 0
    return min(1000, max(0, int(math.pow(similarity, 0.35) * 1000)))

# =========================
# 파일 및 데이터 로딩
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_DATA_DIR = os.path.join(BASE_DIR, "data", "raw")
ANSWERS_FILE = os.path.join(RAW_DATA_DIR, "answers.txt")
CUSTOM_WORDS_FILE = os.path.join(RAW_DATA_DIR, "custom_words.txt")
MODEL_PATH = os.path.join(BASE_DIR, "word2vec.model")

DEFAULT_CUSTOM_WORDS = {
    "한국소방시설협회", "소방시설업", "소방공사업", "소방설계업", 
    "소방감리업", "소방시설관리업", "소방기술자", "소방시설", 
    "성능위주설계", "방염처리업", "소방시설공사", "소방공사"
}

def load_file_lines(filepath: str) -> set:
    if os.path.exists(filepath):
        with open(filepath, "r", encoding="utf-8") as f:
            return {line.strip() for line in f if line.strip() and " " not in line.strip()}
    return set()

CUSTOM_WORDS = DEFAULT_CUSTOM_WORDS | load_file_lines(CUSTOM_WORDS_FILE)
ANSWER_LIST = list(load_file_lines(ANSWERS_FILE)) or ["소방관", "소방서", "화재", "진화", "안전", "물", "불", "한국소방시설협회"]

# Word2Vec 로드
try:
    wv_model = Word2Vec.load(MODEL_PATH).wv
except Exception:
    wv_model = None

WORD_SET = (set(wv_model.key_to_index.keys()) if wv_model else set()) | CUSTOM_WORDS

INVALID_ENDINGS = ("한다", "이다", "했다", "이며", "에서", "으로", "로써")
def is_valid_word(word: str) -> bool:
    return not any(word.endswith(ending) for ending in INVALID_ENDINGS)

# =========================
# 경량화된 유사도 추론 함수
# =========================
def get_custom_word_similarity(target_word: str, custom_word: str) -> float:
    if not wv_model:
        return 0.0

    if target_word in wv_model and custom_word in wv_model:
        return float(wv_model.similarity(target_word, custom_word))

    def get_vec(word):
        if word in wv_model:
            return wv_model[word]
        matched = [wv_model[k] for k in wv_model.key_to_index.keys() if len(k) >= 2 and (k in word or word in k)]
        return np.mean(matched, axis=0) if matched else None

    v1, v2 = get_vec(target_word), get_vec(custom_word)
    if v1 is not None and v2 is not None:
        n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
        if n1 > 0 and n2 > 0:
            return float(np.dot(v1, v2) / (n1 * n2))
    return 0.0

# =========================
# 랭킹 생성
# =========================
ANSWER = ""
RANKS, SCORES, SORTED_RANK_LIST = {}, {}, []

def create_ranking():
    global RANKS, SCORES, SORTED_RANK_LIST, ANSWER
    RANKS, SCORES, SORTED_RANK_LIST = {}, {}, []

    if not wv_model:
        return

    valid_answers = [w for w in ANSWER_LIST if w in WORD_SET]
    ANSWER = random.choice(valid_answers) if valid_answers else random.choice(list(WORD_SET))

    # 유사 단어 점수 계산
    if ANSWER in wv_model:
        similar_words = wv_model.most_similar(ANSWER, topn=1000)
    else:
        temp = [(w, get_custom_word_similarity(w, ANSWER)) for w in wv_model.key_to_index.keys()]
        temp.sort(key=lambda x: x[1], reverse=True)
        similar_words = temp[:1000]

    for word, sim in similar_words:
        if is_valid_word(word) and word != ANSWER:
            SCORES[word] = calculate_score(sim)

    for c_word in CUSTOM_WORDS:
        if c_word not in SCORES and c_word != ANSWER:
            SCORES[c_word] = calculate_score(get_custom_word_similarity(ANSWER, c_word))

    # [핵심 보정] 정답 단어 본인은 무조건 1000점 및 1위 고정
    SCORES[ANSWER] = 1000
    RANKS[ANSWER] = 1

    # 정답 외 단어 정렬
    other_items = [(w, score) for w, score in SCORES.items() if w != ANSWER]
    other_items.sort(key=lambda x: x[1], reverse=True)

    # 1위(정답)부터 리스트 생성
    SORTED_RANK_LIST = [{"rank": 1, "word": ANSWER, "score": 1000}]

    for idx, (w, score) in enumerate(other_items, start=2):
        if idx <= 1000:
            RANKS[w] = idx
            SORTED_RANK_LIST.append({"rank": idx, "word": w, "score": score})
        else:
            RANKS[w] = 9999

create_ranking()

# =========================
# API 엔드포인트
# =========================
@app.get("/")
def home():
    return {"message": "소시맨틀 AI 서버 작동 중", "words": len(WORD_SET)}

@app.get("/daily")
def daily():
    return {"length": len(ANSWER)}

@app.post("/guess")
def guess(data: dict):
    user_word = data.get("guess", "").strip()
    if not user_word:
        return {"word": "", "score": 0, "rank": None, "answer": False, "exists": False, "message": "단어를 입력해주세요."}

    # 문자열 완전 일치 검사
    if user_word == ANSWER:
        return {"word": user_word, "exists": True, "score": 1000, "rank": 1, "answer": True}

    # [핵심 보정] 정답 여부 최우선 판단
    if user_word == ANSWER:
        return {"word": user_word, "exists": True, "score": 1000, "rank": 1, "answer": True}

    is_custom_subword = False
    if user_word not in WORD_SET:
        if any(user_word in c or c in user_word for c in CUSTOM_WORDS) and len(user_word) >= 1:
            is_custom_subword = True
        else:
            return {"word": user_word, "score": 0, "rank": None, "answer": False, "exists": False, "message": "사전에 등록되지 않은 단어입니다."}

    if user_word in SCORES:
        score = SCORES[user_word]
        rank = RANKS.get(user_word, 9999)
    else:
        if wv_model and user_word in wv_model and ANSWER in wv_model and not is_custom_subword:
            sim = wv_model.similarity(ANSWER, user_word)
        else:
            sim = get_custom_word_similarity(ANSWER, user_word)
        score, rank = calculate_score(sim), 9999

    return {"word": user_word, "exists": True, "score": score, "rank": rank, "answer": False}

@app.get("/top-ranks")
def get_top_ranks(limit: int = 100):
    return {"answer": ANSWER, "top_ranks": SORTED_RANK_LIST[:limit]}

@app.post("/reset-answer")
def reset_answer():
    create_ranking()
    return {"message": "새로운 정답으로 변경되었습니다.", "success": True}