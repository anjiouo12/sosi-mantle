import os
import glob
import json
import random
import pickle
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================
# 단어 데이터 로딩
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_DATA_DIR = os.path.join(BASE_DIR, "data", "raw")
CACHE_FILE = os.path.join(BASE_DIR, "word_vectors.pkl")
WORDS_LARGE_FILE = os.path.join(RAW_DATA_DIR, "words_large.txt")
ANSWERS_FILE = os.path.join(RAW_DATA_DIR, "answers.txt")

def load_answers():
    """answers.txt에서 소방 정답 단어 목록을 로드합니다."""
    if os.path.exists(ANSWERS_FILE):
        with open(ANSWERS_FILE, "r", encoding="utf-8") as f:
            answers = [line.strip() for line in f if line.strip() and " " not in line.strip()]
            if answers:
                return answers
    return ["소방관", "소방서", "화재", "진화", "안전", "물", "불"]

ANSWER_LIST = load_answers()

def load_words_from_raw():
    """words_large.txt에서 전체 추측 단어를 로드합니다 (1글자 이상 허용)."""
    word_set = set()
    
    # 1. words_large.txt가 존재하면 우선 로드
    if os.path.exists(WORDS_LARGE_FILE):
        with open(WORDS_LARGE_FILE, "r", encoding="utf-8") as f:
            for line in f:
                word = line.strip()
                if word and " " not in word and len(word) >= 1:
                    word_set.add(word)
    else:
        # words_large.txt가 없을 경우 raw 폴더의 모든 txt 로드
        txt_files = glob.glob(os.path.join(RAW_DATA_DIR, "*.txt"))
        for file_path in txt_files:
            with open(file_path, "r", encoding="utf-8") as f:
                for line in f:
                    word = line.strip()
                    if word and " " not in word and len(word) >= 1:
                        word_set.add(word)

    # 2. 정답 단어(answers)도 추측 가능 단어장에 포함 보장
    for ans in ANSWER_LIST:
        word_set.add(ans)
                    
    return sorted(list(word_set))

WORD_LIST = load_words_from_raw()

if not WORD_LIST:
    print("경고: data/raw 폴더에서 단어를 찾지 못했습니다. 기본 단어를 사용합니다.")
    WORD_LIST = ["소방관", "소방서", "화재", "진화", "안전", "물", "불"]
    ANSWER_LIST = WORD_LIST

WORD_SET = set(WORD_LIST)
print("로드된 전체 단어 수:", len(WORD_LIST))

# =========================
# TF-IDF 벡터 생성 (메모리 최적화)
# =========================
print("TF-IDF 단어 벡터 계산 중...")
vectorizer = TfidfVectorizer(analyzer="char", ngram_range=(1, 2))
WORD_VECTORS = vectorizer.fit_transform(WORD_LIST)
print("⚡ TF-IDF 벡터화 완료!")

# =========================
# 오늘의 정답
# =========================
ANSWER = random.choice(ANSWER_LIST)
print("오늘의 정답:", ANSWER)

ANSWER_INDEX = WORD_LIST.index(ANSWER) if ANSWER in WORD_LIST else 0
ANSWER_VECTOR = WORD_VECTORS[ANSWER_INDEX]

# =========================
# 랭킹 데이터 생성
# =========================
RANKS = {}
SCORES = {}
SORTED_RANK_LIST = []

def create_ranking():
    global RANKS, SCORES, SORTED_RANK_LIST
    result = []

    # 전체 유사도 한번에 계산 (속도 최적화)
    similarities = cosine_similarity(ANSWER_VECTOR, WORD_VECTORS)[0]

    for idx, score in enumerate(similarities):
        score_val = round(float(score) * 1000)
        word = WORD_LIST[idx]

        SCORES[word] = score_val
        result.append({"word": word, "score": score_val})

    result.sort(key=lambda x: x["score"], reverse=True)

    SORTED_RANK_LIST = []
    for rank, item in enumerate(result, start=1):
        RANKS[item["word"]] = rank
        SORTED_RANK_LIST.append({
            "rank": rank,
            "word": item["word"],
            "score": item["score"]
        })

    print("랭킹 생성 완료:", len(RANKS))

create_ranking()

# =========================
# API 엔드포인트
# =========================
@app.get("/")
def home():
    return {
        "message": "소시맨틀 AI 서버 정상 작동",
        "words": len(WORD_LIST)
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

    if user_word not in WORD_SET:
        return {
            "word": user_word,
            "score": 0,
            "rank": None,
            "answer": False,
            "exists": False,
            "message": "사전에 등록되지 않은 단어입니다."
        }

    score = SCORES.get(user_word, 0)
    rank = RANKS.get(user_word, None)
    is_answer = (user_word == ANSWER)

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