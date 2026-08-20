import random
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from gensim.models import KeyedVectors

app = FastAPI()

# CORS 설정 (프론트엔드 통신 허용)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------
# 1. Word2Vec 모델 로드 및 단어 사전 준비
# ---------------------------------------------------------
try:
    model = KeyedVectors.load_word2vec_format("ko_wv.bin", binary=True)
    print("✅ Word2Vec 모델 로드 완료!")
except Exception as e:
    print(f"⚠️ 모델 로드 실패: {e}")
    model = None

# 정답 후보 단어 리스트
CANDIDATE_WORDS = [
    "연기포집", "화재감지기", "소방관", "소화기", "건조주의보",
    "정온식감지기", "옥내소화전", "비상방송설비", "한국소방시설협회"
]

ANSWER = ""
RANKING_DICT = {}
TOP_RANKS_LIST = []

def create_ranking_for_answer(target_word: str):
    global ANSWER, RANKING_DICT, TOP_RANKS_LIST
    ANSWER = target_word
    RANKING_DICT = {}
    TOP_RANKS_LIST = []

    print(f"🎯 새로운 정답 설정 완료: [{ANSWER}]")

    if not model or ANSWER not in model:
        print(f"⚠️ 경고: [{ANSWER}] 단어가 Word2Vec 모델 사전에 없습니다.")
        return

    similarities = []
    for word in model.index_to_key:
        try:
            sim = float(model.similarity(ANSWER, word))
            score = int(round(max(0.0, sim) * 1000))
            similarities.append((word, score))
        except KeyError:
            continue

    similarities.sort(key=lambda x: x[1], reverse=True)

    TOP_RANKS_LIST = []
    for idx, (word, score) in enumerate(similarities[:10000], start=1):
        if word == ANSWER:
            score = 1000
            idx = 1

        rank_info = {"rank": idx, "score": score}
        RANKING_DICT[word] = rank_info

        if idx <= 100:
            TOP_RANKS_LIST.append({"rank": idx, "word": word, "score": score})

    RANKING_DICT[ANSWER] = {"rank": 1, "score": 1000}


@app.on_event("startup")
def startup_event():
    initial_word = random.choice(CANDIDATE_WORDS)
    create_ranking_for_answer(initial_word)


# ---------------------------------------------------------
# 2. API 엔드포인트 정의
# ---------------------------------------------------------

@app.post("/guess")
def guess(data: dict):
    user_word = data.get("guess", "").strip()

    # 입력값이 비어있는 경우
    if not user_word:
        return {
            "word": "",
            "score": 0,
            "rank": None,
            "answer": False,
            "exists": False,
            "message": "단어를 입력해 주세요."
        }

    # 1단계: 정답과 문자열이 정확히 일치하는지 검사 (최우선)
    if user_word == ANSWER:
        return {
            "word": user_word,
            "exists": True,
            "score": 1000,
            "rank": 1,
            "answer": True
        }

    # 2단계: 모델 사전에 단어가 존재하는지 검사
    if not model or user_word not in model:
        return {
            "word": user_word,
            "score": 0,
            "rank": None,
            "answer": False,
            "exists": False,
            "message": "사전에 등록되지 않은 단어입니다."
        }

    # 3단계: 미리 계산된 1~10,000위 순위 사전(RANKING_DICT)에 있는지 검사
    if user_word in RANKING_DICT:
        info = RANKING_DICT[user_word]
        return {
            "word": user_word,
            "exists": True,
            "score": info["score"],
            "rank": info["rank"],
            "answer": False
        }

    # 4단계: 사전에 존재하지만 10,000위 밖인 단어의 유사도 실시간 계산
    try:
        sim = float(model.similarity(ANSWER, user_word))
        score = int(round(max(0.0, sim) * 1000))
        return {
            "word": user_word,
            "exists": True,
            "score": score,
            "rank": 9999,  # 10,000위 밖
            "answer": False
        }
    except Exception as e:
        print(f"실시간 유사도 계산 중 오류 발생: {e}")
        return {
            "word": user_word,
            "exists": False,
            "score": 0,
            "rank": None,
            "answer": False,
            "message": "유사도 계산 처리 중 오류가 발생했습니다."
        }


@app.get("/top-ranks")
def get_top_ranks(limit: int = 100):
    return {
        "answer": ANSWER,
        "top_ranks": TOP_RANKS_LIST[:limit]
    }


@app.post("/reset-answer")
def reset_answer():
    new_word = random.choice([w for w in CANDIDATE_WORDS if w != ANSWER])
    create_ranking_for_answer(new_word)
    return {
        "message": "새로운 정답으로 변경되었습니다.",
        "answer": ANSWER
    }