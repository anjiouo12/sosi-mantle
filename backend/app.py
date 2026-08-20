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
# 주의: 모델 파일 경로(ko_wv.bin 또는 model.wv)는 실제 환경에 맞게 확인해 주세요.
try:
    model = KeyedVectors.load_word2vec_format("ko_wv.bin", binary=True)
    print("✅ Word2Vec 모델 로드 완료!")
except Exception as e:
    print(f"⚠️ 모델 로드 실패: {e}")
    model = None

# 정답 후보 단어 리스트 (원하시는 단어들로 자유롭게 구성)
CANDIDATE_WORDS = [
    "연기포집", "화재감지기", "소방관", "소화기", "건조주의보",
    "정온식감지기", "옥내소화전", "비상방송설비", "한국소방시설협회"
]

# 현재 게임의 정답 단어 및 TOP 1000 순위 데이터
ANSWER = ""
RANKING_DICT = {}  # { word: {"rank": rank, "score": score} }
TOP_RANKS_LIST = []  # [{"rank": 1, "word": "...", "score": 1000}, ...]

def create_ranking_for_answer(target_word: str):
    """
    정답 단어 기준으로 전체 단어 사전과의 유사도를 계산하여 순위를 매깁니다.
    """
    global ANSWER, RANKING_DICT, TOP_RANKS_LIST
    ANSWER = target_word
    RANKING_DICT = {}
    TOP_RANKS_LIST = []

    print(f"🎯 새로운 정답 설정 완료: [{ANSWER}]")

    if not model or ANSWER not in model:
        print(f"⚠️ 경고: [{ANSWER}] 단어가 Word2Vec 모델 사전에 없습니다.")
        return

    # 모델 내 모든 단어와의 유사도 계산
    similarities = []
    for word in model.index_to_key:
        try:
            sim = float(model.similarity(ANSWER, word))
            # 코사인 유사도(-1 ~ 1)를 0 ~ 1000점 범위로 변환
            score = int(round(max(0.0, sim) * 1000))
            similarities.append((word, score))
        except KeyError:
            continue

    # 점수 높은 순 정렬
    similarities.sort(key=lambda x: x[1], reverse=True)

    # 순위 부여 (상위 10,000개 저장)
    TOP_RANKS_LIST = []
    for idx, (word, score) in enumerate(similarities[:10000], start=1):
        # 정답 단어 본인은 무조건 1위/1000점 보정
        if word == ANSWER:
            score = 1000
            idx = 1

        rank_info = {"rank": idx, "score": score}
        RANKING_DICT[word] = rank_info

        if idx <= 100:
            TOP_RANKS_LIST.append({"rank": idx, "word": word, "score": score})

    # 정답 단어가 RANKING_DICT에 1위로 확실히 들어가도록 강제 보정
    RANKING_DICT[ANSWER] = {"rank": 1, "score": 1000}


# 서버 시작 시 정답 초기화
@app.on_event("startup")
def startup_event():
    initial_word = random.choice(CANDIDATE_WORDS)
    create_ranking_for_answer(initial_word)


# ---------------------------------------------------------
# 2. API 엔드포인트 정의
# ---------------------------------------------------------

@app.post("/guess")
def guess(data: dict):
    """
    사용자가 입력한 단어의 정답 여부, 점수, 순위를 반환합니다.
    """
    user_word = data.get("guess", "").strip()

    if not user_word:
        return {
            "word": "",
            "score": 0,
            "rank": None,
            "answer": False,
            "exists": False,
            "message": "단어를 입력해 주세요."
        }

    # 🔥 핵심 보정: 정답 문자와 완벽히 일치하면 유사도 계산 없이 무조건 1위/1000점 정답 처리!
    if user_word == ANSWER:
        return {
            "word": user_word,
            "exists": True,
            "score": 1000,
            "rank": 1,
            "answer": True
        }

    # 사전에 없는 단어인 경우
    if model and user_word not in model:
        return {
            "word": user_word,
            "score": 0,
            "rank": None,
            "answer": False,
            "exists": False,
            "message": "사전에 등록되지 않은 단어입니다."
        }

    # 미리 계산된 순위 테이블에서 검색
    if user_word in RANKING_DICT:
        info = RANKING_DICT[user_word]
        return {
            "word": user_word,
            "exists": True,
            "score": info["score"],
            "rank": info["rank"],
            "answer": False
        }

    # 순위권(10,000위) 밖 단어의 점수 계산
    if model:
        sim = float(model.similarity(ANSWER, user_word))
        score = int(round(max(0.0, sim) * 1000))
        return {
            "word": user_word,
            "exists": True,
            "score": score,
            "rank": 9999,  # 10,000위 밖
            "answer": False
        }

    return {
        "word": user_word,
        "exists": False,
        "score": 0,
        "rank": None,
        "answer": False,
        "message": "유사도 계산에 실패했습니다."
    }


@app.get("/top-ranks")
def get_top_ranks(limit: int = 100):
    """
    TOP 100 순위 목록을 반환합니다.
    """
    return {
        "answer": ANSWER,
        "top_ranks": TOP_RANKS_LIST[:limit]
    }


@app.post("/reset-answer")
def reset_answer():
    """
    다음 문제로 진행할 때 정답을 새 단어로 변경합니다.
    """
    new_word = random.choice([w for w in CANDIDATE_WORDS if w != ANSWER])
    create_ranking_for_answer(new_word)
    return {
        "message": "새로운 정답으로 변경되었습니다.",
        "answer": ANSWER
    }