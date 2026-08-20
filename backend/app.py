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

def find_model_key(word: str):
    """
    입력된 단어가 모델 사전에 없더라도 품사 태그(/NNG 등)가 붙은 키를 자동 탐색합니다.
    """
    if not model:
        return None
    if word in model:
        return word
    
    # 형태소 분석 태그 붙은 키 탐색 (예: 화재 -> 화재/NNG, 화재/NNP)
    possible_tags = ["/NNG", "/NNP", "/VV", "/VA", "/MAG"]
    for tag in possible_tags:
        tagged_word = word + tag
        if tagged_word in model:
            return tagged_word

    # 키의 앞부분이 단어로 시작하는 경우 탐색
    for key in model.index_to_key:
        if key.split("/")[0] == word:
            return key

    return None


def create_ranking_for_answer(target_word: str):
    global ANSWER, RANKING_DICT, TOP_RANKS_LIST
    ANSWER = target_word
    RANKING_DICT = {}
    TOP_RANKS_LIST = []

    print(f"🎯 새로운 정답 설정 완료: [{ANSWER}]")

    target_key = find_model_key(ANSWER)

    if not model or not target_key:
        print(f"⚠️ 경고: [{ANSWER}] 단어(또는 관련 태그)가 Word2Vec 모델 사전에 없습니다.")
        return

    similarities = []
    for key in model.index_to_key:
        try:
            sim = float(model.similarity(target_key, key))
            score = int(round(max(0.0, sim) * 1000))
            # 사용자에게 보여줄 때는 품사 태그 제거 (/NNG -> '')
            display_word = key.split("/")[0]
            similarities.append((display_word, key, score))
        except KeyError:
            continue

    # 유사도 점수 기준 내림차순 정렬
    similarities.sort(key=lambda x: x[2], reverse=True)

    # 상위 순위 데이터 구축
    TOP_RANKS_LIST = []
    seen_words = set()
    rank = 1

    for display_word, key, score in similarities:
        if display_word in seen_words:
            continue
        seen_words.add(display_word)

        if display_word == ANSWER:
            score = 1000
            current_rank = 1
        else:
            current_rank = rank
            rank += 1

        RANKING_DICT[display_word] = {"rank": current_rank, "score": score}

        if current_rank <= 100:
            TOP_RANKS_LIST.append({"rank": current_rank, "word": display_word, "score": score})

        if rank > 10000:
            break

    # 정답 단어 1위 강제 보정
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

    if not user_word:
        return {
            "word": "",
            "score": 0,
            "rank": None,
            "answer": False,
            "exists": False,
            "message": "단어를 입력해 주세요."
        }

    # 1. 정답 문자열 직접 일치 체크 (최우선)
    if user_word == ANSWER:
        return {
            "word": user_word,
            "exists": True,
            "score": 1000,
            "rank": 1,
            "answer": True
        }

    # 2. 모델 내 단어/태그 매핑 키 찾기
    user_key = find_model_key(user_word)
    answer_key = find_model_key(ANSWER)

    if not model or not user_key:
        return {
            "word": user_word,
            "score": 0,
            "rank": None,
            "answer": False,
            "exists": False,
            "message": "사전에 등록되지 않은 단어입니다."
        }

    # 3. 1~10,000위 미리 계산된 사전 조회
    if user_word in RANKING_DICT:
        info = RANKING_DICT[user_word]
        return {
            "word": user_word,
            "exists": True,
            "score": info["score"],
            "rank": info["rank"],
            "answer": False
        }

    # 4. 10,000위 밖 단어 실시간 유사도 계산
    try:
        if answer_key and user_key:
            sim = float(model.similarity(answer_key, user_key))
            score = int(round(max(0.0, sim) * 1000))
        else:
            score = 0

        return {
            "word": user_word,
            "exists": True,
            "score": score,
            "rank": 9999,  # 10,000위 밖
            "answer": False
        }
    except Exception as e:
        print(f"실시간 유사도 계산 에러: {e}")
        return {
            "word": user_word,
            "exists": False,
            "score": 0,
            "rank": None,
            "answer": False,
            "message": "유사도 계산 중 오류가 발생했습니다."
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