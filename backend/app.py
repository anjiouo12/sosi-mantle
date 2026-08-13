import os
import glob
import json
import random
import pickle
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
# 단어 데이터 및 모델 로딩
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_DATA_DIR = os.path.join(BASE_DIR, "data", "raw")
ANSWERS_FILE = os.path.join(RAW_DATA_DIR, "answers.txt")
MODEL_PATH = os.path.join(BASE_DIR, "word2vec.model")

def load_answers():
    """answers.txt에서 소방 정답 단어 목록을 로드합니다."""
    if os.path.exists(ANSWERS_FILE):
        with open(ANSWERS_FILE, "r", encoding="utf-8") as f:
            answers = [line.strip() for line in f if line.strip() and " " not in line.strip()]
            if answers:
                return answers
    return ["소방관", "소방서", "화재", "진화", "안전", "물", "불"]

ANSWER_LIST = load_answers()

# =========================
# Word2Vec 모델 로드
# =========================
print("🧠 Word2Vec 의미 기반 모델 로딩 중...")
try:
    model_obj = Word2Vec.load(MODEL_PATH)
    wv_model = model_obj.wv
    print("⚡ Word2Vec 모델 로딩 완료!")
except Exception as e:
    print(f"⚠️ 모델 로드 실패: {e}")
    wv_model = None

# =========================
# 단어 찾기 보정 도우미 함수
# =========================
def find_model_key(word):
    """
    사용자 입력 단어가 모델 사전에 없으면 
    /NNG(일반명사), /NNP(고유명사) 태그를 붙여서 사전을 다시 찾습니다.
    """
    if not wv_model or not word:
        return None
    if word in wv_model:
        return word
    if f"{word}/NNG" in wv_model:
        return f"{word}/NNG"
    if f"{word}/NNP" in wv_model:
        return f"{word}/NNP"
    return None

# 단어장에 포함시킬 후보군 구성
WORD_SET = set(wv_model.key_to_index.keys()) if wv_model else set()
for ans in ANSWER_LIST:
    key = find_model_key(ans)
    if key:
        WORD_SET.add(ans)

print("로드된 사전 단어 수:", len(WORD_SET))

# 불필요한 조사/어미 끝자리 필터링 함수
INVALID_ENDINGS = ("한다", "이다", "했다", "이며", "에서", "으로", "로써")
def is_valid_word(word):
    clean_w = word.split("/")[0]
    if any(clean_w.endswith(ending) for ending in INVALID_ENDINGS):
        return False
    return True

# =========================
# 오늘의 정답 & 랭킹 생성
# =========================
ANSWER = random.choice(ANSWER_LIST)
ANSWER_KEY = None

RANKS = {}
SCORES = {}
SORTED_RANK_LIST = []

def create_ranking():
    global RANKS, SCORES, SORTED_RANK_LIST, ANSWER, ANSWER_KEY
    
    RANKS = {}
    SCORES = {}
    SORTED_RANK_LIST = []

    if not wv_model:
        print("⚠️ 모델이 로드되지 않아 랭킹을 생성할 수 없습니다.")
        return

    # 정답 단어의 모델 사전 키 검색
    ANSWER_KEY = find_model_key(ANSWER)
    
    # 정답 단어가 모델 사전에 없을 경우 무작위 재선택
    while not ANSWER_KEY:
        print(f"⚠️ 정답 '{ANSWER}'이(가) 모델 사전에 없습니다. 다른 단어를 선택합니다.")
        valid_answers = [w for w in ANSWER_LIST if find_model_key(w)]
        if valid_answers:
            ANSWER = random.choice(valid_answers)
            ANSWER_KEY = find_model_key(ANSWER)
        else:
            # answers.txt의 모든 단어가 모델에 없을 경우 모델 키 중 임의 1개 선택
            raw_key = random.choice(list(wv_model.key_to_index.keys()))
            ANSWER = raw_key.split("/")[0]
            ANSWER_KEY = raw_key

    print(f"🎯 최종 정답 설정 완료: {ANSWER} (사전 키: {ANSWER_KEY})")

    # 코사인 유사도 기준 상위 10,000개 추출
    similar_words = wv_model.most_similar(ANSWER_KEY, topn=10000)
    
    # 1위 정답 본인 등록
    clean_ans = ANSWER_KEY.split("/")[0]
    RANKS[clean_ans] = 1
    SCORES[clean_ans] = 1000
    SORTED_RANK_LIST.append({"rank": 1, "word": clean_ans, "score": 1000})

    current_rank = 2
    for word_key, sim_score in similar_words:
        if not is_valid_word(word_key):
            continue
            
        clean_word = word_key.split("/")[0]
        score_val = max(0, int(sim_score * 1000))
        
        # 중복 단어 방지 (태그 분리 후 동일 단어 처리)
        if clean_word not in SCORES:
            SCORES[clean_word] = score_val
            RANKS[clean_word] = current_rank
            
            SORTED_RANK_LIST.append({
                "rank": current_rank,
                "word": clean_word,
                "score": score_val
            })
            current_rank += 1

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
        "length": len(ANSWER.split("/")[0])
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

    # 사용자 입력 단어를 모델 사전 키로 변환
    target_key = find_model_key(user_word)

    if not wv_model or not target_key:
        return {
            "word": user_word,
            "score": 0,
            "rank": None,
            "answer": False,
            "exists": False,
            "message": "사전에 등록되지 않은 단어입니다."
        }

    clean_answer = ANSWER.split("/")[0]
    is_answer = (user_word == clean_answer)

    if is_answer:
        score = 1000
        rank = 1
    elif user_word in SCORES:
        score = SCORES[user_word]
        rank = RANKS[user_word]
    else:
        # 상위 10,000위 밖의 단어 점수 실시간 계산
        sim = wv_model.similarity(ANSWER_KEY, target_key)
        score = max(0, int(sim * 1000))
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
        "answer": ANSWER.split("/")[0],
        "top_ranks": SORTED_RANK_LIST[:limit]
    }

@app.post("/reset-answer")
def reset_answer():
    global ANSWER
    
    valid_answers = [w for w in ANSWER_LIST if find_model_key(w)]
    ANSWER = random.choice(valid_answers) if valid_answers else random.choice(ANSWER_LIST)
    
    create_ranking()
    
    return {
        "message": "새로운 정답으로 변경되었습니다.",
        "success": True
    }