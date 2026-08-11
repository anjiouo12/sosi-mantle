import re
import json
from pathlib import Path

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


# 경로 설정
BASE_DIR = Path(__file__).parent

WORDS_FILE = (
    BASE_DIR
    / "frontend"
    / "data"
    / "words.ts"
)

OUTPUT_FILE = (
    BASE_DIR
    / "frontend"
    / "data"
    / "similarity.json"
)


print("단어 데이터 읽는 중...")


# words.ts에서 word 값 추출
text = WORDS_FILE.read_text(
    encoding="utf-8"
)

words = re.findall(
    r'word:\s*"([^"]+)"',
    text
)


print(f"총 {len(words)}개 단어 발견")
print(words)


print("모델 불러오는 중...")


model = SentenceTransformer(
    "jhgan/ko-sroberta-multitask"
)


print("임베딩 생성 중...")


embeddings = model.encode(words)


print("유사도 계산 중...")


result = {}


for i, word in enumerate(words):

    scores = cosine_similarity(
        [embeddings[i]],
        embeddings
    )[0]


    word_scores = {}

    for j, score in enumerate(scores):

        if i == j:
            continue

        word_scores[words[j]] = round(
            float(score) * 100,
            1
        )


    # 높은 점수 순 정렬
    word_scores = dict(
        sorted(
            word_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )
    )

    result[word] = word_scores


print("파일 저장 중...")


OUTPUT_FILE.write_text(
    json.dumps(
        result,
        ensure_ascii=False,
        indent=2
    ),
    encoding="utf-8"
)


print("완료!")
print(
    f"생성 위치: {OUTPUT_FILE}"
)