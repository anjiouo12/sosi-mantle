import json
import pickle

from sentence_transformers import SentenceTransformer


print("임베딩 생성 시작")


model = SentenceTransformer(
    "jhgan/ko-sroberta-multitask"
)


with open(
    "words.json",
    "r",
    encoding="utf-8"
) as f:
    data = json.load(f)


print(
    "JSON 타입:",
    type(data)
)


print(
    "첫번째 데이터:",
    data[0]
)


# words.json 구조:
# [
#   {
#     "word": "가구",
#     "category": "자동수집",
#     "hint": "한국어 명사 데이터"
#   }
# ]

words = [
    item["word"]
    for item in data
    if isinstance(item, dict) and "word" in item
]


print(
    "단어 개수:",
    len(words)
)


vectors = model.encode(
    words,
    show_progress_bar=True
)


with open(
    "word_vectors.pkl",
    "wb"
) as f:
    pickle.dump(
        {
            "words": words,
            "vectors": vectors
        },
        f
    )


print("임베딩 저장 완료")