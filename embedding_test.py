from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


print("모델 불러오는 중...")

model = SentenceTransformer(
    "jhgan/ko-sroberta-multitask"
)

print("모델 로딩 완료")


words = [
    "소방관",
    "화재",
    "소화기",
    "자동차",
    "강아지"
]


embeddings = model.encode(words)


for i in range(len(words)):
    for j in range(i + 1, len(words)):

        score = cosine_similarity(
            [embeddings[i]],
            [embeddings[j]]
        )[0][0]

        print(
            f"{words[i]} ↔ {words[j]} : {score:.3f}"
        )