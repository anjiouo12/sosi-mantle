import json


# 기존 전체 단어
with open(
    "words.json",
    "r",
    encoding="utf-8"
) as f:
    words = json.load(f)


# 정답 후보
with open(
    "answer_words.json",
    "r",
    encoding="utf-8"
) as f:
    answers = json.load(f)

print("전체 단어 개수:", len(words))
print("정답 후보 개수:", len(answers))

print("words 첫 데이터:")
print(words[0])

print("answers 첫 데이터:")
print(answers[0])




before = len(words)


for word in answers:

    if word not in words:
        words.append(word)



with open(
    "words.json",
    "w",
    encoding="utf-8"
) as f:
    json.dump(
        words,
        f,
        ensure_ascii=False,
        indent=2
    )



print("기존 단어 :", before)
print("추가 단어 :", len(words)-before)
print("최종 단어 :", len(words))