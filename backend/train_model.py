import os
import glob
from gensim.models import Word2Vec
from kiwipiepy import Kiwi

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_DATA_DIR = os.path.join(BASE_DIR, "data", "raw")
MODEL_OUTPUT_PATH = os.path.join(BASE_DIR, "word2vec.model")

kiwi = Kiwi()

print("📄 소방 데이터 읽는 중...")
sentences = []
txt_files = glob.glob(os.path.join(RAW_DATA_DIR, "*.txt"))

for file_path in txt_files:
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                # 명사(NNG, NNP) 단위로 잘라내기 (자바 설치 없이 동작)
                tokens = kiwi.tokenize(line)
                nouns = [token.form for token in tokens if token.tag in ("NNG", "NNP")]
                if len(nouns) > 1:
                    sentences.append(nouns)

print(f"총 {len(sentences)}개 문장 데이터 준비 완료!")

print("🧠 소방 전용 Word2Vec 모델 학습 시작...")
model = Word2Vec(
    sentences=sentences, 
    vector_size=100, 
    window=5, 
    min_count=1,  # 1번만 등장해도 단어장에 포함
    workers=4, 
    sg=1
)

model.save(MODEL_OUTPUT_PATH)
print(f"🎉 소방 전용 모델 학습 완료! 파일 저장됨: {MODEL_OUTPUT_PATH}")