import os
import glob
from gensim.models import Word2Vec
from kiwipiepy import Kiwi

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_DATA_DIR = os.path.join(BASE_DIR, "data", "raw")
MODEL_OUTPUT_PATH = os.path.join(BASE_DIR, "word2vec.model")

kiwi = Kiwi()

print("📄 데이터 읽는 중...")
sentences = []
txt_files = glob.glob(os.path.join(RAW_DATA_DIR, "*.txt"))

for file_path in txt_files:
    file_name = os.path.basename(file_path).lower()
    is_fire_file = "fire" in file_name or "answer" in file_name
    
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                tokens = kiwi.tokenize(line)
                # 🔧 핵심 수정: len(token.form) > 1 제약을 제거하여 '물', '불' 등 1글자 명사 허용
                nouns = [token.form for token in tokens if token.tag in ("NNG", "NNP")]
                
                if len(nouns) > 1:
                    sentences.append(nouns)
                    # 소방 관련 데이터는 3번 가중치 적용
                    if is_fire_file:
                        sentences.append(nouns)
                        sentences.append(nouns)

print(f"총 {len(sentences)}개 문장 데이터 준비 완료!")

print("🧠 Word2Vec 모델 학습 시작...")
model = Word2Vec(
    sentences=sentences, 
    vector_size=100, 
    window=4, 
    min_count=1,     # 🔧 핵심 수정: 1번만 나와도 사전에 포함 ('사랑', '우정' 등 살리기)
    workers=4, 
    sg=0,            # CBOW 모델
    epochs=25
)

model.save(MODEL_OUTPUT_PATH)
print(f"🎉 재학습 완료! 저장 경로: {MODEL_OUTPUT_PATH}")