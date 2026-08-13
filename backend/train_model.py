import os
import glob
from gensim.models import Word2Vec
from kiwipiepy import Kiwi

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_DATA_DIR = os.path.join(BASE_DIR, "data", "raw")
MODEL_OUTPUT_PATH = os.path.join(BASE_DIR, "word2vec.model")

kiwi = Kiwi()

# 허용할 핵심 1글자 명사 리스트 (필요시 추가)
ALLOWED_SINGLE_CHAR_NOUNS = {"물", "불", "해", "달", "뼘", "옷", "집", "길"}

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
                nouns = []
                for token in tokens:
                    if token.tag in ("NNG", "NNP"):
                        # 2글자 이상이거나, 1글자 중 선별된 단어만 허용
                        if len(token.form) > 1 or token.form in ALLOWED_SINGLE_CHAR_NOUNS:
                            nouns.append(token.form)
                
                if len(nouns) > 1:
                    sentences.append(nouns)
                    # 소방 데이터는 10번 반복 추가하여 소방 연관도를 압도적으로 강화
                    if is_fire_file:
                        for _ in range(10):
                            sentences.append(nouns)

print(f"총 {len(sentences)}개 문장 데이터 준비 완료!")

print("🧠 소방 연관도 강화 Word2Vec 모델 학습 시작...")
model = Word2Vec(
    sentences=sentences, 
    vector_size=100, 
    window=3,        # 좁은 문맥으로 연관성 밀도 향상
    min_count=2,     # 노이즈 단어 배제
    workers=4, 
    sg=0,            # CBOW 모델
    epochs=30
)

model.save(MODEL_OUTPUT_PATH)
print(f"🎉 모델 재학습 완료! 저장 경로: {MODEL_OUTPUT_PATH}")