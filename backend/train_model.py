import os
import glob
from gensim.models import Word2Vec
from kiwipiepy import Kiwi

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_DATA_DIR = os.path.join(BASE_DIR, "data", "raw")
MODEL_OUTPUT_PATH = os.path.join(BASE_DIR, "word2vec.model")

kiwi = Kiwi()

print("📄 소방 및 일반 데이터 분석 중...")
sentences = []
txt_files = glob.glob(os.path.join(RAW_DATA_DIR, "*.txt"))

for file_path in txt_files:
    # 소방 관련 파일인지 체크
    file_name = os.path.basename(file_path).lower()
    is_fire_file = "fire" in file_name or "answer" in file_name
    
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                tokens = kiwi.tokenize(line)
                # 2글자 이상의 명사만 추출
                nouns = [token.form for token in tokens if token.tag in ("NNG", "NNP") and len(token.form) > 1]
                
                if len(nouns) > 1:
                    sentences.append(nouns)
                    # 소방 데이터는 3번 반복 학습시켜 연관성 가중치 대폭 강화
                    if is_fire_file:
                        sentences.append(nouns)
                        sentences.append(nouns)

print(f"총 {len(sentences)}개 문장 학습 데이터 준비 완료!")

print("🧠 소방 연관도 중심 Word2Vec 모델 학습 시작...")
model = Word2Vec(
    sentences=sentences, 
    vector_size=100, 
    window=3,        # 문맥 범위를 좁혀 단어 간 밀접도 강화
    min_count=2,     # 1번 나온 노이즈 단어 제거
    workers=4, 
    sg=0,            # CBOW 적용 (주제 집중도 향상)
    epochs=25        # 학습 횟수 늘림
)

model.save(MODEL_OUTPUT_PATH)
print(f"🎉 개선된 모델 학습 완료! 저장 경로: {MODEL_OUTPUT_PATH}")