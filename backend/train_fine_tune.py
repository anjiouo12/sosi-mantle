import os
import glob
import re
import fasttext.util
from gensim.models import Word2Vec
from kiwipiepy import Kiwi

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_DATA_DIR = os.path.join(BASE_DIR, "data", "raw")
FINAL_MODEL_OUTPUT = os.path.join(BASE_DIR, "word2vec.model")

kiwi = Kiwi()
ALLOWED_SINGLE_CHAR = {"물", "불", "해", "달", "옷", "집", "길"}

print("📄 소방 데이터 파싱 및 노이즈 정화 중...")
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
                for t in tokens:
                    # 순수 한글 명사만 허용 및 불필요한 단어 파편 필터링
                    if t.tag in ("NNG", "NNP") and re.match(r"^[가-힣]+$", t.form):
                        if len(t.form) > 1 or t.form in ALLOWED_SINGLE_CHAR:
                            nouns.append(t.form)
                
                if len(nouns) > 1:
                    sentences.append(nouns)
                    if is_fire_file:
                        for _ in range(3):
                            sentences.append(nouns)

print(f"총 {len(sentences)}개 소방 문장 수집 완료!")

print("📦 FastText 공식 한국어 사전 학습 모델 로딩 중...")
fasttext.util.download_model('ko', if_exists='ignore')
ft = fasttext.load_model('cc.ko.300.bin')

print("🧠 소방 데이터 추가 학습(Fine-tuning) 진행 중...")
model = Word2Vec(vector_size=300, window=5, min_count=2, workers=4) # min_count=2로 최소 빈도 조절
model.build_vocab(sentences)

for word in model.wv.index_to_key:
    model.wv[word] = ft.get_word_vector(word)

model.train(sentences, total_examples=len(sentences), epochs=20)
model.save(FINAL_MODEL_OUTPUT)

print(f"🎉 노이즈 제거 완료! 저장 경로: {FINAL_MODEL_OUTPUT}")