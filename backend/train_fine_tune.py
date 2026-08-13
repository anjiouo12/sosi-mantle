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
# 꼬맨틀 스타일의 자연스러운 연관도를 방해하는 일반 추상명사 필터링
STOPWORDS = {"과정", "운전", "경우", "이유", "정도", "때문", "가지", "사실", "부분", "관련", "마찰"}

print("📄 소방 데이터 파싱 및 정화 작업 중...")
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
                    if t.tag in ("NNG", "NNP") and re.match(r"^[가-힣]+$", t.form):
                        if (len(t.form) > 1 or t.form in ALLOWED_SINGLE_CHAR) and t.form not in STOPWORDS:
                            nouns.append(t.form)
                
                if len(nouns) > 1:
                    sentences.append(nouns)
                    if is_fire_file:
                        for _ in range(3):
                            sentences.append(nouns)

print(f"총 {len(sentences)}개 고품질 문장 수집 완료!")

print("📦 이미 다운로드된 대용량 FastText 한국어 모델(cc.ko.300.bin) 로딩 중...")
fasttext.util.download_model('ko', if_exists='ignore')
ft = fasttext.load_model('cc.ko.300.bin')

print("🧠 꼬맨틀급 고밀도 벡터 공간 재구성 및 Fine-tuning 진행 중...")
# min_count=2로 노이즈 단어 완전 제어
model = Word2Vec(vector_size=300, window=5, min_count=2, workers=4, sg=1) # Skip-gram(sg=1) 적용으로 단어 간 밀도 극대화
model.build_vocab(sentences)

# FastText 대용량 위키 사전 학습 벡터 주입
for word in model.wv.index_to_key:
    model.wv[word] = ft.get_word_vector(word)

# 소방 문맥 미세 조정 (Epochs 및 Learning rate 조절)
model.train(sentences, total_examples=len(sentences), epochs=25, start_alpha=0.025, end_alpha=0.001)
model.save(FINAL_MODEL_OUTPUT)

print(f"🎉 꼬맨틀급 소방 Fine-tuning 모델 생성 완료! 저장 경로: {FINAL_MODEL_OUTPUT}")