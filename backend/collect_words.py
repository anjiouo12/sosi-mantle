import os
import re
from kiwipiepy import Kiwi

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_DIR = os.path.join(BASE_DIR, "data", "raw")
WORDS_FILE = os.path.join(RAW_DIR, "words_large.txt")
ANSWER_FILE = os.path.join(RAW_DIR, "answers.txt")
CACHE_FILE = os.path.join(BASE_DIR, "word_vectors.pkl")

kiwi = Kiwi()

# 어절 끝의 대표 조사 제거용 정규식
JOSA_PATTERN = re.compile(r'(은|는|이|가|을|를|에|의|로|으로|와|과|도|만|나|이나|에서|까지|부터|에게|한테)$')

def process_file_words(file_path):
    """일반 파일 내에서 명사 및 단어 원형을 추출합니다 (1글자 이상 포함)."""
    words = set()
    if not os.path.exists(file_path):
        return words

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        # 1. Kiwi 형태소 분석 (1글자 이상 허용, 복합명사 유지)
        tokens = kiwi.tokenize(content, match_options=0)
        for token in tokens:
            if token.tag in ['NNG', 'NNP'] and len(token.form) >= 1:
                if token.form.isalpha():
                    words.add(token.form)

        # 2. 어절 단위 정교화 (조사 제거, 1글자 이상 허용)
        raw_words = re.findall(r'[가-힣a-zA-Z0-9]+', content)
        for raw in raw_words:
            cleaned = JOSA_PATTERN.sub('', raw)
            if len(cleaned) >= 1 and cleaned.isalpha():
                words.add(cleaned)

    except Exception as e:
        print(f"⚠️ 파일 처리 오류 ({file_path}): {e}")

    return words

def process_words_and_answers():
    if not os.path.exists(RAW_DIR):
        os.makedirs(RAW_DIR, exist_ok=True)

    answer_words = set()
    all_words = set()

    # Target 파일 탐색
    target_files = []
    for root, _, files in os.walk(BASE_DIR):
        if 'venv' in root or '.next' in root or 'node_modules' in root:
            continue
        for file in files:
            if file.endswith('.txt'):
                target_files.append(os.path.join(root, file))

    for file_path in target_files:
        file_name = os.path.basename(file_path)

        if file_name in ["words_large.txt", "answers.txt", "requirements.txt"]:
            continue

        # 1. fire_words.txt는 쪼개지 않고 원본 단어 줄단위 그대로 읽기
        if "fire_words" in file_name.lower():
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    lines = [line.strip() for line in f if line.strip()]
                    answer_words.update(lines)
                    all_words.update(lines)
            except Exception as e:
                print(f"⚠️ fire_words 읽기 오류: {e}")
        else:
            # 2. 일반 파일들은 형태소 분석 진행하여 추측 단어장에 추가
            extracted = process_file_words(file_path)
            all_words.update(extracted)

    if not answer_words:
        answer_words = all_words.copy()

    # 1. answers.txt 저장
    with open(ANSWER_FILE, "w", encoding="utf-8") as f:
        for w in sorted(list(answer_words)):
            f.write(f"{w}\n")

    # 2. words_large.txt 저장
    with open(WORDS_FILE, "w", encoding="utf-8") as f:
        for w in sorted(list(all_words)):
            f.write(f"{w}\n")

    # 3. 기존 임베딩 캐시 자동 삭제
    if os.path.exists(CACHE_FILE):
        try:
            os.remove(CACHE_FILE)
            print("🗑️ 기존 임베딩 캐시(word_vectors.pkl)를 삭제했습니다.")
        except Exception as e:
            print(f"⚠️ 캐시 삭제 오류: {e}")

    print("--------------------------------------------------")
    print(f"🎯 [answers.txt] 소방 정답 후보: {len(answer_words)}개")
    print(f"📦 [words_large.txt] 전체 추측 단어장: {len(all_words)}개")
    print("--------------------------------------------------")

if __name__ == "__main__":
    process_words_and_answers()