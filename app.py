import streamlit as st
from gensim.models import Word2Vec
import os

# Streamlit UI 설정
st.title("📘 Word2Vec 유사 단어 찾기 & 단어 연산")
st.sidebar.title("🛠 사용법")
st.sidebar.markdown("""
1. 텍스트를 입력하면 유사한 단어를 확인할 수 있습니다.  
2. 단어 연산 (예: 한국 - 서울 + 도쿄)을 통해 새로운 단어 추론이 가능합니다.
""")

# 모델 경로 및 로딩
model_path = "word2vec.model"
model = None
if os.path.exists(model_path):
    try:
        model = Word2Vec.load(model_path)
        st.success("✅ Word2Vec 모델이 성공적으로 로드되었습니다.")
    except Exception as e:
        st.error(f"❌ 모델 로드 실패: {e}")
else:
    st.warning("⚠️ Word2Vec 모델이 존재하지 않습니다. 먼저 학습하고 저장하세요.")

# 모델이 있을 때만 UI 동작
if model:
    # 1. 유사 단어 찾기
    user_input = st.text_input("🔍 유사 단어를 찾고 싶은 단어를 입력하세요:")

    if user_input:
        try:
            similar_words = model.wv.most_similar(user_input, topn=5)
            st.markdown(f"**입력한 단어 `{user_input}` 와 유사한 단어들:**")
            for word, similarity in similar_words:
                st.write(f"- {word} (유사도: {similarity:.4f})")
        except KeyError:
            st.error("입력한 단어는 모델에 존재하지 않습니다.")

    # 2. 단어 연산
    operation_input = st.text_input("🧠 단어 연산 입력 (예: 한국 - 서울 + 도쿄)")

    if operation_input:
        try:
            tokens = operation_input.split()
            if len(tokens) == 5 and tokens[1] == '-' and tokens[3] == '+':
                positive = [tokens[0], tokens[4]]
                negative = [tokens[2]]
                result = model.wv.most_similar(positive=positive, negative=negative, topn=1)
                st.success(f"🧾 결과 단어: **{result[0][0]}** (유사도: {result[0][1]:.4f})")
            else:
                st.warning("❗ 입력 형식을 정확히 지켜주세요. 예: 한국 - 서울 + 도쿄")
        except KeyError as e:
            st.error(f"입력한 단어 중 모델에 존재하지 않는 단어가 있습니다: {e}")
else:
    st.stop()  # 모델이 없으면 앱 실행 중단
