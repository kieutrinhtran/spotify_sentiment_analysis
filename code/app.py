# Bước 2: Viết app vào file app.py (đặt tên file đồng nhất)
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from collections import Counter

# Tải dữ liệu nltk lần đầu (cache lại)
nltk.download('stopwords')
nltk.download('punkt')

st.set_page_config(layout="wide")
st.title("Phân tích sự khác biệt cảm xúc & từ khóa phổ biến trong đánh giá tiêu cực")

# --- Phần 1: Phân tích sự khác biệt cảm xúc giữa hai nền tảng ---
@st.cache_data
def load_sentiment_data():
    lexicon_df = pd.read_csv('result/lexicon_results.csv')
    distilbert_df = pd.read_csv('result/distilbert_results.csv')
    return lexicon_df, distilbert_df

lexicon_df, distilbert_df = load_sentiment_data()

st.header("Phân tích sự khác biệt cảm xúc giữa hai nền tảng")

# Tổng hợp số lượng sentiment theo source và nhãn
lexicon_summary = lexicon_df.groupby(['source', 'tb_label']).size().unstack(fill_value=0)
distilbert_summary = distilbert_df.groupby(['source', 'dl_label']).size().unstack(fill_value=0)

# Tính % theo từng source
lexicon_percent = lexicon_summary.div(lexicon_summary.sum(axis=1), axis=0) * 100
distilbert_percent = distilbert_summary.div(distilbert_summary.sum(axis=1), axis=0) * 100

st.subheader("Phân bố sentiment theo nền tảng (Lexicon)")
st.dataframe(lexicon_summary)
st.subheader("Tỷ lệ phần trăm sentiment theo nền tảng (Lexicon)")
st.dataframe(lexicon_percent.style.format("{:.2f}%"))

st.subheader("Phân bố sentiment theo nền tảng (DistilBERT)")
st.dataframe(distilbert_summary)
st.subheader("Tỷ lệ phần trăm sentiment theo nền tảng (DistilBERT)")
st.dataframe(distilbert_percent.style.format("{:.2f}%"))

# Vẽ biểu đồ stacked bar cho Lexicon
fig1, ax1 = plt.subplots(figsize=(8,5))
lexicon_percent.plot(kind='bar', stacked=True, ax=ax1)
ax1.set_ylabel('Tỷ lệ phần trăm (%)')
ax1.set_title('Phân bố Sentiment theo Nền tảng (Lexicon)')
st.pyplot(fig1)

# Vẽ biểu đồ stacked bar cho DistilBERT
fig2, ax2 = plt.subplots(figsize=(8,5))
distilbert_percent.plot(kind='bar', stacked=True, ax=ax2)
ax2.set_ylabel('Tỷ lệ phần trăm (%)')
ax2.set_title('Phân bố Sentiment theo Nền tảng (DistilBERT)')
st.pyplot(fig2)


# --- Phần 2: Trích xuất các cụm từ khóa phổ biến trong đánh giá tiêu cực ---

@st.cache_data
def extract_negative_phrases(review_file, result_file, text_col, label_col,
                             negative_labels=['neg', 'negative'], ngram=2, top_k=20):
    # Đọc dữ liệu
    df_reviews = pd.read_csv(review_file)
    df_results = pd.read_csv(result_file)
    
    # Ghép dữ liệu theo review_id / id
    merged_df = pd.merge(df_reviews, df_results, left_on='id', right_on='review_id')
    
    # Lọc review tiêu cực
    neg_reviews = merged_df[merged_df[label_col].isin(negative_labels)][text_col].astype(str).tolist()
    
    # Stopwords tiếng Anh
    stop_words = set(stopwords.words('english'))
    
    all_ngrams = []
    for review in neg_reviews:
        tokens = word_tokenize(review.lower())
        filtered_tokens = [t for t in tokens if t.isalpha() and t not in stop_words]
        ngrams = list(nltk.ngrams(filtered_tokens, ngram))
        all_ngrams.extend(ngrams)
    
    counter = Counter(all_ngrams)
    most_common = counter.most_common(top_k)
    
    # Chuyển tuple ngram sang chuỗi
    most_common_phrases = [(" ".join(gram), count) for gram, count in most_common]
    return most_common_phrases

def plot_phrases(phrases_freq):
    phrases, counts = zip(*phrases_freq)
    fig, ax = plt.subplots(figsize=(12,6))
    ax.bar(phrases, counts)
    ax.set_title("Top cụm từ phổ biến trong đánh giá tiêu cực")
    ax.tick_params(axis='x', rotation=45)
    st.pyplot(fig)

st.header("Trích xuất cụm từ khóa phổ biến trong đánh giá tiêu cực")

# Các file dữ liệu
review_file = 'data/combined_spotify_reviews.csv'
result_file = 'result/lexicon_results.csv'  # hoặc distilbert_results.csv tùy bạn chọn
text_col = 'clean_content'
label_col = 'tb_label'  # đổi thành 'dl_label' nếu dùng distilbert_results.csv

negative_labels = ['negative']  # lexicon dùng 'neg', bạn có thể điều chỉnh

phrases = extract_negative_phrases(review_file, result_file, text_col, label_col,
                                   negative_labels=negative_labels, ngram=3, top_k=20)

if phrases:
    st.subheader("Top 20 cụm từ phổ biến trong đánh giá tiêu cực")
    for phrase, count in phrases:
        st.write(f"- {phrase}: {count}")
    plot_phrases(phrases)
else:
    st.warning("Không tìm thấy đánh giá tiêu cực để phân tích từ khóa.")
