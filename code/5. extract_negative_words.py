import pandas as pd
from collections import Counter
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import matplotlib.pyplot as plt

# Tải dữ liệu stopwords và tokenizer
nltk.download('stopwords')
nltk.download('punkt')

def extract_negative_keywords(review_file, result_file, text_col, label_col, negative_labels=['neg', 'negative']):
    # 1. Đọc dữ liệu gốc chứa text
    df_reviews = pd.read_csv(review_file)
    
    # 2. Đọc kết quả phân tích sentiment chứa nhãn tiêu cực
    df_results = pd.read_csv(result_file)
    
    # 3. Ghép dữ liệu theo review_id
    merged_df = pd.merge(df_reviews, df_results, left_on='id', right_on='review_id')
    
    # 4. Lọc các review có nhãn tiêu cực
    neg_reviews = merged_df[merged_df[label_col].isin(negative_labels)][text_col].astype(str).tolist()
    
    # 5. Khởi tạo stopwords tiếng Anh
    stop_words = set(stopwords.words('english'))
    
    all_tokens = []
    for review in neg_reviews:
        tokens = word_tokenize(review.lower())
        tokens = [t for t in tokens if t.isalpha() and t not in stop_words]
        all_tokens.extend(tokens)
    
    # 6. Đếm tần suất từ
    counter = Counter(all_tokens)
    
    # 7. Lấy top 20 từ khóa phổ biến
    most_common = counter.most_common(20)
    
    return most_common

def plot_keywords(keywords_freq):
    words, counts = zip(*keywords_freq)
    plt.figure(figsize=(12,6))
    plt.bar(words, counts)
    plt.title("Top 20 từ khóa phổ biến trong đánh giá tiêu cực")
    plt.xticks(rotation=45)
    plt.show()

if __name__ == "__main__":
    review_file = 'data/combined_spotify_reviews.csv'  # file chứa cột 'clean_content'
    result_file = 'result/lexicon_results.csv'          # file chứa nhãn sentiment lexicon
    text_col = 'clean_content'
    label_col = 'tb_label'  # với lexicon_results.csv là tb_label, với distilbert_results.csv có thể là dl_label
    negative_labels = ['neg']  # Lexicon-based nhãn tiêu cực là 'neg'
    
    keywords = extract_negative_keywords(review_file, result_file, text_col, label_col, negative_labels)
    
    print("Top 20 từ khóa phổ biến trong đánh giá tiêu cực:")
    for word, count in keywords:
        print(f"{word}: {count}")
    
    plot_keywords(keywords)
