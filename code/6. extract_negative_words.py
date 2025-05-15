import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from collections import Counter
import matplotlib.pyplot as plt

# Tải dữ liệu cần thiết
nltk.download('stopwords')
nltk.download('punkt')

def extract_negative_phrases(review_file, result_file, text_col, label_col, negative_labels=['neg', 'negative'], ngram=2, top_k=20):
    # 1. Đọc dữ liệu gốc chứa text
    df_reviews = pd.read_csv(review_file)
    
    # 2. Đọc kết quả phân tích sentiment chứa nhãn tiêu cực
    df_results = pd.read_csv(result_file)
    
    # 3. Ghép dữ liệu theo review_id
    merged_df = pd.merge(df_reviews, df_results, left_on='id', right_on='review_id')
    
    # 4. Lọc các review có nhãn tiêu cực
    neg_reviews = merged_df[merged_df[label_col].isin(negative_labels)][text_col].astype(str).tolist()
    
    # 5. Chuẩn bị stopwords tiếng Anh
    stop_words = set(stopwords.words('english'))
    
    all_ngrams = []
    for review in neg_reviews:
        tokens = word_tokenize(review.lower())
        # Loại bỏ stopwords và các token không phải chữ cái
        filtered_tokens = [t for t in tokens if t.isalpha() and t not in stop_words]
        
        # Tạo n-grams (ví dụ bigram)
        ngrams = list(nltk.ngrams(filtered_tokens, ngram))
        all_ngrams.extend(ngrams)
    
    # 6. Đếm tần suất xuất hiện n-grams
    counter = Counter(all_ngrams)
    most_common = counter.most_common(top_k)
    
    # Chuyển n-grams từ tuple sang chuỗi dễ đọc
    most_common_phrases = [(" ".join(gram), count) for gram, count in most_common]
    
    return most_common_phrases

def plot_phrases(phrases_freq):
    phrases, counts = zip(*phrases_freq)
    plt.figure(figsize=(12,6))
    plt.bar(phrases, counts)
    plt.title("Top cụm từ phổ biến trong đánh giá tiêu cực")
    plt.xticks(rotation=45)
    plt.show()

if __name__ == "__main__":
    review_file = 'data/combined_spotify_reviews.csv'
    result_file = 'result/lexicon_results.csv'  # hoặc 'result/distilbert_results.csv'
    text_col = 'clean_content'
    label_col = 'tb_label'  # với lexicon_results.csv
    negative_labels = ['neg']
    
    phrases = extract_negative_phrases(review_file, result_file, text_col, label_col, negative_labels, ngram=3, top_k=20)
    
    print("Top 20 cụm từ phổ biến trong đánh giá tiêu cực:")
    for phrase, count in phrases:
        print(f"{phrase}: {count}")
    
    plot_phrases(phrases)
