import os
import glob
import pandas as pd
import re
import unicodedata
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Tải dữ liệu NLTK nếu chưa có
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('stopwords', quiet=True)

# Khởi tạo stopwords tiếng Anh
stop = set(stopwords.words('english'))

def remove_emojis(text):
    """Xóa emoji khỏi văn bản."""
    return emoji.replace_emoji(text, '') if isinstance(text, str) else text

def remove_accented_chars(text):
    """Chuyển ký tự có dấu thành không dấu."""
    return unicodedata.normalize('NFKD', text).encode('ASCII', 'ignore').decode('ASCII') if isinstance(text, str) else text

def clean_text(text):
    """Tiền xử lý văn bản: xóa HTML, URL, ký tự đặc biệt, emoji, dấu, stopwords."""
    if not isinstance(text, str):
        return ''
    text = re.sub(r'<.*?>', ' ', text)
    text = re.sub(r'http\\S+', ' ', text)
    text = re.sub(r'[^a-zA-Z\\s]', ' ', text)
    text = re.sub(r'\\s+', ' ', text).strip().lower()
    text = remove_accented_chars(text)
    tokens = word_tokenize(text)
    tokens = [t for t in tokens if t not in stop]
    return ' '.join(tokens)

def standardize_columns(df, source):
    """
    Chuẩn hóa tên cột cho đồng nhất giữa hai nguồn dữ liệu.
    Luôn đảm bảo có đủ các cột mục tiêu.
    """
    df = df.copy()
    df['source'] = source
    # Map các biến thể về tên cột chuẩn
    merge_map = {
        'userName': 'author', 'author': 'author',
        'rating': 'rating', 'score': 'rating',
        'appVersion': 'version', 'version': 'version', 'reviewCreatedVersion': 'version',
        'review': 'content', 'content': 'content',
        'at': 'date', 'updated': 'date',
        'thumbsUpCount': 'thumbs_up', 'thumbs_up': 'thumbs_up',
        'replyContent': 'reply', 'reply': 'reply',
        # scraped_at giữ nguyên
    }
    for old_col, new_col in merge_map.items():
        if old_col in df.columns:
            if new_col in df.columns:
                df[new_col] = df[new_col].fillna(df[old_col])
            else:
                df[new_col] = df[old_col]
            if old_col != new_col:
                df.drop(columns=[old_col], inplace=True)
    # Đảm bảo luôn có đủ các cột mục tiêu
    for col in ['author', 'content', 'rating', 'version', 'date', 'scraped_at', 'source']:
        if col not in df.columns:
            df[col] = None
    return df

def display_data_info(df, title):
    """Hiển thị thông tin tổng quan về dataframe."""
    print(f"\n{'='*50}\n{title}\n{'='*50}")
    print(f"Kích thước: {df.shape}")
    print("Các cột:", list(df.columns))
    print("5 dòng đầu:")
    print(df.head())
    print("Số lượng theo nguồn:")
    print(df['source'].value_counts())
    print("\nThống kê cơ bản:")
    print(df.describe(include='all'))
    print(f"{'='*50}\n")

def read_and_concat(files, source):
    dfs = []
    for f in files:
        try:
            df = pd.read_csv(f)
            df = standardize_columns(df, source)
            dfs.append(df)
        except Exception as e:
            print(f"Lỗi đọc file {f}: {e}")
    if dfs:
        df_all = pd.concat(dfs, ignore_index=True)
        df_all = df_all.drop_duplicates()  # Lọc trùng toàn bộ dòng
        return df_all
    else:
        return pd.DataFrame()

def main():
    try:
        # Lấy tất cả file Apple và Google trong thư mục data/
        apple_files = glob.glob(os.path.join('data', 'spotify_reviews_apple_*_en.csv'))
        google_files = glob.glob(os.path.join('data', 'spotify_reviews_ggplay_*.csv'))
        if not apple_files and not google_files:
            raise FileNotFoundError("Không tìm thấy file dữ liệu đầu vào trong thư mục data/.")

        # Đọc và chuẩn hóa dữ liệu
        apple_df = read_and_concat(apple_files, 'apple_store') if apple_files else pd.DataFrame()
        google_df = read_and_concat(google_files, 'google_play') if google_files else pd.DataFrame()

        display_data_info(apple_df, "APPLE STORE")
        display_data_info(google_df, "GOOGLE PLAY STORE")

        # Gộp và xử lý
        combined_df = pd.concat([apple_df, google_df], ignore_index=True)
        if combined_df.empty:
            print("Không có dữ liệu để xử lý.")
            return

        combined_df['clean_content'] = combined_df['content'].apply(clean_text)
        combined_df = combined_df[combined_df['clean_content'].str.strip() != '']

        display_data_info(combined_df, "SAU KHI TIỀN XỬ LÝ")

        # Chỉ giữ lại các cột quan trọng, xóa reviewId, thêm id làm index
        keep_cols = ['author', 'content', 'clean_content', 'rating', 'version', 'date', 'scraped_at', 'source']
        save_df = combined_df[[col for col in keep_cols if col in combined_df.columns]]
        if 'reviewId' in save_df.columns:
            save_df = save_df.drop(columns=['reviewId'])

        # Hỏi người dùng có muốn lưu file không
        save = input("Bạn có muốn lưu kết quả vào file CSV không? (y/n): ")
        if save.lower() == 'y':
            save_df.insert(0, 'id', range(1, len(save_df) + 1))
            save_df.set_index('id').to_csv('data/combined_spotify_reviews.csv')
            print("Đã lưu file 'data/combined_spotify_reviews.csv' (chỉ giữ lại các cột quan trọng, có cột id làm index)")
        else:
            print("Không lưu kết quả.")

    except Exception as e:
        print(f"Lỗi: {e}")

if __name__ == "__main__":
    main()
