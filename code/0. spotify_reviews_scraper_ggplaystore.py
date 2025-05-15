# Import các thư viện cần thiết
from google_play_scraper import Sort, reviews  # Thư viện để scrape dữ liệu từ Google Play Store
import pandas as pd  # Thư viện để xử lý và lưu trữ dữ liệu dạng bảng
from datetime import datetime  # Thư viện để xử lý thời gian
import ftfy # Thư viện để sửa lỗi encoding
import html # Thư viện để sửa lỗi encoding

def scrape_spotify_ggplay_reviews(count=1000):
    """
    Hàm để thu thập đánh giá của ứng dụng Spotify từ Google Play Store
    Tham số:
        count (int): Số lượng đánh giá cần thu thập
    Trả về:
        DataFrame: Bảng dữ liệu pandas chứa các đánh giá
    """
    # ID của ứng dụng Spotify trên Google Play Store
    app_id = 'com.spotify.music'
    
    # Lấy dữ liệu đánh giá
    result, continuation_token = reviews(
        app_id,
        lang='en',  # Ngôn ngữ của đánh giá (tiếng Anh)
        country='us',  # Quốc gia (Mỹ)
        sort=Sort.NEWEST,  # Sắp xếp theo đánh giá mới nhất
        count=count  # Số lượng đánh giá cần lấy
    )
    
    # Chuyển đổi dữ liệu thành DataFrame
    df = pd.DataFrame(result)
    
    # Thêm cột thời gian scrape dữ liệu
    df['scraped_at'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    # Lưu dữ liệu vào file CSV với tên file bao gồm timestamp
    filename = f'data/spotify_reviews_ggplay_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
    df.to_csv(filename, index=False)
    print(f"Đã lưu {len(df)} đánh giá vào file {filename}")
    
    return df

def fix_encoding_df(df):
    # Sửa lỗi encoding và entity cho tất cả các cột dạng object (chuỗi)
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].apply(lambda x: ftfy.fix_text(html.unescape(x)) if isinstance(x, str) else x)
    return df

def read_and_concat(files, source):
    dfs = []
    for f in files:
        try:
            df = pd.read_csv(f, encoding='utf-8')
            df = fix_encoding_df(df)  # Sửa lỗi encoding, giữ emoji
            df = standardize_columns(df, source)
            dfs.append(df)
        except Exception as e:
            print(f"Lỗi đọc hoặc chuẩn hóa file {f}: {e}")
    if dfs:
        df_all = pd.concat(dfs, ignore_index=True)
        df_all = df_all.drop_duplicates()  # Lọc trùng toàn bộ dòng
        return df_all
    else:
        return pd.DataFrame()

if __name__ == "__main__":
    # Thu thập đánh giá
    reviews_df = scrape_spotify_ggplay_reviews(count=500)
    
    # Hiển thị thống kê cơ bản
    print("\nThống kê cơ bản:")
    print(f"Tổng số đánh giá đã thu thập: {len(reviews_df)}")
    print("\nPhân bố điểm đánh giá:")
    print(reviews_df['score'].value_counts().sort_index()) 