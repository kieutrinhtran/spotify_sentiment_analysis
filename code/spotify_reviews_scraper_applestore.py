import requests
import pandas as pd
from datetime import datetime
import time
from langdetect import detect, LangDetectException
import ftfy

def fix_encoding_df(df):
    # Sửa lỗi encoding cho tất cả các cột dạng object (chuỗi)
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].apply(lambda x: ftfy.fix_text(x) if isinstance(x, str) else x)
    return df

def scrape_spotify_reviews_apple_multi_country(count=1000, countries=None):
    if countries is None:
        countries = [
            'us', 'gb', 'au', 'ca', 'nz', 'za', 'jp', 'de', 'fr', 'vn', 
            'kr', 'ru', 'it', 'es', 'br', 'in', 'mx', 'sg', 'th', 'id', 
            'nl', 'se', 'tr', 'ch', 'pl', 'be', 'no', 'dk', 'fi', 'at', 
            'cz', 'pt', 'hu', 'gr', 'il', 'sa', 'ae', 'ar', 'cl', 'co', 'pe'
        ]
    app_id = 324684580
    reviews = []
    seen_ids = set()
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    for country in countries:
        print(f"\nĐang lấy đánh giá từ quốc gia: {country}")
        page = 1
        while len(reviews) < count * 1.3:  # lấy dư để lọc tiếng Anh
            try:
                url = (
                    f"https://itunes.apple.com/{country}/rss/customerreviews/"
                    f"page={page}/id={app_id}/sortby=mostrecent/json"
                )
                resp = requests.get(url, headers=headers)
                resp.raise_for_status()
                data = resp.json().get("feed", {})
                entries = data.get("entry", [])
                if not entries or len(entries) <= 1:
                    print(f"  Không còn đánh giá mới ở {country}")
                    break
                for entry in entries[1:]:
                    review_id = entry.get('id', {}).get('label', None)
                    if review_id and review_id in seen_ids:
                        continue
                    seen_ids.add(review_id)
                    reviews.append({
                        "author": entry["author"]["name"]["label"],
                        "title": entry.get("title", {}).get("label", ""),
                        "content": entry.get("content", {}).get("label", ""),
                        "rating": int(entry["im:rating"]["label"]),
                        "version": entry["im:version"]["label"],
                        "updated": entry["updated"]["label"],
                        "country": country
                    })
                    if len(reviews) >= count * 2:
                        break
                print(f"  Đã lấy được {len(reviews)} đánh giá...")
                page += 1
                time.sleep(1)
            except requests.exceptions.RequestException as e:
                print(f"  Lỗi khi lấy dữ liệu: {e}")
                break
            except Exception as e:
                print(f"  Lỗi không xác định: {e}")
                break
        if len(reviews) >= count * 2:
            break
    if reviews:
        df = pd.DataFrame(reviews)
        df = fix_encoding_df(df)  # Sửa lỗi encoding cho tất cả các cột dạng object (chuỗi)
        # Lọc các đánh giá tiếng Anh
        def is_english(text):
            try:
                return detect(text) == 'en'
            except LangDetectException:
                return False
        print("\nĐang lọc các đánh giá tiếng Anh...")
        mask = df['content'].apply(is_english) | df['title'].apply(is_english)
        df_en = df[mask].copy()
        df_en = df_en.head(count)  # chỉ lấy đủ số lượng yêu cầu
        df_en = fix_encoding_df(df_en)  # Đảm bảo fix cho cả df_en trước khi lưu
        df_en["scraped_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        filename = f"data/spotify_reviews_apple_{datetime.now().strftime('%Y%m%d_%H%M%S')}_en.csv"
        df_en.to_csv(filename, index=False, encoding='utf-8')
        print(f"\nĐã lưu {len(df_en)} đánh giá tiếng Anh vào file {filename}")
        return df_en
    else:
        print("Không lấy được đánh giá nào")
        return None

if __name__ == "__main__":
    # Thu thập 1000 đánh giá tiếng Anh từ nhiều quốc gia
    reviews_df = scrape_spotify_reviews_apple_multi_country(count=100)
    
    if reviews_df is not None:
        # Hiển thị thống kê cơ bản
        print("\nThống kê cơ bản:")
        print(f"Tổng số đánh giá đã thu thập: {len(reviews_df)}")
        print("\nPhân bố điểm đánh giá:")
        print(reviews_df['rating'].value_counts().sort_index()) 