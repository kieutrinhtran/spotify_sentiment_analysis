# Spotify Sentiment Analysis

Dự án phân tích cảm xúc (sentiment analysis) của người dùng đối với ứng dụng Spotify trên cả hai nền tảng Google Play Store và Apple App Store.

## Cấu trúc dự án

```
.
├── code/                    # Mã nguồn chính
│   ├── 0. spotify_reviews_scraper_ggplaystore.py    # Thu thập đánh giá từ Google Play Store
│   ├── 1. spotify_reviews_scraper_applestore.py     # Thu thập đánh giá từ Apple App Store
│   ├── 2. pre_process.py                           # Tiền xử lý dữ liệu
│   ├── 3. sentiment_analysis.py                    # Phân tích cảm xúc
│   ├── 4. sentiment_comparison_two_platforms.py     # So sánh kết quả giữa hai nền tảng
│   └── 5. extract_negative_words.py                # Trích xuất từ ngữ tiêu cực
├── data/                   # Dữ liệu gốc
├── data-test/             # Dữ liệu test
├── logs/                  # File log
├── result/                # Kết quả phân tích
└── requirements.txt       # Các thư viện cần thiết
```

## Yêu cầu hệ thống

- Python 3.x
- Các thư viện Python được liệt kê trong `requirements.txt`

## Cài đặt

1. Tạo môi trường ảo:
```bash
python -m venv .venv
```

2. Kích hoạt môi trường ảo:
- Windows:
```bash
.venv\Scripts\activate
```
- Linux/Mac:
```bash
source .venv/bin/activate
```

3. Cài đặt các thư viện cần thiết:
```bash
pip install -r requirements.txt
```

## Cách sử dụng

Dự án được thực hiện theo các bước sau:

1. Thu thập dữ liệu:
   - Chạy `0. spotify_reviews_scraper_ggplaystore.py` để lấy đánh giá từ Google Play Store
   - Chạy `1. spotify_reviews_scraper_applestore.py` để lấy đánh giá từ Apple App Store

2. Tiền xử lý dữ liệu:
   - Chạy `2. pre_process.py` để làm sạch và chuẩn hóa dữ liệu

3. Phân tích cảm xúc:
   - Chạy `3. sentiment_analysis.py` để thực hiện phân tích cảm xúc

4. So sánh kết quả:
   - Chạy `4. sentiment_comparison_two_platforms.py` để so sánh kết quả giữa hai nền tảng

5. Trích xuất từ ngữ tiêu cực:
   - Chạy `5. extract_negative_words.py` để phân tích các từ ngữ tiêu cực

## Các thư viện chính

- pandas, numpy: Xử lý dữ liệu
- scikit-learn: Machine learning
- nltk, textblob, vaderSentiment: Phân tích cảm xúc
- transformers, torch: Deep learning
- matplotlib, seaborn: Trực quan hóa dữ liệu
- google-play-scraper: Thu thập dữ liệu từ Google Play Store