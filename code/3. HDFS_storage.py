import pyhdfs

# Đường dẫn tới file CSV trên máy cục bộ
output_file = 'D:\Final\spotify_sentiment_analysis\data\combined_spotify_reviews.csv'  # Thay bằng đường dẫn thực tế

# Khởi tạo client kết nối đến HDFS
client = pyhdfs.HdfsClient(hosts='127.0.0.1:9870', user_name='khuong')  # Thay 'dr.who' nếu cần

# Đường dẫn đích trên HDFS
hdfs_path = '/csvfinal/combined_spotify_reviews.csv'

# Sao chép file từ local lên HDFS
client.copy_from_local(output_file, hdfs_path)

print(f"Uploaded {output_file} to {hdfs_path} on HDFS")