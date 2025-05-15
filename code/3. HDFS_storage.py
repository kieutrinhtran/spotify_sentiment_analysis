import pyhdfs
import os
from pathlib import Path

# Đường dẫn tới thư mục data
data_dir = Path('data')

# Khởi tạo client kết nối đến HDFS
client = pyhdfs.HdfsClient(hosts='127.0.0.1:9870', user_name='khuong')

# Tạo thư mục trên HDFS nếu chưa tồn tại
hdfs_base_path = '/csvfinal'
try:
    client.mkdirs(hdfs_base_path)
except:
    pass

# Quét và tải lên tất cả các file CSV trong thư mục data
for csv_file in data_dir.glob('*.csv'):
    # Tạo đường dẫn đích trên HDFS
    hdfs_path = f'{hdfs_base_path}/{csv_file.name}'
    
    try:
        # Sao chép file từ local lên HDFS
        client.copy_from_local(str(csv_file), hdfs_path)
        print(f"Đã tải lên thành công: {csv_file.name} -> {hdfs_path}")
    except Exception as e:
        print(f"Lỗi khi tải file {csv_file.name}: {str(e)}")

print("Hoàn tất quá trình tải dữ liệu lên HDFS")