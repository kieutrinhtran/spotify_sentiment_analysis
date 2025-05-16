import subprocess
import time
import os
import logging
import socket
from pyngrok import ngrok, conf

# Thiết lập logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def is_port_in_use(port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('localhost', port)) == 0

def main():
    try:
        # Lấy token từ người dùng
        ngrok_token = input("Vui lòng nhập YOUR_NGROK_AUTH_TOKEN: ").strip() #2x8kFDTaBil3PDgNyNg56s27Ooz_4bTuHe1f4pCyJidYZwDFJ
        if not ngrok_token:
            print("❌ Không có token được cung cấp. Chương trình sẽ kết thúc.")
            return
            
        # Set token ngrok
        conf.get_default().auth_token = ngrok_token
        logging.info("Đã thiết lập ngrok token")

        # Kiểm tra port 8501
        if is_port_in_use(8501):
            logging.error("Port 8501 đang được sử dụng. Vui lòng đóng ứng dụng đang chạy trên port này.")
            return

        # Đóng tất cả tunnel cũ (nếu có)
        ngrok.kill()
        os.system("pkill -f streamlit")
        logging.info("Đã đóng các tunnel và tiến trình Streamlit cũ")

        # Get the current directory
        current_dir = os.path.dirname(os.path.abspath(__file__))
        app_path = os.path.join(current_dir, "app.py")

        # Khởi chạy Streamlit app ở background (port 8501, localhost)
        streamlit_cmd = [
            "streamlit", "run", app_path,
            "--server.port=8501",
            "--server.address=0.0.0.0"
        ]
        proc = subprocess.Popen(streamlit_cmd)
        logging.info("Đã khởi chạy Streamlit")

        print("⏳ Đang chờ Streamlit khởi động (5 giây)...")
        time.sleep(5)  # Chờ app chạy ổn định

        # Mở tunnel ngrok tới port 8501
        public_url = ngrok.connect(8501).public_url
        print(f"🚀 Ứng dụng của bạn đã chạy tại địa chỉ public:\n{public_url}\n")
        logging.info(f"Đã tạo public URL: {public_url}")

        try:
            # Đợi tiến trình Streamlit kết thúc
            proc.wait()
        except KeyboardInterrupt:
            logging.info("Nhận tín hiệu dừng từ người dùng")
            print("🛑 Đang dừng Streamlit và ngrok...")
            proc.terminate()
            ngrok.kill()
            logging.info("Đã dừng Streamlit và ngrok")

    except Exception as e:
        logging.error(f"Có lỗi xảy ra: {str(e)}")
        print(f"❌ Có lỗi xảy ra: {str(e)}")
        # Cleanup nếu có lỗi
        try:
            proc.terminate()
            ngrok.kill()
        except:
            pass

if __name__ == "__main__":
    main()
