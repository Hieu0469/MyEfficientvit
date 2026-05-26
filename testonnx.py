import onnxruntime as ort
import numpy as np
import time

def measure_onnx_fps(model_path, input_shape, iterations=100, warmup_iters=10):
    print(f"Đang tải mô hình: {model_path} ...")
    
    # Ưu tiên chạy trên GPU, nếu không có sẽ tự lùi về CPU
    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
    try:
        session = ort.InferenceSession(model_path, providers=providers)
    except Exception as e:
        print("Lỗi khi tải mô hình:", e)
        return

    # Lấy thông tin input
    input_name = session.get_inputs()[0].name
    
    # Kiểm tra xem mô hình đang chạy trên thiết bị nào
    current_provider = session.get_providers()[0]
    print(f"Mô hình đang chạy trên: {current_provider}")

    # Tạo dữ liệu giả (dummy data) với kích thước chỉ định
    # Ví dụ: input_shape = (1, 3, 512, 1024)
    print(f"Khởi tạo dữ liệu giả với shape: {input_shape}")
    dummy_input = np.random.randn(*input_shape).astype(np.float32)

    # --- BƯỚC 1: WARM-UP (KHỞI ĐỘNG) ---
    print(f"Đang Warm-up ({warmup_iters} lần)...")
    for _ in range(warmup_iters):
        session.run(None, {input_name: dummy_input})

    # --- BƯỚC 2: BENCHMARK (ĐO ĐẠC) ---
    print(f"Đang đo FPS ({iterations} vòng lặp)...")
    
    start_time = time.time()
    for _ in range(iterations):
        session.run(None, {input_name: dummy_input})
    end_time = time.time()

    # --- BƯỚC 3: TÍNH TOÁN KẾT QUẢ ---
    total_time = end_time - start_time
    time_per_inference = total_time / iterations
    fps = iterations / total_time

    print("-" * 30)
    print("KẾT QUẢ BENCHMARK:")
    print(f"- Tổng thời gian chạy {iterations} ảnh: {total_time:.4f} giây")
    print(f"- Thời gian trung bình 1 ảnh (Latency): {time_per_inference * 1000:.2f} ms")
    print(f"- Tốc độ khung hình (FPS): {fps:.2f} FPS")
    print("-" * 30)

# ==========================================
# CÁCH SỬ DỤNG
# ==========================================

# 1. Đường dẫn tới file ONNX của bạn
model_path = r"e:\work\DL\new city\onnx\efficientvit-seg-b0-cityscapes.onnx"

# 2. Định nghĩa shape đầu vào (Batch_size, Channels, Height, Width)
# Thay đổi độ phân giải tại đây để xem FPS thay đổi thế nào nhé (vd: 512x1024)
input_shape = (1, 3, 512, 1024) 

# 3. Gọi hàm kiểm tra
measure_onnx_fps(model_path, input_shape, iterations=100, warmup_iters=20)