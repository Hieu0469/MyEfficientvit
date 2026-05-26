import onnxruntime as ort
import numpy as np
import time
import os

def measure_single_onnx(model_path, input_shape, iterations=100, warmup_iters=10):
    """Hàm lõi: Đo FPS và Latency cho 1 mô hình cụ thể"""
    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
    
    try:
        session = ort.InferenceSession(model_path, providers=providers)
    except Exception as e:
        return {"error": f"Lỗi tải mô hình: {e}"}

    input_name = session.get_inputs()[0].name
    
    # Tạo dữ liệu giả
    dummy_input = np.random.randn(*input_shape).astype(np.float32)

    # 1. Warm-up
    for _ in range(warmup_iters):
        session.run(None, {input_name: dummy_input})

    # 2. Benchmark
    start_time = time.time()
    for _ in range(iterations):
        session.run(None, {input_name: dummy_input})
    end_time = time.time()

    # 3. Tính toán
    total_time = end_time - start_time
    time_per_inference = (total_time / iterations) * 1000 # đổi ra ms
    fps = iterations / total_time
    
    # Trả về device đang chạy để biết có nhận GPU không
    device = session.get_providers()[0]

    return {
        "latency_ms": time_per_inference,
        "fps": fps,
        "device": device,
        "error": None
    }

def benchmark_models(models_config, iterations=100, warmup_iters=10):
    """Hàm quản lý: Chạy test nhiều mô hình và in bảng báo cáo"""
    print(f"Bắt đầu benchmark {len(models_config)} mô hình...\n")
    
    results = []
    
    for config in models_config:
        name = config.get("name", os.path.basename(config["path"]))
        path = config["path"]
        shape = config["shape"]
        
        print(f"-> Đang test: {name} | Shape: {shape} ...")
        
        res = measure_single_onnx(path, shape, iterations, warmup_iters)
        
        # Gộp thông tin cấu hình và kết quả lại
        res["name"] = name
        res["shape"] = str(shape)
        results.append(res)
        
    # --- IN BẢNG BÁO CÁO TỔNG HỢP ---
    print("\n" + "="*80)
    print(f"{'TÊN MÔ HÌNH':<30} | {'SHAPE':<18} | {'DEVICE':<22} | {'LATENCY (ms)':<12} | {'FPS':<8}")
    print("-" * 80)
    
    for r in results:
        if r["error"]:
            print(f"{r['name']:<30} | LỖI: {r['error']}")
        else:
            print(f"{r['name']:<30} | {r['shape']:<18} | {r['device']:<22} | {r['latency_ms']:<12.2f} | {r['fps']:<8.2f}")
    print("="*80)


# ==========================================
# CÁCH SỬ DỤNG
# ==========================================
if __name__ == "__main__":
    # Khai báo danh sách các mô hình cần test
    # Bạn có thể thiết lập shape khác nhau cho từng mô hình nếu muốn
    models_to_test = [
        {
            "name": "EfficientViT-B0",
            "path": r"e:\work\DL\new city\onnx\efficientvit-seg-b0-cityscapes.onnx",
            "shape": (1, 3, 512, 1024)
        },
        {
            "name": "EfficientViT-B1",
            "path": r"e:\work\DL\new city\onnx\efficientvit-seg-b1-cityscapes.onnx",
            "shape": (1, 3, 512, 1024)
        },
        {
            "name": "EfficientViT-B2",
            "path": r"e:\work\DL\new city\onnx\efficientvit-seg-b2-cityscapes.onnx",
            "shape": (1, 3, 512, 1024)
        },
        {
            "name": "EfficientViT-B3",
            "path": r"e:\work\DL\new city\onnx\efficientvit-seg-b3-cityscapes.onnx",
            "shape": (1, 3, 512, 1024)
        },
        {
            "name": "EfficientViT-L1",
            "path": r"e:\work\DL\new city\onnx\efficientvit-seg-l1-cityscapes.onnx",
            "shape": (1, 3, 512, 1024)
        },
        {
            "name": "EfficientViT-L2",
            "path": r"e:\work\DL\new city\onnx\efficientvit-seg-l2-cityscapes.onnx",
            "shape": (1, 3, 512, 1024)
        },
    ]

    # Chạy benchmark
    benchmark_models(models_to_test, iterations=10, warmup_iters=5)
