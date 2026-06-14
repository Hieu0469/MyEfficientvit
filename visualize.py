import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


## Test merge

import pycuda.driver as cuda
import tensorrt as trt
import numpy as np


from PIL import Image
import os
from pathlib import Path
import torch.nn.functional as F
import torch
cuda.init()
device = cuda.Device(0)
cuda_ctx = device.make_context()


# ── 1. Load TensorRT Engine ──────────────────────────────────────
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

def load_engine(engine_path):
    with open(engine_path, "rb") as f:
        runtime = trt.Runtime(TRT_LOGGER)
        return runtime.deserialize_cuda_engine(f.read())

# ── 2. Preprocess ────────────────────────────────────────────────
# Cityscapes standard normalization
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

def preprocess(image_path, input_h=512, input_w=1024):
    img = Image.open(image_path).convert("RGB")
    img = img.resize((input_w, input_h), Image.BILINEAR)
    img = np.array(img, dtype=np.float32) / 255.0
    img = (img - MEAN) / STD
    img = img.transpose(2, 0, 1)          # HWC → CHW
    img = np.ascontiguousarray(img[None])  # thêm batch dim
    return img

# ── 3. Inference với TensorRT ────────────────────────────────────
def infer(engine, input_data):
    cuda_ctx.push()
    try:
        context = engine.create_execution_context()
        inputs, outputs, bindings = [], [], []
        stream = cuda.Stream()

        for i in range(engine.num_io_tensors):
            name      = engine.get_tensor_name(i)
            dtype     = trt.nptype(engine.get_tensor_dtype(name))
            shape     = engine.get_tensor_shape(name)
            size      = trt.volume(shape)

            host_mem   = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            bindings.append(int(device_mem))

            if engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                inputs.append({"host": host_mem, "device": device_mem})
            else:
                outputs.append({"host": host_mem, "device": device_mem,
                                "shape": shape})  # ← lưu shape luôn

        np.copyto(inputs[0]["host"], input_data.ravel())
        cuda.memcpy_htod_async(inputs[0]["device"], inputs[0]["host"], stream)
        context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
        cuda.memcpy_dtoh_async(outputs[0]["host"], outputs[0]["device"], stream)
        stream.synchronize()

        # Trả về cả data lẫn shape thật
        return outputs[0]["host"], outputs[0]["shape"]

    finally:
        cuda_ctx.pop()  # luôn chạy dù có lỗi

# ── 4. Cityscapes label mapping ──────────────────────────────────
# 19 classes eval (ignore class 255)
CITYSCAPES_IGNORE = 255
NUM_CLASSES = 19

# Map từ trainId → evaluationId (nếu mask là raw label)
TRAIN_ID_MAP = {
    0:7, 1:8, 2:11, 3:12, 4:13, 5:17, 6:19, 7:20,
    8:21, 9:22, 10:23, 11:24, 12:25, 13:26, 14:27,
    15:28, 16:31, 17:32, 18:33
}

def load_gt_mask(mask_path, input_h=512, input_w=1024):
    mask = Image.open(mask_path)
    mask = mask.resize((input_w, input_h), Image.NEAREST)
    return np.array(mask, dtype=np.int64)

# ── 5. Tính mIoU ─────────────────────────────────────────────────
def compute_iou(pred, gt, num_classes=19, ignore_index=255):
    iou_list = []
    for cls in range(num_classes):
        pred_cls = (pred == cls)
        gt_cls   = (gt == cls) & (gt != ignore_index)
        
        intersection = (pred_cls & gt_cls).sum()
        union        = (pred_cls | gt_cls).sum()
        
        if union == 0:
            continue  # class không xuất hiện → bỏ qua
        iou_list.append(intersection / union)
    
    return np.mean(iou_list)

# ── 6. Eval loop ─────────────────────────────────────────────────
def evaluate(engine_path, cityscapes_val_dir, input_h=512, input_w=1024):
    engine = load_engine(engine_path)
    
    img_dir  = Path(cityscapes_val_dir) / "leftImg8bit/val"
    mask_dir = Path(cityscapes_val_dir) / "gtFine/val"
    
    image_paths = sorted(img_dir.rglob("*_leftImg8bit.png"))
    
    iou_scores = []
    for i, img_path in enumerate(image_paths):
        # Tìm mask tương ứng
        city = img_path.parent.name
        stem = img_path.stem.replace("_leftImg8bit", "")
        mask_path = mask_dir / city / f"{stem}_gtFine_labelTrainIds.png"
        
        if not mask_path.exists():
            continue
        
        # Inference
        inp  = preprocess(str(img_path), input_h, input_w)
        out  = infer(engine, inp)
        
        # Postprocess: output shape thường là (1, num_classes, H, W)
        out  = out.reshape(NUM_CLASSES, input_h, input_w)
        pred = np.argmax(out, axis=0)  # → (H, W)
        
        # Load GT
        gt = load_gt_mask(str(mask_path), input_h, input_w)
        
        # Tính IoU từng ảnh
        iou = compute_iou(pred, gt)
        iou_scores.append(iou)
        
        if i % 50 == 0:
            print(f"[{i}/{len(image_paths)}] Current mIoU: {np.mean(iou_scores):.4f}")
    
    miou = np.mean(iou_scores)
    print(f"\n✅ Final mIoU: {miou:.4f} ({miou*100:.2f}%)")
    return miou



# ── Cityscapes 19 class colors ────────────────────────────────────
CITYSCAPES_COLORS = np.array([
    [128,  64, 128],  # 0  road
    [244,  35, 232],  # 1  sidewalk
    [ 70,  70,  70],  # 2  building
    [102, 102, 156],  # 3  wall
    [190, 153, 153],  # 4  fence
    [153, 153, 153],  # 5  pole
    [250, 170,  30],  # 6  traffic light
    [220, 220,   0],  # 7  traffic sign
    [107, 142,  35],  # 8  vegetation
    [152, 251, 152],  # 9  terrain
    [ 70, 130, 180],  # 10 sky
    [220,  20,  60],  # 11 person
    [255,   0,   0],  # 12 rider
    [  0,   0, 142],  # 13 car
    [  0,   0,  70],  # 14 truck
    [  0,  60, 100],  # 15 bus
    [  0,  80, 100],  # 16 train
    [  0,   0, 230],  # 17 motorcycle
    [119,  11,  32],  # 18 bicycle
], dtype=np.uint8)

CLASS_NAMES = [
    "road", "sidewalk", "building", "wall", "fence",
    "pole", "traffic light", "traffic sign", "vegetation", "terrain",
    "sky", "person", "rider", "car", "truck",
    "bus", "train", "motorcycle", "bicycle"
]

def pred_to_color(pred_mask):
    """Chuyển mask (H, W) → ảnh màu (H, W, 3)"""
    color_mask = np.zeros((*pred_mask.shape, 3), dtype=np.uint8)
    for cls_id, color in enumerate(CITYSCAPES_COLORS):
        color_mask[pred_mask == cls_id] = color
    return color_mask

def visualize(image_path, pred_mask, gt_mask=None, alpha=0.5, save_path=None):
    """
    image_path : đường dẫn ảnh gốc
    pred_mask  : numpy array (H, W) - kết quả predict
    gt_mask    : numpy array (H, W) - ground truth (optional)
    alpha      : độ trong suốt của mask overlay
    """
    # Load ảnh gốc
    img = np.array(Image.open(image_path).convert("RGB"))

    
    img = np.array(img)

    pred_color = pred_to_color(pred)

    # ── Layout ───────────────────────────────────────────────────
    n_cols = 3 if gt_mask is not None else 2
    fig, axes = plt.subplots(1, n_cols, figsize=(7 * n_cols, 6))

    # Ảnh gốc
    axes[0].imshow(img)
    axes[0].set_title("Original Image", fontsize=13)
    axes[0].axis("off")

    # Prediction overlay
    overlay = (img * (1 - alpha) + pred_color * alpha).astype(np.uint8)
    axes[1].imshow(overlay)
    axes[1].set_title("Prediction", fontsize=13)
    axes[1].axis("off")

    # Ground truth (nếu có)
    if gt_mask is not None:
        gt_color = pred_to_color(gt_mask)
        gt_overlay = (img * (1 - alpha) + gt_color * alpha).astype(np.uint8)
        axes[2].imshow(gt_overlay)
        axes[2].set_title("Ground Truth", fontsize=13)
        axes[2].axis("off")

    # Legend
    patches = [
        mpatches.Patch(color=CITYSCAPES_COLORS[i] / 255, label=CLASS_NAMES[i])
        for i in range(19)
    ]
    fig.legend(
        handles=patches,
        loc="lower center",
        ncol=10,
        fontsize=8,
        bbox_to_anchor=(0.5, -0.05)
    )

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
        print(f"Saved → {save_path}")

    plt.show()

# ── Chạy ────────────────────────────────────────────────────────
input_h = 512
input_w = 1024
engine_path = '/home/hieu/tensorrt/efficientvit-seg-b0-city.trt'
img_path    = '/home/hieu/val_resized/frankfurt/frankfurt_000000_000294_leftImg8bit.png'

inp    = preprocess(str(img_path), input_h, input_w)
engine = load_engine(engine_path)

# Debug engine
print("=== ENGINE INFO ===")
for i in range(engine.num_io_tensors):
    name  = engine.get_tensor_name(i)
    shape = engine.get_tensor_shape(name)
    dtype = engine.get_tensor_dtype(name)
    mode  = engine.get_tensor_mode(name)
    print(f"  {'INPUT ' if mode == trt.TensorIOMode.INPUT else 'OUTPUT'} | {name} | shape={shape} | dtype={dtype}")

# Inference
out, out_shape = infer(engine, inp)
out = out.reshape(out_shape)   # (1, 19, 64, 128)


# Upsample về input resolution
pred = F.interpolate(torch.from_numpy(out),size=(input_h,input_w),mode="bicubic")
pred = pred.argmax(dim=1).squeeze().numpy()
print(f"Pred shape: {pred.shape}")

# Visualize
visualize(
    image_path=img_path,
    pred_mask=pred,
    save_path=None,
    alpha=1
)

# Cleanup CUDA
cuda_ctx.pop()
cuda_ctx.detach()
