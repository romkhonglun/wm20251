#!/bin/bash
# run.sh

# ================= CẤU HÌNH DDP (QUAN TRỌNG) =================
# Số lượng GPU bạn muốn sử dụng (Ví dụ: 2)
NUM_PROC=1

# Danh sách GPU ID (Ví dụ: dùng GPU 0 và 1 thì điền "0,1")
export CUDA_VISIBLE_DEVICES=0

# ================= CẤU HÌNH TÀI NGUYÊN =================
# LƯU Ý: Với DDP, KHÔNG NÊN dùng cpulimit vì sẽ gây lệch pha (desync) giữa các GPU.
# Thay vào đó, hãy kiểm soát CPU bằng OMP_NUM_THREADS.
# Công thức gợi ý: (Tổng số core cho phép / Số lượng GPU)
# Ví dụ: Được dùng 8 core, chạy 2 GPU -> set = 4.
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

# Master Port: Cổng giao tiếp giữa các GPU (đổi nếu bị trùng/lỗi address already in use)
MASTER_PORT=29500

# ================= CẤU HÌNH ĐƯỜNG DẪN =================
PYTHON_SCRIPT="train.py"
DATASET_PATH="/home2/congnh/wm/input/ebnerd_small"
EMBEDDING_PATH="/home2/congnh/wm/input/EB_NeRD_small_embedded_text.pkl"

# ================= HYPERPARAMETERS =================
# Batch size này là PER GPU.
# Nếu set 128 và chạy 2 GPU -> Global Batch Size thực tế = 256
BATCH_SIZE=256
HISTORY_SIZE=30
NEGATIVE_RATIO=4
EPOCHS=10
SEED=42

# Tạo chuỗi arguments
SCRIPT_ARGS="--dataset_path $DATASET_PATH \
             --embedding_path $EMBEDDING_PATH \
             --batch_size $BATCH_SIZE \
             --history_size $HISTORY_SIZE \
             --negative_ratio $NEGATIVE_RATIO \
             --epochs $EPOCHS \
             --seed $SEED"

# ================= THỰC THI =================

echo "----------------------------------------------------------------"
echo "Bắt đầu training DDP với cấu hình:"
echo "GPUs: $NUM_PROC (Devices: $CUDA_VISIBLE_DEVICES)"
echo "Dataset: $DATASET_PATH"
echo "Batch Size per GPU: $BATCH_SIZE"
echo "----------------------------------------------------------------"

# Lệnh chạy chuẩn cho DDP: torchrun
# --nproc_per_node: Số lượng GPU sử dụng
# --rdzv_endpoint: Địa chỉ master node (chạy local thì là localhost)

torchrun \
    --nproc_per_node=$NUM_PROC \
    --master_port=$MASTER_PORT \
    $PYTHON_SCRIPT $SCRIPT_ARGS