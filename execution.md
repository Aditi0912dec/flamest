# Execution Guide (End-to-End)

This document provides the complete execution pipeline to reproduce all experiments.

---

## 0. Environment Setup

(Optional) Create and activate a virtual environment:

    python3 -m venv .venv
    source .venv/bin/activate

Install dependencies:

    pip install -r requirements.txt

---

## A. FLAMeST Pipeline (Slow CNN + BLIP → Federated Learning)

### 1. Extract Keyframes using K-Means

Extract representative keyframes from raw videos.

    python3 extract_keyframes.py \
      --dataset_dir /path/to/input_videos \
      --output_dir /path/to/keyframes_output \
      --num_clusters 8

---

### 2. Split Dataset into Train and Test Sets

Split the keyframe dataset into training and testing splits.

    python3 split_dataset.py \
      --dataset_dir /path/to/keyframes_output \
      --out_root /path/to/split_output \
      --train_ratio 0.8 \
      --seed 0

---

### 3. Extract Slow CNN Embeddings (Train)

Extract spatio-temporal embeddings using a pretrained Slow R50 model.

    python3 extract_slow_embeddings.py \
      --dataset_dir /path/to/split_output/ucf101_train_dataset \
      --checkpoint /path/to/slow_r50_weights.pth \
      --out_dir /path/to/embeddings_output \
      --device cuda

---

### 4. Extract Slow CNN Embeddings (Test)

    python3 extract_slow_embeddings.py \
      --dataset_dir /path/to/split_output/ucf101_test_dataset \
      --checkpoint /path/to/slow_r50_weights.pth \
      --out_dir /path/to/embeddings_output \
      --device cuda

---

### 5. Extract BLIP Text and Cross-Modal Embeddings (Train)

Generate text embeddings and vision–language aligned embeddings using BLIP.

    python3 extract_blip_text_and_align_embeddings.py \
      --frame_list_pkl /path/to/embeddings_output/ucf101_frame_list_train.pkl \
      --blip_model_dir /path/to/blip-model \
      --processor_dir /path/to/blip-processor \
      --text_model_dir /path/to/blip_text_model \
      --out_dir /path/to/embeddings_output \
      --batch_size 128 \
      --save_every 1000

---

### 6. Extract BLIP Text and Cross-Modal Embeddings (Test)

    python3 extract_blip_text_and_align_embeddings.py \
      --frame_list_pkl /path/to/embeddings_output/ucf101_frame_list_test.pkl \
      --blip_model_dir /path/to/blip-model \
      --processor_dir /path/to/blip-processor \
      --text_model_dir /path/to/blip_text_model \
      --out_dir /path/to/embeddings_output \
      --batch_size 128 \
      --save_every 1000

---

### 7. Federated Training using FedAvg

Train a lightweight MLP classifier in a federated setting using FedAvg.

    python3 fedavg_mlp_train.py \
      --img_train_pkl /path/to/embeddings_output/img_train_embeddings.pkl \
      --align_train_pt /path/to/embeddings_output/align_train_embeddings.pt \
      --labels_train_npy /path/to/embeddings_output/labels_train.npy \
      --img_test_pkl /path/to/embeddings_output/img_test_embeddings.pkl \
      --align_test_pt /path/to/embeddings_output/align_test_embeddings.pt \
      --labels_test_npy /path/to/embeddings_output/labels_test.npy \
      --rounds 5 \
      --clients 4 \
      --alpha 0.6

---

## B. InternVideo2 Alternative Pipeline

For InternVideo2-based experiments, first run Steps 1 and 2, then execute the following.

### 8. Extract InternVideo2 Embeddings (Train)

    python3 extract_internvideo2_embeddings.py \
      --model_id OpenGVLab/InternVideo2_5_Chat_8B \
      --dataset_dir /path/to/split_output/hmdb51_train_dataset \
      --out_dir /path/to/embeddings_output \
      --split_name hmdb51_train \
      --input_size 448 \
      --max_frames_per_video 16 \
      --max_videos_per_class 0 \
      --dtype fp16 \
      --save_format pt \
      --seed 42

---

### 9. Extract InternVideo2 Embeddings (Test)

    python3 extract_internvideo2_embeddings.py \
      --model_id OpenGVLab/InternVideo2_5_Chat_8B \
      --dataset_dir /path/to/split_output/hmdb51_test_dataset \
      --out_dir /path/to/embeddings_output \
      --split_name hmdb51_test \
      --input_size 448 \
      --max_frames_per_video 16 \
      --max_videos_per_class 0 \
      --dtype fp16 \
      --save_format pt \
      --seed 42

---

## Notes

- All scripts are fully CLI-driven with no hardcoded paths.
- Expected dataset structure:

      dataset/
        class_name/
          video_name/
            frame_*.jpg

- For GPU memory issues:
  - Use --dtype fp32
  - Or add --force_cpu if supported by the script
