import os
import gc
import argparse
import pickle
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)


def build_transform(input_size: int):
    return T.Compose([
        T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def list_dirs(path: str):
    return sorted([d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))])


def list_frame_paths(video_dir: str):
    files = sorted([f for f in os.listdir(video_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))])
    return [os.path.join(video_dir, f) for f in files]


def sample_frames_uniform(frame_paths, max_frames: int):
    if len(frame_paths) <= max_frames:
        return frame_paths
    idx = np.linspace(0, len(frame_paths) - 1, max_frames).astype(int)
    return [frame_paths[i] for i in idx]


def load_frames(frame_paths, transform):
    px = []
    for p in frame_paths:
        img = Image.open(p).convert("RGB")
        px.append(transform(img))
    return torch.stack(px, dim=0)  # [T, C, H, W]


@torch.no_grad()
def internvideo_frame_embeddings(model, pixel_values, device, dtype):
    """
    pixel_values: [T, C, H, W]
    returns: [T, D] CLS embeddings
    """
    x = pixel_values.to(device=device, dtype=dtype)
    out = model.vision_model(x)
    if hasattr(out, "last_hidden_state"):
        cls = out.last_hidden_state[:, 0, :]  # [T, D]
    else:
        cls = out[0][:, 0, :]
    return cls.detach().cpu()


def main():
    parser = argparse.ArgumentParser(
        description="Extract InternVideo2 frame embeddings and mean-pool to video embeddings from folder-of-frames dataset."
    )

    # Required (no hardcoding)
    parser.add_argument("--model_id", required=True, help="HF model id or local path (e.g., OpenGVLab/InternVideo2_5_Chat_8B)")
    parser.add_argument("--dataset_dir", required=True, help="Dataset root: dataset/class/video/frame.jpg")
    parser.add_argument("--out_dir", required=True, help="Output directory to write embeddings + labels")
    parser.add_argument("--split_name", required=True, help="Name tag for outputs (e.g., hmdb51_train or ucf101_test)")

    # Also configurable (not hardcoded)
    parser.add_argument("--input_size", type=int, required=True, help="Image input size (e.g., 448)")
    parser.add_argument("--max_frames_per_video", type=int, required=True, help="Uniformly sample up to this many frames per video (e.g., 16)")
    parser.add_argument("--max_videos_per_class", type=int, default=0,
                        help="Optional cap per class (0 = use all videos).")
    parser.add_argument("--dtype", choices=["fp16", "fp32"], required=True, help="Model dtype: fp16 or fp32")
    parser.add_argument("--force_cpu", action="store_true", help="Force CPU even if CUDA is available")
    parser.add_argument("--save_format", choices=["pkl", "pt"], required=True, help="Save embeddings as pickle list or torch .pt tensor")
    parser.add_argument("--seed", type=int, required=True, help="Random seed (for reproducibility of any sampling/caps)")

    args = parser.parse_args()

    # Repro
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Memory cleanup
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    # Device
    device = "cpu" if args.force_cpu or (not torch.cuda.is_available()) else "cuda"

    # Dtype
    torch_dtype = torch.float16 if args.dtype == "fp16" else torch.float32

    # Load model/tokenizer (tokenizer often required for trust_remote_code models)
    _ = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)

    try:
        model = AutoModel.from_pretrained(
            args.model_id,
            trust_remote_code=True,
            torch_dtype=torch_dtype,
            device_map="auto" if device == "cuda" else None,
        ).eval()
        # If device_map is None and GPU, move explicitly
        if device == "cuda" and (next(model.parameters()).device.type != "cuda"):
            model = model.to("cuda")
    except RuntimeError as e:
        # If fp16 causes OOM, user should rerun with --dtype fp32 or --force_cpu
        raise RuntimeError(f"Model load failed: {e}\nTry --dtype fp32 or --force_cpu")

    model_device = next(model.parameters()).device
    model_dtype = next(model.parameters()).dtype
    print(f"✅ Model loaded on {model_device} | dtype={model_dtype}")

    transform = build_transform(args.input_size)

    classes = list_dirs(args.dataset_dir)
    if len(classes) == 0:
        raise ValueError(f"No class folders found in dataset_dir={args.dataset_dir}")

    video_embs = []
    labels = []

    for class_idx, cls in enumerate(classes):
        class_dir = os.path.join(args.dataset_dir, cls)
        video_dirs = list_dirs(class_dir)

        if args.max_videos_per_class and args.max_videos_per_class > 0:
            video_dirs = video_dirs[:args.max_videos_per_class]

        print(f"[{class_idx:03d}] {cls} | videos={len(video_dirs)}")

        for vd in video_dirs:
            video_dir = os.path.join(class_dir, vd)
            frame_paths = list_frame_paths(video_dir)
            if len(frame_paths) == 0:
                continue

            frame_paths = sample_frames_uniform(frame_paths, args.max_frames_per_video)

            pixel_values = load_frames(frame_paths, transform)  # [T,C,H,W]
            frame_emb = internvideo_frame_embeddings(model, pixel_values, model_device, model_dtype)  # [T,D]
            video_emb = frame_emb.mean(dim=0)  # [D]

            video_embs.append(video_emb)
            labels.append(class_idx)

    os.makedirs(args.out_dir, exist_ok=True)

    labels_np = np.asarray(labels, dtype=np.int64)
    labels_path = os.path.join(args.out_dir, f"{args.split_name}_labels.npy")
    np.save(labels_path, labels_np)

    if args.save_format == "pkl":
        emb_path = os.path.join(args.out_dir, f"{args.split_name}_internvideo2_emb.pkl")
        with open(emb_path, "wb") as f:
            pickle.dump(video_embs, f)
    else:
        # Save as a single tensor [N, D]
        emb_tensor = torch.stack(video_embs, dim=0) if len(video_embs) else torch.empty(0)
        emb_path = os.path.join(args.out_dir, f"{args.split_name}_internvideo2_emb.pt")
        torch.save(emb_tensor, emb_path)

    

if __name__ == "__main__":
    main()
