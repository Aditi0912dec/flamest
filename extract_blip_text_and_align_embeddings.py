import os, re, glob, pickle, time
import torch
import torch.nn.functional as F
from transformers import BlipProcessor, BlipForConditionalGeneration, BlipTextModel

# -----------------------------
# Helpers
# -----------------------------
def clean_text(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
    return text if text else "unknown description"

def normalize_embeddings(x: torch.Tensor) -> torch.Tensor:
    return F.normalize(x, p=2, dim=1)

# -----------------------------
# Main extraction
# -----------------------------
def get_cross_and_text_embeddings(
    frame_list,
    processor,
    blip_model,
    text_model,
    out_dir,
    batch_size=128,
    device=None,
    save_every=1000,
):
    os.makedirs(out_dir, exist_ok=True)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    blip_model.to(device).eval()
    text_model.to(device).eval()

    # Resume checkpoints (save both lists)
    ckpts = sorted(glob.glob(os.path.join(out_dir, "ckpt_*_caption.pt")))
    if ckpts:
        latest = ckpts[-1]
        start_idx = int(os.path.basename(latest).split("_")[1])
        caption_embedding = torch.load(os.path.join(out_dir, f"ckpt_{start_idx}_caption.pt"))
        align_embedding = torch.load(os.path.join(out_dir, f"ckpt_{start_idx}_align.pt"))
        print(f"Resuming from idx={start_idx} in {out_dir}")
    else:
        start_idx = 0
        caption_embedding, align_embedding = [], []

    n = len(frame_list)
    t0 = time.time()

    for i in range(start_idx, n, batch_size):
        batch_frames = frame_list[i : min(i + batch_size, n)]

        # 1) Caption generation (image -> text)
        with torch.no_grad():
            inputs_img = processor(images=batch_frames, return_tensors="pt").to(device)
            gen_ids = blip_model.generate(**inputs_img)

        captions = [processor.decode(g, skip_special_tokens=True) for g in gen_ids]
        captions = [clean_text(c) for c in captions]

        # 2) Cross-aligned embeddings from BLIP (image + text -> last_hidden_state CLS)
        with torch.no_grad():
            inputs_mm = processor(images=batch_frames, text=captions, return_tensors="pt",
                                  padding=True, truncation=True).to(device)
            out_mm = blip_model(**inputs_mm, output_hidden_states=False)
            mm_cls = out_mm.last_hidden_state[:, 0, :]          # [B, D]
            mm_cls = normalize_embeddings(mm_cls)

        # store as CPU tensors to avoid GPU memory growth
        align_embedding.extend(mm_cls.detach().cpu().split(1, dim=0))

        # 3) Text-only embeddings from BLIP text encoder (text -> CLS)
        with torch.no_grad():
            inputs_txt = processor(text=captions, return_tensors="pt",
                                   padding=True, truncation=True).to(device)
            out_txt = text_model(**inputs_txt)
            txt_cls = out_txt.last_hidden_state[:, 0, :]        # [B, D]
            txt_cls = normalize_embeddings(txt_cls)

        caption_embedding.extend(txt_cls.detach().cpu().split(1, dim=0))

        # checkpoint
        if (i + batch_size) % save_every == 0 or (i + batch_size) >= n:
            idx = min(i + batch_size, n)
            torch.save(caption_embedding, os.path.join(out_dir, f"ckpt_{idx}_caption.pt"))
            torch.save(align_embedding,   os.path.join(out_dir, f"ckpt_{idx}_align.pt"))
            print(f"Saved checkpoint at {idx}/{n} | elapsed {time.time()-t0:.1f}s")

        # cleanup GPU memory
        del inputs_img, gen_ids, inputs_mm, out_mm, inputs_txt, out_txt, mm_cls, txt_cls
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Final saves
    torch.save(caption_embedding, os.path.join(out_dir, "final_text_embeddings_train_ucf101.pt"))
    torch.save(align_embedding,   os.path.join(out_dir, "final_align_embeddings_train_ucf101.pt"))
    print("✅ Final embeddings saved.")

if __name__ == "__main__":
    # --------- paths (convert to argparse if publishing on GitHub) ----------
    TEXT_MODEL_DIR = "/scratch/cs21d001/action_recognition/blip_text_model"
    PROCESSOR_DIR  = "/scratch/cs21d001/action_recognition/content/blip-processor"
    BLIP_MODEL_DIR = "/scratch/cs21d001/action_recognition/content/blip-model"
    FRAME_LIST_PKL = "/scratch/cs21d001/action_recognition/vlm/ucf101_frame_list_train.pkl"
    OUT_DIR        = "/scratch/cs21d001/action_recognition/vlm"

    BATCH_SIZE = 128
    SAVE_EVERY = 1000

    device = "cuda" if torch.cuda.is_available() else "cpu"

    text_model = BlipTextModel.from_pretrained(TEXT_MODEL_DIR)
    processor  = BlipProcessor.from_pretrained(PROCESSOR_DIR)
    blip_model = BlipForConditionalGeneration.from_pretrained(BLIP_MODEL_DIR)

    with open(FRAME_LIST_PKL, "rb") as f:
        frame_list = pickle.load(f)

    get_cross_and_text_embeddings(
        frame_list=frame_list,
        processor=processor,
        blip_model=blip_model,
        text_model=text_model,
        out_dir=OUT_DIR,
        batch_size=BATCH_SIZE,
        device=device,
        save_every=SAVE_EVERY,
    )
