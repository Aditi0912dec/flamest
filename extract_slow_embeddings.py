
from pytorchvideo.models.hub import slow_r50
import torch
import pickle
from PIL import Image
from torch.nn.utils.rnn import pad_sequence
import random
from sklearn.model_selection import train_test_split
import os
import numpy as np
import time
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report


import torchvision.transforms as transforms

from torchvision.transforms import Compose, Resize, ToTensor, Normalize


#from transformers import BlipProcessor, BlipForConditionalGeneration, AutoTokenizer, AutoModel







DATASET_DIR_TRAIN = "/scratch/cs21d001/action_recognition/ucf101_train_dataset"

CLASSES = os.listdir(DATASET_DIR_TRAIN)

# Load the pretrained model (replace with I3D if you have it)
r3d_model = slow_r50(pretrained=False)
checkpoint_path = "/scratch/cs21d001/action_recognition/slow_r50_weights.pth"
r3d_model.load_state_dict(torch.load(checkpoint_path, map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu")))
r3d_model.eval()  # Set to evaluation mode



def normalize_embeddings(embeddings):
    """Normalize text embeddings using L2 normalization."""
    return F.normalize(embeddings, p=2, dim=1)


def get_frames(action_classes, dataset_dir):
    video_paths, labels = [], []
    for class_index, action_class in enumerate(action_classes):
        action_class_path = os.path.join(dataset_dir, action_class)
        if not os.path.isdir(action_class_path):
            continue

        # each subfolder is a "video"
        video_folders = [os.path.join(action_class_path, d)
                         for d in os.listdir(action_class_path)
                         if os.path.isdir(os.path.join(action_class_path, d))]

        for vf in video_folders:
            video_paths.append(vf)
            labels.append(class_index)

        print(f"class={action_class} samples={len(video_folders)}")
    return video_paths, labels



###########
def frames_extraction(video_folder_path,f_list):
    '''
    This function will extract the required frames from a video folder.
    Args:
        video_folder_path: The path of the video folder in the disk, whose frames are to be extracted.
    Returns:
        frames_list: A list containing the resized and normalized frames of the video.
    '''
    # Declare a list to store video frames.
    frames_list = []
  
    # Get the list of image files in the video folder.
   
    image_files = sorted(os.listdir(video_folder_path))
    k = random.choice(range(len(image_files)))

  

    # Iterate through the image files and read the images.
    for image_file in image_files:
        # Construct the full image path.
        image_path = os.path.join(video_folder_path, image_file)
        frame_tensor = Image.open(image_path).convert("RGB")
        if image_file==image_files[k]:
            f_list.append(frame_tensor)

        transform = transforms.ToTensor()  # Converts image to range [0,1]
        frame_tensor = transform(frame_tensor)
        frame_tensor = torch.clamp(frame_tensor, 0, 1)
       
        frames_list.append(frame_tensor)
    return frames_list




device="cuda" if torch.cuda.is_available() else "cpu"


def extract_embeddings(model, batch_frames):
    # Define a local variable to store embeddings
    captured_embeddings = []

    # Hook function to capture output
    def hook_fn(module, input, output):
        captured_embeddings.append(output.view(output.shape[0], -1))  # Flatten output

    # Register the hook temporarily
    hook = model.avgpool.register_forward_hook(hook_fn)

    # Forward pass
    with torch.no_grad():
        model(batch_frames)

    # Remove hook after forward pass
    hook.remove()

    # Ensure we captured the embeddings
    if not captured_embeddings:
        raise RuntimeError("Failed to capture embeddings. Check hook or model.")

    return captured_embeddings[0]  # Return captured output

def get_img_embeddings(video_paths,f_list):
    video_embeddings = []  # To store video embeddings
      # To store random frame selections

    r3d_model.to(device)
    r3d_model.eval()

    # Iterate over the videos
    for video_path in video_paths:
        # Extract frames from video
        frames = frames_extraction(video_path,f_list)  # Assumes extract_frames is implemented
  
        # Convert frames to a batch tensor
        batch_frames = torch.stack([frame.clone().detach() for frame in frames]).unsqueeze(0)
        #batch_frames = torch.stack([torch.tensor(frame) for frame in frames]).unsqueeze(0)  # Shape: [1, T, C, H, W]
        batch_frames = batch_frames.permute(0, 2, 1, 3, 4)  # Reshape to [batch_size, C, T, H, W]
        batch_frames = batch_frames.to(device)
            
        embeddings = extract_embeddings(r3d_model, batch_frames)

    # Normalize embeddings
        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
        video_embeddings.append(embeddings)
    return video_embeddings

   

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Extract SlowR50 embeddings from keyframe folders (per video folder).")
    parser.add_argument("--dataset_dir", required=True, help="Path like ucf101_train_dataset (class/video/frame.jpg)")
    parser.add_argument("--checkpoint", required=True, help="Path to slow_r50_weights.pth")
    parser.add_argument("--out_dir", required=True, help="Output dir to save .npy/.pkl files")
    parser.add_argument("--device", default="cuda", choices=["cuda","cpu"], help="cuda or cpu")
    args = parser.parse_args()

    DATASET_DIR_TRAIN = args.dataset_dir
    CLASSES = [d for d in os.listdir(DATASET_DIR_TRAIN) if os.path.isdir(os.path.join(DATASET_DIR_TRAIN, d))]

    device = args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu"

    r3d_model = slow_r50(pretrained=False)
    r3d_model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    r3d_model.to(device).eval()

    os.makedirs(args.out_dir, exist_ok=True)

    f_list_train=[]
    video_path_train, gt_train = get_frames(CLASSES, DATASET_DIR_TRAIN)  # (see NOTE below)
    start=time.time()
    img_emb_train = get_img_embeddings(video_path_train, f_list_train)
    print(f"len f list train: {len(f_list_train)}")
    print(f"execution time : {time.time()-start}")

    np.save(os.path.join(args.out_dir, "ucf101_labels_train.npy"), np.array(gt_train))

    with open(os.path.join(args.out_dir, "ucf101_frame_list_train.pkl"), "wb") as f:
        pickle.dump(f_list_train, f)

    with open(os.path.join(args.out_dir, "ucf101_img_embeddings_train.pkl"), "wb") as f:
        pickle.dump(img_emb_train, f)

    print("IMAGE EMBEDDINGS EXTRACTED")










