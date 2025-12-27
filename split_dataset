import os
import shutil
import random
import argparse

def split_dataset(dataset_path, out_root, train_ratio=0.8, seed=0):
    random.seed(seed)

    actions = [d for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, d))]
    train_dir = os.path.join(out_root, "ucf101_train_dataset")
    test_dir  = os.path.join(out_root, "ucf101_test_dataset")

    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    for action in actions:
        action_path = os.path.join(dataset_path, action)
        videos = [v for v in os.listdir(action_path) if os.path.isdir(os.path.join(action_path, v))]
        random.shuffle(videos)

        train_size = int(len(videos) * train_ratio)
        train_videos = videos[:train_size]
        test_videos  = videos[train_size:]

        train_action_dir = os.path.join(train_dir, action)
        test_action_dir  = os.path.join(test_dir, action)
        os.makedirs(train_action_dir, exist_ok=True)
        os.makedirs(test_action_dir, exist_ok=True)

        for video in train_videos:
            src = os.path.join(action_path, video)
            dst = os.path.join(train_action_dir, video)
            if os.path.exists(dst):
                shutil.rmtree(dst)
            shutil.copytree(src, dst)

        for video in test_videos:
            src = os.path.join(action_path, video)
            dst = os.path.join(test_action_dir, video)
            if os.path.exists(dst):
                shutil.rmtree(dst)
            shutil.copytree(src, dst)

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split keyframe dataset into train/test folders.")
    parser.add_argument("--dataset_dir", required=True, help="Path to keyframes dataset root (class folders).")
    parser.add_argument("--out_root", required=True, help="Output root where train/test folders will be created.")
    parser.add_argument("--train_ratio", type=float, default=0.8, help="Train split ratio (default: 0.8).")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for reproducibility.")
    args = parser.parse_args()

    split_dataset(args.dataset_dir, args.out_root, args.train_ratio, args.seed)
