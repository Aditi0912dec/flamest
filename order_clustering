import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Function to extract frames from a video
def extract_frames_from_video(video_path):
    cap = cv2.VideoCapture(video_path)
    original_frames = []
    resized_frames = []
    frame_ids = []
    frame_id = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        original_frame = cv2.resize(frame, (224, 224))  # Original frames at 320x240
        resized_frame = cv2.resize(frame, (64, 64))    # Resized frames for clustering
        original_frames.append(original_frame)
        resized_frames.append(resized_frame)
        frame_ids.append(frame_id)  
        frame_id += 1
    cap.release()
    return original_frames, resized_frames, frame_ids

# Function to apply K-means clustering to frames and extract cluster frames
def cluster_frames(resized_frames, frame_ids, num_clusters):
    resized_frames = np.array(resized_frames)
    resized_frames_flat = resized_frames.reshape(resized_frames.shape[0], -1)  # Flatten the frames

    kmeans = KMeans(n_clusters=num_clusters, random_state=0)
    kmeans.fit(resized_frames_flat)

    cluster_frame_indices = []
    for cluster in range(num_clusters):
        cluster_frames = np.where(kmeans.labels_ == cluster)[0]
        cluster_frames_list = list(cluster_frames)  # Convert to list for indexing
        # Select the frame with the minimum index in the cluster
        min_index = np.argmin([frame_ids[i] for i in cluster_frames_list])
        center_frame_index = cluster_frames_list[min_index]
        cluster_frame_indices.append(center_frame_index)

    # Sort the clusters based on the minimum frame index
    sorted_indices = np.argsort(cluster_frame_indices)
    sorted_cluster_frame_indices = [cluster_frame_indices[i] for i in sorted_indices]

    return sorted_cluster_frame_indices

# Function to save cluster frames to the output directory with category name
def save_cluster_frames(original_frames, output_folder, category_name, video_name, cluster_frame_indices):
    # Create subdirectory for the category
    category_folder = os.path.join(output_folder, category_name)
    os.makedirs(category_folder, exist_ok=True)

    # Create subdirectory for the video
    video_folder = os.path.join(category_folder, video_name)
    os.makedirs(video_folder, exist_ok=True)
    saved_paths = []
    for idx, frame_index in enumerate(cluster_frame_indices):
        frame = original_frames[frame_index]
        filename = f"{video_folder}/frame_{frame_index}.jpg"
        cv2.imwrite(filename, frame)
        saved_paths.append(filename)
    return saved_paths

# Function to process all videos in a specific category
def process_videos_in_category(category_path, output_folder, category_name, num_clusters=8):
    all_cluster_frame_paths = [] 
    for video_name in os.listdir(category_path):
        video_path = os.path.join(category_path, video_name)
        print(f"Processing {video_path}...")
        original_frames, resized_frames, frame_ids = extract_frames_from_video(video_path)
        cluster_frame_indices = cluster_frames(resized_frames, frame_ids, num_clusters=num_clusters)
        cluster_frame_paths = save_cluster_frames(original_frames, output_folder, category_name, video_name, cluster_frame_indices)
        all_cluster_frame_paths.extend(cluster_frame_paths)

    return all_cluster_frame_paths



# Example usage
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Extract KMeans keyframes from UCF-101 videos.")
    parser.add_argument("--dataset_dir", type=str, required=True, help="Path to UCF-101 root directory")
    parser.add_argument("--output_dir", type=str, required=True, help="Output folder to save keyframes")
    parser.add_argument("--num_clusters", type=int, default=8, help="Number of clusters/keyframes per video")
    args = parser.parse_args()

    DATASET_DIR = args.dataset_dir
    output_folder = args.output_dir
    num_clusters = args.num_clusters

    CLASSES = [d for d in os.listdir(DATASET_DIR) if os.path.isdir(os.path.join(DATASET_DIR, d))]

    for category_name in CLASSES:
        category_path = os.path.join(DATASET_DIR, category_name)
        process_videos_in_category(category_path, output_folder, category_name, num_clusters=num_clusters)
