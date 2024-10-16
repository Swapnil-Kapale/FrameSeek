import os
from cassandra.cluster import Cluster
from cassandra.auth import PlainTextAuthProvider
from skimage.metrics import structural_similarity as ssim
import hashlib
import cv2
from config import FRAME_FOLDER, VIDEO_FOLDER, model, processor
from concurrent.futures import ThreadPoolExecutor, as_completed
import torch
from PIL import Image
import numpy as np

# Cassandra connection
def get_cassandra_connection():
    auth_provider = PlainTextAuthProvider(username='cassandra', password='cassandra')
    cluster = Cluster(['localhost'], auth_provider=auth_provider, port=9042)
    session = cluster.connect('video_search')
    return session


def calculate_video_hash(video_path):
    with open(video_path, 'rb') as f:
        video_hash = hashlib.sha256(f.read()).hexdigest()
    return video_hash

def save_frame(frame, output_folder, timestamp):
    # Save the frame with timestamp as the name
    filename = os.path.join(output_folder, f"frame_{timestamp:.2f}.jpg")
    cv2.imwrite(filename, frame)

def process_video_chunk(video_path, start_frame, end_frame, output_folder, similarity_threshold=0.95):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)  # Start processing from the specific frame

    ret, prev_frame = cap.read()
    frame_saved = 0
    frame_count = start_frame
    if not ret:
        return frame_saved

    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    timestamp = frame_count / fps
    save_frame(prev_frame, output_folder, timestamp)
    frame_saved += 1

    while ret and frame_count < end_frame:
        ret, current_frame = cap.read()
        if not ret:
            break

        current_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
        similarity = ssim(prev_gray, current_gray)

        # Save the frame if the similarity is below the threshold
        if similarity < similarity_threshold:
            timestamp = frame_count / fps
            save_frame(current_frame, output_folder, timestamp)
            frame_saved += 1
            prev_gray = current_gray

        frame_count += 1

    cap.release()
    return frame_saved


def reduce_frame_quality_and_resolution(input_folder, output_folder, quality=90, scale_factor=0.75):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for root, dirs, files in os.walk(input_folder):
        for file in files:
            if file.endswith('.jpg'):
                # Read the frame
                frame_path = os.path.join(root, file)
                frame = cv2.imread(frame_path)

                if frame is None:
                    continue  # If image loading fails, skip it

                # Reduce the resolution by scale factor
                width = int(frame.shape[1] * scale_factor)
                height = int(frame.shape[0] * scale_factor)
                resized_frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)

                # Generate corresponding output folder structure
                relative_path = os.path.relpath(root, input_folder)
                output_chunk_folder = os.path.join(output_folder, relative_path)

                if not os.path.exists(output_chunk_folder):
                    os.makedirs(output_chunk_folder)

                # Save the frame with reduced quality
                output_path = os.path.join(output_chunk_folder, file)
                cv2.imwrite(output_path, resized_frame, [int(cv2.IMWRITE_JPEG_QUALITY), quality])

                print(f"Processed and saved {output_path}")


def extract_unique_frames_parallel(video_path, video_hash, num_workers=4, similarity_threshold=0.7):
    output_folder = os.path.join(FRAME_FOLDER, video_hash)
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    chunk_size = total_frames // num_workers

    # Define chunks (start and end frame for each worker)
    chunks = [(i * chunk_size, (i + 1) * chunk_size if (i + 1) * chunk_size < total_frames else total_frames)
              for i in range(num_workers)]

    # Run in parallel using ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = []
        for i, (start_frame, end_frame) in enumerate(chunks):
            chunk_output_folder = os.path.join(output_folder, f"chunk_{i}")
            if not os.path.exists(chunk_output_folder):
                os.makedirs(chunk_output_folder)
            futures.append(executor.submit(process_video_chunk, video_path, start_frame, end_frame, chunk_output_folder, similarity_threshold))

        # Collect results and print progress
        total_saved = 0
        for future in as_completed(futures):
            total_saved += future.result()
            print(f"Chunk processed, total unique frames saved so far: {total_saved}")

    print(f"Parallel extraction complete. {total_saved} unique frames saved in {output_folder}.")


# Function to compute image embeddings using CLIP
def compute_image_embeddings(image_paths):
    image_embeddings = []
    print(image_paths)

    # we have nested folders, flatten
    image_paths = [os.path.join(path, file) for path in image_paths for file in os.listdir(path)]

    total_images = len(image_paths)
    

    for idx, image_path in enumerate(image_paths):
        image = Image.open(image_path).convert("RGB")
        inputs = processor(images=image, return_tensors="pt")

        with torch.no_grad():
            image_features = model.get_image_features(**inputs)

        # Normalize the embedding
        image_embedding = image_features / image_features.norm(p=2)
        # add filename + embedding both to image_embeddings
        res = {'filename': os.path.basename(image_path), 'embedding': image_embedding.squeeze().cpu().numpy()}
        image_embeddings.append(res)
        # image_embeddings.append(image_embedding.squeeze().cpu().numpy())

        # Print progress
        print(f"Processed image {idx + 1}/{total_images}: {os.path.basename(image_path)}")

    return image_embeddings;
