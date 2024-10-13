from flask import Flask, request, render_template, send_from_directory,jsonify, url_for
from pathlib import Path
import pandas as pd
import os
import cv2
import numpy as np
import torch
from skimage.metrics import structural_similarity as ssim
from concurrent.futures import ThreadPoolExecutor, as_completed
from PIL import Image
from transformers import CLIPProcessor, CLIPModel


app = Flask(__name__)

# Configuration
UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'unique_frames_parallel'
REDUCED_QUALITY_FOLDER = 'updated_images'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure necessary directories exist
for folder in [UPLOAD_FOLDER, OUTPUT_FOLDER, REDUCED_QUALITY_FOLDER]:
    if not os.path.exists(folder):
        os.makedirs(folder)


# Initialize the CLIP model and processor
print("Initializing CLIP model and processor...")
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
print("CLIP model loaded successfully!")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
print("CLIP processor loaded successfully!")


def mse(imageA, imageB):
    # Compute the Mean Squared Error between two images
    err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
    err /= float(imageA.shape[0] * imageA.shape[1])
    return err

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

def extract_unique_frames_parallel(video_path, output_folder, num_workers=4, similarity_threshold=0.95):
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


# Function to compute image embeddings using CLIP
def compute_image_embeddings(image_paths):
    image_embeddings = []
    total_images = len(image_paths)
    print(f"Total images to process: {total_images}")

    for idx, image_path in enumerate(image_paths):
        image = Image.open(image_path).convert("RGB")
        inputs = processor(images=image, return_tensors="pt")

        with torch.no_grad():
            image_features = model.get_image_features(**inputs)

        # Normalize the embedding
        image_embedding = image_features / image_features.norm(p=2)
        image_embeddings.append(image_embedding.squeeze().cpu().numpy())

        # Print progress
        print(f"Processed image {idx + 1}/{total_images}: {os.path.basename(image_path)}")

    return np.vstack(image_embeddings)


# Function to compute text embeddings using CLIP
def compute_text_embeddings(list_of_strings):
    inputs = processor(text=list_of_strings, return_tensors="pt", padding=True)
    with torch.no_grad():
        text_features = model.get_text_features(**inputs)
    # Normalize the embedding
    return text_features / text_features.norm(p=2)


# Function to generate HTML for image results with similarity scores and timestamps
def get_html(results, height=200):
    html = "<div style='margin-top: 20px; display: flex; flex-wrap: wrap; justify-content: space-evenly'>"
    for image_path, score, timestamp in results:
        html2 = f"<div style='text-align: center; margin: 10px;'><img title='{os.path.basename(image_path)}' style='height: {height}px; margin-bottom: 5px' src='{image_path}'><br><span>Similarity Score: {score:.4f}</span><br><span>Timestamp: {timestamp}</span></div>"
        html += html2
    html += "</div>"
    return html


# Search for images using text queries with timestamps
# def image_search(query, image_embeddings, df, n_results=5):
#     text_embeddings = compute_text_embeddings([query]).cpu().numpy()
#     # Calculate cosine similarity between text and image embeddings
#     similarities = image_embeddings @ text_embeddings.T
#     # Get top-n results
#     results_indices = np.argsort(similarities[:, 0])[-1:-n_results - 1:-1]
#
#     # Get paths, scores, and timestamps for the results
#     results = [(df.iloc[i]['path'], similarities[i, 0], df.iloc[i]['timestamp']) for i in results_indices]
#     return results

# Update the image_search function to return relative paths
def image_search(query, image_embeddings, df, n_results=5):
    text_embeddings = compute_text_embeddings([query]).cpu().numpy()
    similarities = image_embeddings @ text_embeddings.T
    results_indices = np.argsort(similarities[:, 0])[-1:-n_results - 1:-1]
    results = [(os.path.relpath(df.iloc[i]['path'], REDUCED_QUALITY_FOLDER), 
                similarities[i, 0], 
                df.iloc[i]['timestamp']) for i in results_indices]
    return results


def secure_filename(filename):
    return str('sample' + '.' + Path(filename).suffix)


# @app.route('/')
# def index():
#     return render_template('index.html')
#
@app.route('/upload', methods=['POST'])
def upload_video():
    video_file = request.files['video']
    filename = secure_filename(video_file.filename)
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)

    # Save the video to the uploads directory
    video_file.save(file_path)

    # Parameters
    video_path = 'uploads/sample..mp4'  # Path to the video file
    output_folder = 'unique_frames_parallel'  # Folder to save unique frames
    similarity_threshold = 0.70  # Set the similarity threshold
    num_workers = 8  # Number of parallel threads

    extract_unique_frames_parallel(video_path, output_folder, num_workers, similarity_threshold)

    # reduce frame quality and resolution
    # Parameters
    input_folder = 'unique_frames_parallel'  # Folder containing extracted frames with chunk subfolders
    output_folder = 'updated_images'  # Folder to save frames with reduced quality and resolution
    quality = 70  # JPEG quality (reduce by 30%)
    scale_factor = 0.60  # Reduce resolution by 25%

    reduce_frame_quality_and_resolution(input_folder, output_folder, quality, scale_factor)


    image_folder = 'updated_images'  # Update this to your folder path
    image_paths = []
    for root, dirs, files in os.walk(image_folder):
        for file in files:
            if file.endswith(('.png', '.jpg', '.jpeg')):
                image_paths.append(os.path.join(root, file))

    # Compute embeddings for all images
    image_embeddings = compute_image_embeddings(image_paths)

    # Normalize embeddings for faster search
    image_embeddings = image_embeddings / np.linalg.norm(image_embeddings, axis=1, keepdims=True)

    # Example modification assuming your filenames contain timestamps
    df = pd.DataFrame(
        {'path': image_paths, 'timestamp': [os.path.splitext(os.path.basename(path))[0] for path in image_paths]})

    # save image embeddings and metadata in pickle files separately
    df.to_pickle("image_metadata.pkl")
    np.save("image_embeddings.pkl", image_embeddings)

    return 'Video uploaded successfully!'
#
# @app.route('/uploads/<filename>')
# def send_video(filename):
#     return send_from_directory(app.config['UPLOAD_FOLDER'], filename, as_attachment=True)
#
#
# @app.route('/search', methods=['POST'])
# def search_images():
#     query = request.form['query']
#     df = pd.read_pickle("image_metadata.pkl")
#     image_embeddings = np.load("image_embeddings.pkl.npy")
#
#     results = image_search(query, image_embeddings, df)
#     return jsonify([{'path': r[0], 'score': float(r[1]), 'timestamp': r[2]} for r in results])
#
# if __name__ == '__main__':
#     app.run(debug=True)


# Flask routes
@app.route('/')
def index():
    return render_template('index.html')



@app.route('/search', methods=['POST'])
def search_images():
    query = request.form['query']
    df = pd.read_pickle("image_metadata.pkl")
    image_embeddings = np.load("image_embeddings.pkl.npy")

    results = image_search(query, image_embeddings, df)
    return jsonify([{
        'path': url_for('serve_image', filename=r[0]),
        'score': float(r[1]),
        'timestamp': r[2]
    } for r in results])


@app.route('/uploads/<filename>')
def send_video(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename, as_attachment=True)


@app.route('/images/<path:filename>')
def serve_image(filename):
    return send_from_directory(REDUCED_QUALITY_FOLDER, filename)


if __name__ == '__main__':
    app.run(debug=True)