from core import get_cassandra_connection
from flask import Flask, request, render_template, send_from_directory,jsonify, url_for
from config import VIDEO_FOLDER, FRAME_FOLDER
import os
from datetime import datetime
import uuid
from core import (
    calculate_video_hash,
    extract_unique_frames_parallel,
    compute_image_embeddings,
)

app = Flask(__name__)


# @app.route('/upload', methods=['GET', 'POST'])
# def upload_video():
#     if request.method == 'POST':
#         video_file = request.files['video']
#         video_path = os.path.join(VIDEO_FOLDER, video_file.filename)
#         video_file.save(video_path)

#         video_hash = calculate_video_hash(video_path)
#         extract_unique_frames_parallel(video_path, video_hash)

#         # Insert video info into DB
#         session = get_cassandra_connection()
#         session.execute("INSERT INTO videos (id, name, upload_time, hash) VALUES (%s, %s, %s, %s)", (uuid.uuid4(), video_file.filename, datetime.now(), video_hash))

#         return 'Video uploaded successfully!'
#     return render_template('upload.html')


@app.route('/upload', methods=['GET', 'POST'])
def upload_video():
    if request.method == 'POST':
        video_file = request.files['video']
        video_path = os.path.join(VIDEO_FOLDER, video_file.filename)
        video_file.save(video_path)

        video_hash = calculate_video_hash(video_path)
        session = get_cassandra_connection()
        existing_video = session.execute("SELECT * FROM videos WHERE hash = %s ALLOW FILTERING", (video_hash,)).one()
        if existing_video:
            return 'Video already exists in the database'
        else:
            extract_unique_frames_parallel(video_path, video_hash)
            frames_folder = os.path.join(FRAME_FOLDER, video_hash)
            frames = [os.path.join(frames_folder, frame) for frame in os.listdir(frames_folder)]
            res = compute_image_embeddings(frames)
            id = uuid.uuid4()
            session.execute("INSERT INTO videos (id, name, upload_time, hash) VALUES (%s, %s, %s, %s)", (id, video_file.filename, datetime.now(), video_hash))
            print(res[0]);
            for item in res:
                session.execute("INSERT INTO frame_embeddings (video_id, frame_timestamp, embedding) VALUES (%s, %s, %s)", (id, item['filename'], item['embedding'].tolist()))
            return 'Video uploaded and indexed successfully!'
    return render_template('upload.html')


@app.route('/videos')
def list_videos():
    session = get_cassandra_connection()
    videos = session.execute("SELECT * FROM videos")
    return render_template('videos.html', videos=videos)


if __name__ == '__main__':
    app.run(debug=True)