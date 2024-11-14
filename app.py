from core import get_cassandra_connection
from flask import (
    Flask,
    request,
    render_template,
    send_from_directory,
    jsonify,
    url_for,
    redirect,
)
from config import VIDEO_FOLDER, FRAME_FOLDER
import os
from datetime import datetime
import numpy as np
import uuid
from core import (
    calculate_video_hash,
    extract_unique_frames_parallel,
    compute_image_embeddings,
    compute_text_embeddings,
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


@app.route("/upload", methods=["GET", "POST"])
def upload_video():
    if request.method == "POST":
        video_file = request.files["video"]
        video_path = os.path.join(VIDEO_FOLDER, video_file.filename)
        video_file.save(video_path)

        video_hash = calculate_video_hash(video_path)
        session = get_cassandra_connection()
        existing_video = session.execute(
            "SELECT * FROM videos WHERE hash = %s ALLOW FILTERING", (video_hash,)
        ).one()
        if existing_video:
            return "Video already exists in the database"  # TODO: Would be better to redirect to the video page directly
        else:
            extract_unique_frames_parallel(video_path, video_hash)
            frames_folder = os.path.join(FRAME_FOLDER, video_hash)
            frames = [
                os.path.join(frames_folder, frame)
                for frame in os.listdir(frames_folder)
            ]
            res = compute_image_embeddings(frames)
            id = uuid.uuid4()
            session.execute(
                "INSERT INTO videos (id, name, upload_time, hash) VALUES (%s, %s, %s, %s)",
                (id, video_file.filename, datetime.now(), video_hash),
            )
            for item in res:
                session.execute(
                    "INSERT INTO frame_embeddings (video_id, frame_timestamp, embedding) VALUES (%s, %s, %s)",
                    (id, item["filename"], item["embedding"].tolist()),
                )
            return redirect("/videos")
    return render_template("upload.html")


@app.route("/videos")
def list_videos():
    session = get_cassandra_connection()
    videos = session.execute("SELECT * FROM videos")
    return render_template("videos.html", videos=videos)


@app.route("/videos/<video_id>")
def video_page(video_id):
    print("Getting video")
    session = get_cassandra_connection()
    video = session.execute(
        "SELECT * FROM videos WHERE id = %s", (uuid.UUID(video_id),)
    ).one()
    if video:
        return render_template("video_detail.html", video=video)
    else:
        return "Video not found"


@app.route("/videos/<video_id>/search", methods=["POST"])
def search_frames(video_id):
    session = get_cassandra_connection()
    query = request.form["query"]
    print(video_id, query)
    video = session.execute(
        "SELECT * FROM videos WHERE id = %s", (uuid.UUID(video_id),)
    ).one()
    if video:
        frames = list(
            session.execute(
                "SELECT * FROM frame_embeddings WHERE video_id = %s",
                (uuid.UUID(video_id),),
            )
        )
        frame_timestamps = [frame.frame_timestamp for frame in frames]
        frame_embeddings = [frame.embedding for frame in frames]
        text_embeddings = compute_text_embeddings([query]).cpu().numpy()
        print(np.array(frame_embeddings).shape, text_embeddings.shape)
        similarities = np.array(frame_embeddings) @ text_embeddings.T
        results = sorted(
            zip(frame_timestamps, similarities[:, 0]), key=lambda x: x[1], reverse=True
        )[:5]
        print(results)
        # return jsonify(results)
        return render_template("search_results.html", video=video, results=results)
    else:
        return "Video not found"


@app.route("/frames/<path:filename>")
def serve_frame(filename):
    return send_from_directory(".", filename)


@app.route("/get-video/<path:filename>")
def serve_video(filename):
    return send_from_directory(VIDEO_FOLDER, filename)


@app.route("/")
def home():
    return redirect("/videos")


if __name__ == "__main__":
    app.run(debug=True)
