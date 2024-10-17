CREATE KEYSPACE IF NOT EXISTS video_search 
WITH replication = {'class': 'SimpleStrategy', 'replication_factor': 1};

USE video_search;

CREATE TABLE IF NOT EXISTS videos (
    id UUID PRIMARY KEY,
    name TEXT,
    hash TEXT,
    upload_time TIMESTAMP
);

CREATE TABLE IF NOT EXISTS frame_embeddings (
    video_id UUID,
    frame_timestamp TEXT,
    embedding VECTOR <FLOAT, 512>,
    PRIMARY KEY (video_id, frame_timestamp)
);
