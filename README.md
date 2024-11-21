#### Setup Cassandra with Docker
1. Pull the Cassandra image from Docker Hub
```bash
docker pull cassandra:latest
```
2. Run the Cassandra container
```bash
docker run --rm -d --name cassandra --hostname cassandra --network cassandra -p 9042:9042 cassandra
```
3. Wait for 1-2 mins for the container to start and then connect to the CQL shell
```bash
docker exec -it cassandra cqlsh -u cassandra -p cassandra  localhost 9042
```
4. Run the following commands one by one in the CQL shell to create a keyspace and a table
```sql
CREATE KEYSPACE IF NOT EXISTS video_search
WITH REPLICATION = {
  'class' : 'SimpleStrategy',
  'replication_factor' : 1
};

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

```

#### Run the application
1. Clone the repository
```bash
git clone https://github.com/Swapnil-Kapale/FrameSeek
```

2. Setup the environment
```bash
cd FrameSeek
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

3. Run the application
```bash
python app2.py
```

4. The application will be running on http://localhost:5000


## Collaborators

- [Aabid](https://github.com/aabidk20)
- [Manoj](https://github.com/manojjamble)
