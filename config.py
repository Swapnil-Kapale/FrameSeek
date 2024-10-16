import os
from transformers import CLIPProcessor, CLIPModel

# Folder paths
VIDEO_FOLDER = 'videos'
FRAME_FOLDER = 'frames'

# Create the folders if they don't exist
for folder in [VIDEO_FOLDER, FRAME_FOLDER]:
    if not os.path.exists(folder):
        os.makedirs(folder)

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")