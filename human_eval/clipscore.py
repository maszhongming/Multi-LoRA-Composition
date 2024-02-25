import json
import torch
import numpy as np
from tqdm import tqdm
from os.path import join
from functools import partial
from diffusers.utils import load_image
from torchmetrics.functional.multimodal import clip_score

clip_score_fn = partial(clip_score, model_name_or_path="openai/clip-vit-base-patch16")
image_n = 120 # number of images to evaluate

def calculate_clip_score(images, prompts):
    images_int = (images * 255).astype("uint8")
    clip_score = clip_score_fn(torch.from_numpy(images_int).permute(0, 3, 1, 2), prompts).detach()
    return round(float(clip_score), 4)

# Load images
# The size of anime and realistic style images is different.
image_path = 'images'
images = []
for i in range(1, image_n + 1):
    cur_image = load_image(join(image_path, f'{i}.png'))
    # scale pixel values to [0, 1]
    cur_image = np.array(cur_image, dtype=np.float32) / 255.0
    images.append(cur_image)

# Load prompts
prompts = []
with open('image_info.json') as f:
    image_info = json.loads(f.read())
for i in range(len(image_info)):
    cur_prompt = ', '.join(content.split(': ')[1] for content in image_info[i]['prompt'])
    prompts.append(cur_prompt)

# Calculate CLIP score
scores = []
for i in tqdm(range(image_n)):
    cur_image = np.expand_dims(images[i], axis=0)
    cur_prompt = [prompts[i]]
    sd_clip_score = calculate_clip_score(cur_image, cur_prompt)
    scores.append(sd_clip_score)

# Save results
result = []
for i in range(len(scores)):
    cur = {}
    cur['index'] = i + 1
    cur['clip_score'] = scores[i]
    result.append(cur)
with open('clip_results.json', 'w') as f:
    json.dump(result, f, indent=4, ensure_ascii=False)
