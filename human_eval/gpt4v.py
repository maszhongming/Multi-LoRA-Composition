import os
import re
import json
import copy
import openai
import base64
import logging
import requests
import argparse
from tqdm import tqdm
from os.path import join, exists
from openai import OpenAI

class GPT4V:
    def comparative_evaluate(
        self, prompt, image_1, image_2, max_tokens=2048, temperature=1.0, max_retries=5, **kwargs
    ):
        self.api_key = os.environ.get('OPENAI_API_KEY', None)
        self.client = OpenAI(api_key=self.api_key)
        retry_interval_exp = 1 
        retry_count = 0
        while retry_count < max_retries:
            try:
                response = self.client.chat.completions.create(
                    model="gpt-4-vision-preview",
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "text",
                                    "text": prompt,
                                },
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/png;base64,{image_1}"
                                    }
                                },
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/png;base64,{image_2}"
                                    }
                                },
                            ]
                        }
                    ],
                    max_tokens=max_tokens,
                    temperature=temperature,
                )
                return response.choices[0].message.content
            except openai.RateLimitError:
                logging.warning("OpenAI rate limit error. Retry!")
            except openai.APIConnectionError:
                logging.warning("OpenAI API connection error. Retry!")
            except openai.APITimeoutError:
                logging.warning("OpenAI timeout error. Retry!")
            except Exception as e:
                logging.error(f"Unexpected error: {e}")
                break

            # Simple backoff mechanism
            time.sleep(min(60, 0.5 * (2 ** retry_interval_exp)))
            retry_interval_exp += 1
            retry_count += 1

        return "An error occurred while processing the request."

# Function to encode the image
def encode_image(image_path):
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')

def get_eval_prompt(element_prompt):
    prompt = "I need assistance in comparatively evaluating two text-to-image models based on their ability to compose different elements into a single image. The elements and their key features are as follows:\n\n"
    prompt += element_prompt + '\n\n'
    prompt += "Please help me rate both given images on the following evaluation dimensions and criteria:\n\nComposition quality:\n- Score on a scale of 0 to 10, in 0.5 increments, where 10 is the best and 0 is the worst.\n- Deduct 3 points if any element is missing or incorrectly depicted.\n- Deduct 1 point for each missing or incorrect feature within an element.\n- Deduct 1 point for minor inconsistencies or lack of harmony between elements.\n- Additional deductions can be made for compositions that lack coherence, creativity, or realism.\n\nImage quality:\n- Score on a scale of 0 to 10, in 0.5 increments, where 10 is the best and 0 is the worst.\n- Deduct 3 points for each deformity in the image (e.g., extra limbs or fingers, distorted face, incorrect proportions).\n- Deduct 2 points for noticeable issues with texture, lighting, or color.\n- Deduct 1 point for each minor flaw or imperfection.\n- Additional deductions can be made for any issues affecting the overall aesthetic or clarity of the image.\n\nPlease format the evaluation as follows:\n\nFor Image 1:\n[Explanation of evaluation]\n\nFor Image 2:\n[Explanation of evaluation]\n\nScores:\nImage 1: Composition Quality: [score]/10, Image Quality: [score]/10\nImage 2: Composition Quality: [score]/10, Image Quality: [score]/10\n\nBased on the above guidelines, help me to conduct a step-by-step comparative evaluation of the given images. The scoring should follow two principles:\n1. Please evaluate critically.\n2. Try not to let the two models end in a tie on both dimensions."
    return prompt

def parse_scores(text):
    # Regular expression pattern to match scores, 
    pattern = r"image (\d): composition quality: ([\d\.]+)\/10.*?image quality: ([\d\.]+)\/10"
    
    # Find all matches in the evaluation text, case-insensitive
    matches = re.findall(pattern, text, re.IGNORECASE)

    # Check if exactly two images are present
    if len(matches) != 2:
        return False, "Expected scores for exactly two images"

    results = {}
    for match in matches:
        image_number, comp_quality, image_quality = match
        comp_quality = float(comp_quality)
        image_quality = float(image_quality)

        # Check if scores are within the valid range
        if not (0 <= comp_quality <= 10) or not (0 <= image_quality <= 10):
            return False, "Scores must be between 0 and 10"

        results[f'image {image_number}'] = {
            'composition quality': comp_quality,
            'image quality': image_quality
        }

    return True, results

def evaluate():
    image_n = 120 # number of images to evaluate
    gpt4v = GPT4V()
    
    # Load images
    image_path = 'images'
    # Initialize the list with a placeholder to ensure indexing starts at 1.
    images = [None]
    for i in range(1, image_n + 1):
        cur_image = encode_image(join(image_path, f'{i}.png'))
        images.append(cur_image)
    
    # Load prompts
    prompts = [None]
    with open('image_info.json') as f:
        image_info = json.loads(f.read())
    for i in range(len(image_info)):
        cur_prompt = '\n'.join(image_info[i]['prompt'])
        prompts.append(cur_prompt)

    # Comparative evaluation
    gpt4v = GPT4V()
    gpt4v_scores = [{} for _ in range(image_n + 1)]
    # i:     merge
    # i + 1: switch
    # i + 2: composite
    for i in tqdm(range(1, image_n + 1, 3)):
        merge_image     = images[i]
        switch_image    = images[i + 1]
        composite_image = images[i + 2]

        cur_prompt = get_eval_prompt(prompts[i])
        print(i)
        
        print("merge (image 1) vs switch (image 2)")
        retry_cnt = 0
        max_retries = 20
        while retry_cnt < max_retries:
            result = gpt4v.comparative_evaluate(cur_prompt, merge_image, switch_image)
            valid, scores = parse_scores(result)
            if valid == True:
                print(scores)
                # merge
                if len(gpt4v_scores[i]) == 0:
                    gpt4v_scores[i] = copy.deepcopy(scores['image 1'])
                else:
                    gpt4v_scores[i]['composition quality'] += scores['image 1']['composition quality']
                    gpt4v_scores[i]['image quality'] += scores['image 1']['image quality']
                # switch
                if len(gpt4v_scores[i + 1]) == 0:
                    gpt4v_scores[i + 1] = copy.deepcopy(scores['image 2'])
                else:
                    gpt4v_scores[i + 1]['composition quality'] += scores['image 2']['composition quality']
                    gpt4v_scores[i + 1]['image quality'] += scores['image 2']['image quality']
                break
            else:
                print(f"Retry for {i}.png")
                retry_cnt += 1
        if retry_cnt == max_retries:
            print(f"Can't get evaluation scores for {i}.png!")

        print("switch (image 1) vs merge (image 2)")
        retry_cnt = 0
        max_retries = 20
        while retry_cnt < max_retries:
            result = gpt4v.comparative_evaluate(cur_prompt, switch_image, merge_image)
            valid, scores = parse_scores(result)
            if valid == True:
                print(scores)
                # merge
                if len(gpt4v_scores[i]) == 0:
                    gpt4v_scores[i] = copy.deepcopy(scores['image 2'])
                else:
                    gpt4v_scores[i]['composition quality'] += scores['image 2']['composition quality']
                    gpt4v_scores[i]['image quality'] += scores['image 2']['image quality']
                # switch
                if len(gpt4v_scores[i + 1]) == 0:
                    gpt4v_scores[i + 1] = copy.deepcopy(scores['image 1'])
                else:
                    gpt4v_scores[i + 1]['composition quality'] += scores['image 1']['composition quality']
                    gpt4v_scores[i + 1]['image quality'] += scores['image 1']['image quality']
                break
            else:
                print(f"Retry for {i}.png")
                retry_cnt += 1
        if retry_cnt == max_retries:
            print(f"Can't get evaluation scores for {i}.png!")

        print("merge (image 1) vs composite (image 2)")
        retry_cnt = 0
        max_retries = 20
        while retry_cnt < max_retries:
            result = gpt4v.comparative_evaluate(cur_prompt, merge_image, composite_image)
            valid, scores = parse_scores(result)
            if valid == True:
                print(scores)
                # merge
                if len(gpt4v_scores[i]) == 0:
                    gpt4v_scores[i] = copy.deepcopy(scores['image 1'])
                else:
                    gpt4v_scores[i]['composition quality'] += scores['image 1']['composition quality']
                    gpt4v_scores[i]['image quality'] += scores['image 1']['image quality']
                # switch
                if len(gpt4v_scores[i + 2]) == 0:
                    gpt4v_scores[i + 2] = copy.deepcopy(scores['image 2'])
                else:
                    gpt4v_scores[i + 2]['composition quality'] += scores['image 2']['composition quality']
                    gpt4v_scores[i + 2]['image quality'] += scores['image 2']['image quality']
                break
            else:
                print(f"Retry for {i}.png")
                retry_cnt += 1
        if retry_cnt == max_retries:
            print(f"Can't get evaluation scores for {i}.png!")
        
        print("composite (image 1) vs merge (image 2)")
        retry_cnt = 0
        max_retries = 20
        while retry_cnt < max_retries:
            result = gpt4v.comparative_evaluate(cur_prompt, composite_image, merge_image)
            valid, scores = parse_scores(result)
            if valid == True:
                print(scores)
                # merge
                if len(gpt4v_scores[i]) == 0:
                    gpt4v_scores[i] = copy.deepcopy(scores['image 2'])
                else:
                    gpt4v_scores[i]['composition quality'] += scores['image 2']['composition quality']
                    gpt4v_scores[i]['image quality'] += scores['image 2']['image quality']
                # switch
                if len(gpt4v_scores[i + 2]) == 0:
                    gpt4v_scores[i + 2] = copy.deepcopy(scores['image 1'])
                else:
                    gpt4v_scores[i + 2]['composition quality'] += scores['image 1']['composition quality']
                    gpt4v_scores[i + 2]['image quality'] += scores['image 1']['image quality']
                break
            else:
                print(f"Retry for {i}.png")
                retry_cnt += 1
        if retry_cnt == max_retries:
            print(f"Can't get evaluation scores for {i}.png!")

    results = []
    for i in range(1, image_n + 1):
        cur = {}
        cur['index'] = i
        # merge
        if i % 3 == 1:
            cur['composition quality'] = gpt4v_scores[i]['composition quality'] / 4
            cur['image quality'] = gpt4v_scores[i]['image quality'] / 4
        else:
            cur['composition quality'] = gpt4v_scores[i]['composition quality'] / 2
            cur['image quality'] = gpt4v_scores[i]['image quality'] / 2
        results.append(cur)
    with open('gpt4v_results.json', 'w') as f:
        json.dump(results, f, indent=4, ensure_ascii=False)

if __name__ == "__main__":
    evaluate()