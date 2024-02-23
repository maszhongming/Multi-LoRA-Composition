import json
import itertools

def load_lora_info(image_style, path='lora_info.json'):
    path = f"{image_style}_{path}"
    with open(path) as f:
        lora_info = json.loads(f.read())
    return lora_info

def get_prompt(image_style):
    if image_style == 'anime':
        prompt = "masterpiece, best quality"
        negative_prompt = "EasyNegative, extra fingers, extra limbs, fewer fingers, fewer limbs, multiple girls, multiple views, worst quality, low quality, depth of field, blurry, greyscale, 3D face, cropped, lowres, text, jpeg artifacts, signature, watermark, username, blurry, artist name, trademark, watermark, title, reference sheet, curvy, plump, fat, muscular female, strabismus, clothing cutout, side slit, tattoo, nsfw"
    else:
        prompt = "RAW photo, subject, 8k uhd, dslr, high quality, Fujifilm XT3, half-length portrait from knees up"
        negative_prompt = "extra heads, nsfw, deformed iris, deformed pupils, semi-realistic, cgi, 3d, render, sketch, cartoon, drawing, anime, text, cropped, out of frame, worst quality, low quality, jpeg artifacts, ugly, duplicate, morbid, mutilated, extra fingers, mutated hands, poorly drawn hands, poorly drawn face, mutation, deformed, blurry, dehydrated, bad anatomy, bad proportions, extra limbs, cloned face, disfigured, gross proportions, malformed limbs, missing arms, missing legs, extra arms, extra legs, fused fingers, too many fingers, long neck"
    return prompt, negative_prompt

def get_eval_prompt(combo):
    prompt = "I need assistance in comparatively evaluating two text-to-image models based on their ability to compose different elements into a single image. The elements and their key features are as follows:\n\n"
    cur_index = 1
    element_prompt = ""
    for lora in combo:
        ele_type = lora['id'].split('_')[0]
        name = lora['name']
        features = ', '.join(lora['trigger'])
        element_prompt += f"{cur_index}. {ele_type} ({name}): {features}\n"
        cur_index += 1
    prompt += element_prompt + '\n'
    prompt += "Please help me rate both given images on the following evaluation dimensions and criteria:\n\nComposition quality:\n- Score on a scale of 0 to 10, in 0.5 increments, where 10 is the best and 0 is the worst.\n- Deduct 3 points if any element is missing or incorrectly depicted.\n- Deduct 1 point for each missing or incorrect feature within an element.\n- Deduct 1 point for minor inconsistencies or lack of harmony between elements.\n- Additional deductions can be made for compositions that lack coherence, creativity, or realism.\n\nImage quality:\n- Score on a scale of 0 to 10, in 0.5 increments, where 10 is the best and 0 is the worst.\n- Deduct 3 points for each deformity in the image (e.g., extra limbs or fingers, distorted face, incorrect proportions).\n- Deduct 2 points for noticeable issues with texture, lighting, or color.\n- Deduct 1 point for each minor flaw or imperfection.\n- Additional deductions can be made for any issues affecting the overall aesthetic or clarity of the image.\n\nPlease format the evaluation as follows:\n\nFor Image 1:\n[Explanation of evaluation]\n\nFor Image 2:\n[Explanation of evaluation]\n\nScores:\nImage 1: Composition Quality: [score]/10, Image Quality: [score]/10\nImage 2: Composition Quality: [score]/10, Image Quality: [score]/10\n\nBased on the above guidelines, help me to conduct a step-by-step comparative evaluation of the given images. The scoring should follow two principles:\n1. Please evaluate critically.\n2. Try not to let the two models end in a tie on both dimensions."
    return prompt

def generate_combinations(lora_info, compos_num):
    """
    Generate all combinations of LoRA elements ensuring that each combination includes at least one 'character'.

    Args:
    lora_info (dict): A dictionary containing LoRA elements and their instances.
    compos_num (int): The number of elements to be included in each combination.

    Returns:
    list: A list of all possible combinations, each combination is a list of element instances.
    """
    elements = list(lora_info.keys())

    # Check if the composition number is greater than the number of element types
    if compos_num > len(elements):
        raise ValueError("The composition number cannot be greater than the number of elements.")

    all_combinations = []

    # Ensure that 'character' is always included in the combinations
    if 'character' in elements:
        # Remove 'character' from the list to avoid duplicating
        elements.remove('character')

        # Generate all possible combinations of the remaining element types
        selected_types = list(itertools.combinations(elements, compos_num - 1))

        # For each combination of types, generate all possible combinations of instances
        for types in selected_types:
            # Add 'character' to the current combination of types
            current_types = ['character', *types]

            # Gather instances for each type in the current combination
            instances = [lora_info[t] for t in current_types]

            # Create combinations of instances across the selected types
            for combination in itertools.product(*instances):
                all_combinations.append(combination)

    return all_combinations