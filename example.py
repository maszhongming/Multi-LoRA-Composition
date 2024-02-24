import torch
import argparse
from diffusers import DiffusionPipeline, StableDiffusionPipeline, AutoencoderKL
from diffusers import DPMSolverMultistepScheduler
from callbacks import make_callback

def get_example_prompt():
    prompt = "RAW photo, subject, 8k uhd, dslr, high quality, Fujifilm XT3, half-length portrait from knees up, scarlett, short red hair, blue eyes, school uniform, white shirt, red tie, blue pleated microskirt"
    negative_prompt = "extra heads, nsfw, deformed iris, deformed pupils, semi-realistic, cgi, 3d, render, sketch, cartoon, drawing, anime, text, cropped, out of frame, worst quality, low quality, jpeg artifacts, ugly, duplicate, morbid, mutilated, extra fingers, mutated hands, poorly drawn hands, poorly drawn face, mutation, deformed, blurry, dehydrated, bad anatomy, bad proportions, extra limbs, cloned face, disfigured, gross proportions, malformed limbs, missing arms, missing legs, extra arms, extra legs, fused fingers, too many fingers, long neck"
    return prompt, negative_prompt

def main(args):

    # set the prompts for image generation
    prompt, negative_prompt = get_example_prompt()

    # base model for the realistic style example
    model_name = 'SG161222/Realistic_Vision_V5.1_noVAE'

    # set base model
    pipeline = DiffusionPipeline.from_pretrained(
        model_name,
        custom_pipeline="MingZhong/StableDiffusionPipeline-with-LoRA-C",
        use_safetensors=True
    ).to("cuda")

    # set vae
    vae = AutoencoderKL.from_pretrained(
        "stabilityai/sd-vae-ft-mse",
    ).to("cuda")
    pipeline.vae = vae

    # set scheduler
    schedule_config = dict(pipeline.scheduler.config)
    schedule_config["algorithm_type"] = "dpmsolver++"
    pipeline.scheduler = DPMSolverMultistepScheduler.from_config(schedule_config)

    # initialize LoRAs
    # This example shows the composition of a character LoRA and a clothing LoRA
    pipeline.load_lora_weights(args.lora_path, weight_name="character_2.safetensors", adapter_name="character")
    pipeline.load_lora_weights(args.lora_path, weight_name="clothing_2.safetensors", adapter_name="clothing")
    cur_loras = ["character", "clothing"]

    # select the method for the composition
    if args.method == "merge":
        pipeline.set_adapters(cur_loras)
        switch_callback = None
    elif args.method == "switch":
        pipeline.set_adapters([cur_loras[0]])
        switch_callback = make_callback(switch_step=args.switch_step, loras=cur_loras)
    else:
        pipeline.set_adapters(cur_loras)
        switch_callback = None

    image = pipeline(
        prompt=prompt, 
        negative_prompt=negative_prompt,
        height=args.height,
        width=args.width,
        num_inference_steps=args.denoise_steps,
        guidance_scale=args.cfg_scale,
        generator=args.generator,
        cross_attention_kwargs={"scale": args.lora_scale},
        callback_on_step_end=switch_callback,
        lora_composite=True if args.method == "composite" else False
    ).images[0]

    image.save(args.save_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Example code for multi-LoRA composition'
    )

    # Arguments for composing LoRAs
    parser.add_argument('--method', default='switch',
                        choices=['merge', 'switch', 'composite'],
                        help='methods for combining LoRAs', type=str)
    parser.add_argument('--save_path', default='example.png',
                        help='path to save the generated image', type=str)
    parser.add_argument('--lora_path', default='models/lora/reality',
                        help='path to store all LoRAs', type=str)
    parser.add_argument('--lora_scale', default=0.8,
                        help='scale of each LoRA when generating images', type=float)
    parser.add_argument('--switch_step', default=5,
                        help='number of steps to switch LoRA during denoising, applicable only in the switch method', type=int)

    # Arguments for generating images
    parser.add_argument('--height', default=1024,
                        help='height of the generated images', type=int)
    parser.add_argument('--width', default=768,
                        help='width of the generated images', type=int)
    parser.add_argument('--denoise_steps', default=100,
                        help='number of the denoising steps', type=int)
    parser.add_argument('--cfg_scale', default=7,
                        help='scale for classifier-free guidance', type=float)
    parser.add_argument('--seed', default=11,
                        help='seed for generating images', type=int)

    args = parser.parse_args()
    args.generator = torch.manual_seed(args.seed)

    main(args)