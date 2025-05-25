import torch
from diffusers.utils import load_image
from diffusers import StableDiffusionXLControlNetInpaintPipeline, ControlNetModel, UniPCMultistepScheduler
from transformers import pipeline
from PIL import Image
import numpy as np

# Configuration
ALL_EVS = [0, -2.5, -5]
LOWEST_EV = -5
SWITCH_LORA_TIMESTEP = 800
SEED = 200
PROMPT_BLACK = "a perfect black dark mirrored reflective chrome ball sphere"
PROMPT = "a perfect mirrored reflective chrome ball sphere"
NEGATIVE_PROMPT = "matte, diffuse, flat, dull"
IMAGE_URL = "https://raw.githubusercontent.com/DiffusionLight/image-other/main/input/wall.png" #make sure input image is 1024x1024

# if input is not 1024x1024, please do "black padding" to make it 1024x1024

# load pipeline
controlnet = ControlNetModel.from_pretrained("diffusers/controlnet-depth-sdxl-1.0", torch_dtype=torch.float16)
pipe = StableDiffusionXLControlNetInpaintPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    controlnet=controlnet,
    torch_dtype=torch.float16,
    add_watermarker=False
).to("cuda")


pipe.load_lora_weights("DiffusionLight/Flickr2K", adapter_name="turbo")
pipe.load_lora_weights("DiffusionLight/DiffusionLight", adapter_name="exposure")

# we prefer UNI PC for better quality with about 30 step
pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)

# we need to do bilinear interpolation on prepare_mask_latents, so we do monkey patch 
original_prepare_mask_latent = pipe.prepare_mask_latents
def new_prepare_mask_latent(self, *args, **kwargs):
    args = list(args)
    args[0] = torch.nn.functional.interpolate(
        args[0],
        size=(args[3] // self.vae_scale_factor, args[4] // self.vae_scale_factor),
        mode="bilinear", # Diffusers used "nearest" but we need bilinear for DiffusionLight
        align_corners=False,
    )
    args = tuple(args)
    return original_prepare_mask_latent(*args, **kwargs)
pipe.prepare_mask_latents = new_prepare_mask_latent.__get__(pipe, pipe.__class__)

# load depth model
depth_estimator = pipeline(task="depth-estimation", model="Intel/dpt-large", device="cuda", torch_dtype=torch.float16)

# prepare the input (image and depth)
init_image = load_image(IMAGE_URL)
depth_image = depth_estimator(images=init_image)['depth']

# create mask and depth map with mask for inpainting
def get_circle_mask(size=256):
    x = torch.linspace(-1, 1, size)
    y = torch.linspace(1, -1, size)
    y, x = torch.meshgrid(y, x)
    z = (1 - x**2 - y**2)
    mask = z >= 0
    return mask 

# function to apply LoRA weights
def apply_lora(pipe, adapter_name, lora_scale):
    pipe.unfuse_lora()     # unload previous lora weights (if any)
    pipe.set_adapters(adapter_name)
    pipe.fuse_lora(lora_scale=lora_scale)

# for image size 1024x1024 we add ball size 256x256 in the middle
depth_mask = get_circle_mask(256).numpy()

# add white circle into middle of depth map
depth = np.asarray(depth_image).copy()
depth[384:640, 384:640] = depth[384:640, 384:640] * (1 - depth_mask) + (depth_mask * 255)
depth_mask = Image.fromarray(depth)

# we create mask for inpaitning that dilate 20px (10px each side) around the ball
mask_image = np.zeros_like(depth)
inpaint_mask = get_circle_mask(256 + 20).numpy() 
mask_image[384 - 10:640 + 10, 384 - 10:640+10] = inpaint_mask * 255
mask_image = Image.fromarray(mask_image)

prompt_embeds, _, pooled_prompt_embeds, _ = pipe.encode_prompt(PROMPT)
prompt_embeds_black, _, pooled_prompt_embeds_black, _ = pipe.encode_prompt(PROMPT_BLACK)

for ev in ALL_EVS:
    # interpolate between the two prompts
    
    ratio = (ev - LOWEST_EV) / (0 - LOWEST_EV) #  0 = prompt, 1 = prompt_black
    prompt_embeds = prompt_embeds * ratio + prompt_embeds_black * (1 - ratio)
    pooled_prompt_embeds = pooled_prompt_embeds * ratio + pooled_prompt_embeds_black * (1 - ratio)

    # handle lora swapping between TurboLoRA and ExposureLoRA
    is_exposure_lora_loaded = False
    def callback(pipeline, i, t, callback_kwargs):
        global is_exposure_lora_loaded
        # t will start from 999 and decrease to 0, we only activate once at t=800
        if not is_exposure_lora_loaded and t <= SWITCH_LORA_TIMESTEP:
            apply_lora(pipeline, "exposure", lora_scale=0.75)
            is_exposure_lora_loaded = True
        return callback_kwargs
        
    apply_lora(pipe, "turbo", lora_scale=1.0)
    
    # DiffusionLight is sensitive to seed, seed should be same across EVs
    generator = torch.Generator(device="cuda").manual_seed(SEED)

    # run the pipeline
    output = pipe(
        prompt_embeds=prompt_embeds,
        pooled_prompt_embeds=pooled_prompt_embeds,
        negative_prompt=NEGATIVE_PROMPT,
        num_inference_steps=30,
        image=init_image,
        mask_image=mask_image,
        control_image=depth_mask,
        controlnet_conditioning_scale=0.5,
        callback_on_step_end=callback,
        generator=generator
    )
    is_exposure_lora_loaded = False

    # save output
    output["images"][0].save(f"output_ev-{int(np.abs(ev*10)):02d}.png")
