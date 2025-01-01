import os
os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"

import torch
import argparse
from PIL import Image, ImageDraw
from omegaconf import OmegaConf
from transformers import CLIPImageProcessor, ViTImageProcessor, CLIPVisionModelWithProjection, LlamaTokenizer, ViTMAEModel
import numpy as np
import gc

# Import required model components
from src.models.unet import UNetMangaModel
from src.models.resampler import Resampler
from src.models.qwen_resampler import QwenResampler
from src.models.mllm.seed_x import ContinuousLVLM
from src.models.mllm.modeling_llama_xformer import LlamaForCausalLM
from src.pipelines.pipeline_diffsensei import DiffSenseiPipeline

# Constants
PAGE_WIDTH = 2048
PAGE_HEIGHT = 2896
PANEL_COLOR = "blue"
CHARACTER_COLORS = ["green", "red", "yellow", "purple"]
DIALOG_COLOR = "black"

# MLLM Constants
BOI_TOKEN = '<img>'
EOI_TOKEN = '</img>'
IMG_TOKEN = '<img_{:05d}>'
BBOX_START_TOKEN = '<box_start>'
BBOX_END_TOKEN = '<box_end>'

# Initialize processors
clip_image_processor = CLIPImageProcessor()
magi_image_processor = ViTImageProcessor()

def get_mllm_inputs(prompt, tokenizer):
    instruction = ""
    image_tokens = BOI_TOKEN + ''.join([IMG_TOKEN.format(int(item)) for item in range(64)]) + EOI_TOKEN
    instruction += prompt + '\n'
    instruction += image_tokens + '\n'

    input_ids = [tokenizer.bos_token_id] + tokenizer.encode(instruction, add_special_tokens=False)
    boi_token_id = tokenizer.encode(BOI_TOKEN, add_special_tokens=False)[1]
    eoi_token_id = tokenizer.encode(EOI_TOKEN, add_special_tokens=False)[1]
    ids_cmp_mask = [False] * len(input_ids)

    input_ids = torch.tensor(input_ids, dtype=torch.long)
    ids_cmp_mask = torch.tensor(ids_cmp_mask, dtype=torch.bool)

    boi_idx = torch.where(input_ids == boi_token_id)[0].tolist()
    eoi_idx = torch.where(input_ids == eoi_token_id)[0].tolist()

    for i in range(1):
        ids_cmp_mask[boi_idx[i] + 1 : eoi_idx[i]] = True

    return input_ids, ids_cmp_mask

def result_generation(
    pipeline,
    tokenizer_mllm,
    agent_model,
    prompt,
    height,
    width,
    num_samples,
    seed,
    ip_images,
    ip_bbox,
    dialog_bbox,
    num_inference_steps,
    guidance_scale,
    negative_prompt,
    ip_scale,
    mllm_scale
):
    generator = torch.Generator('cuda:0').manual_seed(seed)
    input_ids, ids_cmp_mask = get_mllm_inputs(prompt, tokenizer_mllm)    

    while len(ip_images) < pipeline.unet.config.max_num_ips:
        image = Image.new('RGB', (224, 224), (0, 0, 0))
        ip_images.append(image)
        ip_bbox.append([0.0, 0.0, 0.0, 0.0])

    clip_ip_images = clip_image_processor(images=ip_images, return_tensors="pt").pixel_values
    magi_ip_images = magi_image_processor(images=ip_images, return_tensors="pt").pixel_values
    clip_image_embeds = pipeline.image_encoder(clip_ip_images.to('cuda:0', dtype=pipeline.dtype), output_hidden_states=True).hidden_states[-2]
    magi_image_embeds = pipeline.magi_image_encoder(magi_ip_images.to('cuda:0', dtype=pipeline.dtype)).last_hidden_state[:, 0]
    image_embeds = pipeline.image_proj_model(clip_image_embeds.unsqueeze(0), magi_image_embeds)
    image_embeds = image_embeds[:, pipeline.unet.config.num_vision_tokens:, :]

    output = agent_model.generate(
        tokenizer=tokenizer_mllm,
        input_ids=input_ids.unsqueeze(0).to('cuda:0'),
        image_embeds=image_embeds,
        ids_cmp_mask=ids_cmp_mask.unsqueeze(0).to('cuda:0', dtype=torch.bool),
        max_new_tokens=500,
        num_img_gen_tokens=agent_model.output_resampler.num_queries,
    )

    img_gen_feat = output['img_gen_feat'].view(pipeline.unet.config.max_num_ips, pipeline.unet.config.num_vision_tokens, -1)
    img_gen_feat = img_gen_feat * mllm_scale + image_embeds.view(pipeline.unet.config.max_num_ips, pipeline.unet.config.num_vision_tokens, -1) * (1 - mllm_scale)

    try:
        images = pipeline(
            prompt=prompt,
            prompt_2=prompt,
            height=height,
            width=width,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            negative_prompt=negative_prompt,
            negative_prompt_2=negative_prompt,
            num_samples=num_samples,
            generator=generator,
            ip_images=[],
            ip_image_embeds=img_gen_feat,
            ip_bbox=ip_bbox,
            ip_scale=ip_scale,
            dialog_bbox=dialog_bbox,
        ).images
    except Exception as e:
        print(f"Generation failed: {e}")
        gc.collect()
        torch.cuda.empty_cache()
        return None

    return images


def load_models(args):
    """Load all required models and pipeline"""
    weight_dtype = torch.float16
     
    # Load config
    config = OmegaConf.load(args.config_path)
    #inference_config = OmegaConf.load(args.inference_config_path)
    
    # Load UNet
    unet = UNetMangaModel.from_config(
        os.path.join(args.ckpt_path, "image_generator"), 
        subfolder="unet", 
        torch_dtype=weight_dtype
    )
    unet.set_manga_modules(
        max_num_ips=config.image_generator.max_num_ips,
        num_vision_tokens=config.image_generator.num_vision_tokens,
        max_num_dialogs=config.image_generator.max_num_dialogs,
    )
    checkpoint = torch.load(os.path.join(args.ckpt_path, "image_generator", "unet", "pytorch_model.bin"))
    unet.load_state_dict(checkpoint)
    
    # Load encoders
    suffix = "_8bit" if args.bit_8 else ""
    clip_image_encoder_dir = f"clip_image_encoder{suffix}"
    print(f"clip_image_encoder_dir: {clip_image_encoder_dir}")
    image_encoder = CLIPVisionModelWithProjection.from_pretrained(
        os.path.join(args.ckpt_path, "image_generator", clip_image_encoder_dir), 
        torch_dtype=weight_dtype
    )
    magi_image_encoder = ViTMAEModel.from_pretrained(
        os.path.join(args.ckpt_path, "image_generator", "magi_image_encoder"), 
        torch_dtype=weight_dtype
    ).to(device="cuda:0")
    
    # Load image projection model
    image_proj_model = Resampler(
        dim=config.resampler.dim,
        depth=config.resampler.depth,
        dim_head=config.resampler.dim_head,
        heads=config.resampler.heads,
        num_queries=config.image_generator.num_vision_tokens,
        num_dummy_tokens=config.image_generator.num_dummy_tokens,
        embedding_dim=image_encoder.config.hidden_size,
        output_dim=unet.config.cross_attention_dim,
        ff_mult=config.resampler.ff_mult,
        magi_embedding_dim=magi_image_encoder.config.hidden_size
    ).to(device='cuda:0', dtype=weight_dtype)
    
    checkpoint = torch.load(
        os.path.join(args.ckpt_path, "image_generator", "image_proj_model", "pytorch_model.bin"), 
        map_location='cpu'
    )
    image_proj_model.load_state_dict(checkpoint)

    # Load MLLM components
    llm_path = f"llm{suffix}"
    print(f"llm_path: {llm_path}")
    tokenizer_mllm = LlamaTokenizer.from_pretrained(os.path.join(args.ckpt_path, "mllm", "tokenizer"))
    llm_model = LlamaForCausalLM.from_pretrained(os.path.join(args.ckpt_path, "mllm", llm_path), torch_dtype=weight_dtype)
    
    input_resampler = QwenResampler(**config.agent.input_resampler)
    output_resampler = QwenResampler(**config.agent.output_resampler)
    agent_model = ContinuousLVLM.from_pretrained(
        llm=llm_model,
        input_resampler=input_resampler,
        output_resampler=output_resampler,
    ).to(device='cuda:0', dtype=weight_dtype)
    
    checkpoint = torch.load(os.path.join(args.ckpt_path, "mllm", "agent", "pytorch_model.bin"))
    agent_model.load_state_dict(checkpoint, strict=False)

    # Create pipeline
    pipeline = DiffSenseiPipeline.from_pretrained(
        os.path.join(args.ckpt_path, "image_generator"),
        unet=unet,
        image_encoder=image_encoder,
        torch_dtype=weight_dtype,
    )
    pipeline.progress_bar_config = {"disable": True}
    pipeline.register_manga_modules(
        image_proj_model=image_proj_model,
        magi_image_encoder=magi_image_encoder,
    )
    pipeline.to(device='cuda:0', dtype=weight_dtype)

    return pipeline, tokenizer_mllm, agent_model

def generate_panels_from_json(json_path, output_dir,pipeline, tokenizer_mllm=None, agent_model=None):
    """Generate panels from JSON specification file and save them"""
    import json
    import os
    from datetime import datetime
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Create timestamped subdirectory for this run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(output_dir, f"generation_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)
    
    # Default generation parameters
    num_samples = 1
    seed = 0
    num_inference_steps = 30
    guidance_scale = 7.5
    negative_prompt = ""
    ip_scale = 0.7
    mllm_scale = 0.3
    
    # Load JSON data
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    generated_images = []
    
    for panel_idx, panel in enumerate(data["panels"]):
        print(f"\nProcessing panel: {panel['guid']}")
        # Get panel dimensions
        panel_width = panel["box"]["width"]
        panel_height = panel["box"]["height"]
        prompt = panel["description"]
        # Process character boxes and images
        ip_images = []
        ip_bbox = []
        
        for char in panel.get("characters", []):
            try:
                # Load character image
                img = Image.open(char["path"]).convert('RGB')
                ip_images.append(img)
                
                # Convert character box to normalized coordinates
                char_x = char["box"]["x"] / panel_width
                char_y = char["box"]["y"] / panel_height
                char_w = char["box"]["width"] / panel_width
                char_h = char["box"]["height"] / panel_height
                
                # Store as [x1, y1, x2, y2] format
                ip_bbox.append([
                    char_x,
                    char_y,
                    char_x + char_w,
                    char_y + char_h
                ])
                
            except Exception as e:
                print(f"Error processing character: {e}")
                continue
        
        # Process dialog boxes
        dialog_bbox = []
        for dialog in panel.get("dialog", []):
            # Convert dialog box to normalized coordinates
            dialog_x = dialog["box"]["x"] / panel_width
            dialog_y = dialog["box"]["y"] / panel_height
            dialog_w = dialog["box"]["width"] / panel_width
            dialog_h = dialog["box"]["height"] / panel_height
            
            # Store as [x1, y1, x2, y2] format
            dialog_bbox.append([
                dialog_x,
                dialog_y,
                dialog_x + dialog_w,
                dialog_y + dialog_h
            ])
        try:
           # Call the generation function
            results = result_generation(
                pipeline=pipeline,
                tokenizer_mllm=tokenizer_mllm,
                agent_model=agent_model,
                prompt=prompt,  
                height=panel_height,
                width=panel_width,
                num_samples=num_samples,
                seed=seed,
                ip_images=ip_images,  # Pass loaded PIL images instead of paths
                ip_bbox=panel.ip_bbox,  # Already normalized
                dialog_bbox=panel.dialog_bbox,  # Already normalized
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                negative_prompt=negative_prompt,
                ip_scale=ip_scale,
                mllm_scale=mllm_scale
            )
                
            if results:
                # Save each generated image
                for img_idx, img in enumerate(results):
                    # Create filename with panel info
                    filename = f"panel_{panel_idx:03d}_{panel['guid']}_{img_idx:02d}.png"
                    filepath = os.path.join(run_dir, filename)
                    
                    # Save image
                    img.save(filepath)
                    print(f"Saved image to: {filepath}")
                
                generated_images.extend(results)
                print(f"Successfully generated image for panel {panel['guid']}")
            
        except Exception as e:
            print(f"Error generating panel {panel['guid']}: {str(e)}")
            continue
        
        # Clean up loaded images
        for img in ip_images:
            img.close()
    
    # Save generation metadata
    metadata = {
        "timestamp": timestamp,
        "num_panels": len(data["panels"]),
        "generation_params": {
            "num_samples": num_samples,
            "seed": seed,
            "num_inference_steps": num_inference_steps,
            "guidance_scale": guidance_scale,
            "ip_scale": ip_scale,
            "mllm_scale": mllm_scale if tokenizer_mllm else None
        },
        "input_json": json_path
    }
    
    with open(os.path.join(run_dir, "metadata.json"), 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\nGeneration complete. Output saved to: {run_dir}")
    return generated_images

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, required=True)
    parser.add_argument("--inference_config_path", type=str, required=True)
    parser.add_argument("--ckpt_path", type=str, required=True)
    parser.add_argument("--bit_8",action="store_true")
    parser.add_argument("--json", type=str, required=True) #path to json
    parser.add_argument("--output_dir", type=str, required=True) #path to json
    args = parser.parse_args()
    
    # Load models and pipeline
    pipeline, tokenizer_mllm, agent_model = load_models(args)
    generate_panels_from_json(args.json, args.output_dir,pipeline,tokenizer_mllm,agent_model)


    
    
    

if __name__ == "__main__":
    main()