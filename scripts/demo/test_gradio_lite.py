import os
os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"
import gradio as gr
import gradio_image_prompter as gr_ext
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
from src.pipelines.pipeline_diffsensei import DiffSenseiPipeline

# Constants
PAGE_WIDTH = 2048
PAGE_HEIGHT = 2896
PANEL_COLOR = "blue"
CHARACTER_COLORS = ["green", "red", "yellow", "purple"]
DIALOG_COLOR = "black"

#

# Initialize processors
clip_image_processor = CLIPImageProcessor()
magi_image_processor = ViTImageProcessor()


def result_generation(
    pipeline,
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
    ip_scale
):
    generator = torch.Generator('cuda:0').manual_seed(seed)
        
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
            ip_images=ip_images,
            ip_image_embeds=None,
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

class PanelData:
    def __init__(self, coords, prompt=""):
        self.coords = coords  # [x1, y1, x2, y2]
        self.ip_images = []   # list of image paths
        self.ip_bbox = []     # normalized coordinates matching ip_images
        self.dialog_bbox = [] # normalized coordinates for dialog boxes
        self.prompt = prompt  # individual prompt for this panel

class Page:
    def __init__(self, num_images=0):
        self.num_images = num_images
        self.panels = []  # list of PanelData objects
    
    def add_panel(self, panel):
        """Add a panel to the page"""
        self.panels.append(panel)
    
    def get_panel_containing_point(self, x, y):
        """Find panel containing the given point"""
        for panel in self.panels:
            px1, py1, px2, py2 = panel.coords
            if (x >= px1 and x <= px2 and y >= py1 and y <= py2):
                return panel
        return None
    
class MangaPageApp:
    def __init__(self):
        self.current_page = None
        self.demo = None
        self.char_map = {}  # Add char_map as class member
        self.reverse_char_map = {}
        self.prompt=None
        
    def init_page(self, ip_images=None, prompt=None):
        """Initialize or reset the page and character map"""
        self.current_page = Page(len(ip_images))
        self.prompt = prompt
        self.char_map = {i: path for i, path in enumerate(ip_images)} if ip_images else {}
        self.reverse_char_map = {path: i for i, path in self.char_map.items()}  # Create reverse mapping
    #canvas init
    def create_blank_image_dict(self, width=PAGE_WIDTH, height=PAGE_HEIGHT, color='white'):
        """Creates a blank image dictionary for Gradio"""
        image = Image.new('RGB', (width, height), color=color)
        return {"image": image, "points": []}
    
  
    
    def normalize_box_to_panel(self, box_coords, panel_coords):
        """Normalize box coordinates relative to panel dimensions"""
        bx1, by1, bx2, by2 = box_coords
        px1, py1, px2, py2 = panel_coords
        
        panel_width = px2 - px1
        panel_height = py2 - py1
        
        # First translate to panel origin
        rel_x1 = bx1 - px1
        rel_y1 = by1 - py1
        rel_x2 = bx2 - px1
        rel_y2 = by2 - py1
        
        # Then normalize by panel dimensions
        norm_x1 = rel_x1 / panel_width
        norm_y1 = rel_y1 / panel_height
        norm_x2 = rel_x2 / panel_width
        norm_y2 = rel_y2 / panel_height
        
        return [norm_x1, norm_y1, norm_x2, norm_y2]
    
    def denormalize_box_from_panel(self, norm_coords, panel_coords):
        """Convert normalized coordinates back to page coordinates"""
        nx1, ny1, nx2, ny2 = norm_coords
        px1, py1, px2, py2 = panel_coords
        
        panel_width = px2 - px1
        panel_height = py2 - py1
        
        # Denormalize to panel-relative coordinates
        rel_x1 = nx1 * panel_width
        rel_y1 = ny1 * panel_height
        rel_x2 = nx2 * panel_width
        rel_y2 = ny2 * panel_height
        
        # Translate back to page coordinates
        x1 = rel_x1 + px1
        y1 = rel_y1 + py1
        x2 = rel_x2 + px1
        y2 = rel_y2 + py1
        
        return [x1, y1, x2, y2]
    
    def process_box_points(self, points_data):
        """Convert raw points data into list of (x1,y1,x2,y2) boxes"""
        if not points_data:
            return []
        
        try:
            points = np.array(points_data).reshape((-1, 2, 3))
            boxes = []
            for start, end in points:
                x1, y1 = start[0], start[1]
                x2, y2 = end[0], end[1]
                # Ensure coordinates are in correct order
                x1, x2 = min(x1, x2), max(x1, x2)
                y1, y2 = min(y1, y2), max(y1, y2)
                boxes.append((x1, y1, x2, y2))
            return boxes
        except (ValueError, IndexError) as e:
            print(f"Error processing box points: {e}")
            return []
    
    def point_in_panel(self, point_coords, panel_coords):
        """Check if a point is inside a panel"""
        x, y = point_coords[0], point_coords[1]
        px1, py1, px2, py2 = panel_coords
        return (x >= px1 and x <= px2 and y >= py1 and y <= py2)
    
    def is_box_within_panel(self, box_coords, panel_coords):
        """Check if a box is completely within a panel"""
        bx1, by1, bx2, by2 = box_coords
        return (self.point_in_panel((bx1, by1), panel_coords) and 
                self.point_in_panel((bx2, by2), panel_coords))
    
    def process_panel_boxes(self, panel_canvas, char_canvas, dialog_canvas, prompt=None, ip_images=None, current_char_idx=None):
        """Process all canvases into a Page object"""
        try:
            if self.current_page is None:
                print("Error: page not initialized")
                return
            else:
                page = self.current_page
            debug_messages = []
            
            # Split prompt into individual panel prompts
            panel_prompts = []
            print(f"Received prompt type: {type(prompt)}, value: {prompt}")  # Debug print
            if prompt and isinstance(prompt, str):
                panel_prompts = [p.strip() for p in prompt.split('\n') if p.strip()]
                print(f"Processed prompts: {panel_prompts}")
            
            # Process panel boundaries
            if panel_canvas is not None:
                # Get all panel boxes
                panel_boxes = self.process_box_points(panel_canvas.get("points", []))
                
                # Sort panels by manga reading order (right to left, top to bottom)
                # First, group panels by rows based on vertical position
                ROW_THRESHOLD = 50  # Pixels threshold to consider panels in the same row
                rows = {}
                for box in panel_boxes:
                    x1, y1, x2, y2 = box
                    center_y = (y1 + y2) / 2
                    
                    # Find or create row
                    row_found = False
                    for row_y in rows.keys():
                        if abs(center_y - row_y) < ROW_THRESHOLD:
                            rows[row_y].append(box)
                            row_found = True
                            break
                    
                    if not row_found:
                        rows[center_y] = [box]
                
                # Sort rows by y-coordinate (top to bottom)
                sorted_rows = sorted(rows.items(), key=lambda x: x[0])
                
                # Within each row, sort panels right to left
                ordered_panels = []
                for row_y, row_panels in sorted_rows:
                    # Sort panels in this row from right to left based on x-coordinate
                    row_panels_sorted = sorted(row_panels, key=lambda box: -(box[0] + box[2])/2)  # Negative for right-to-left
                    ordered_panels.extend(row_panels_sorted)
                
                # Now create panels in the correct order
                for i, (x1, y1, x2, y2) in enumerate(ordered_panels):
                    # Get corresponding prompt or use default
                    panel_prompt = panel_prompts[i] if i < len(panel_prompts) else ""
                    panel = PanelData([x1, y1, x2, y2], prompt=panel_prompt)
                    page.add_panel(panel)
                    print(f"Added panel {i} with prompt: {panel_prompt}")
                    print(f"Panel coordinates: ({x1:.1f}, {y1:.1f}) -> ({x2:.1f}, {y2:.1f})")
            
            # Process character boxes
            if char_canvas and char_canvas.get("points"):
                try:
                    points = np.array(char_canvas.get("points")).reshape((-1, 2, 3))
                    
                    for start, end in points:
                        x1, y1 = start[0], start[1]
                        x2, y2 = end[0], end[1]
                        box_coords = (x1, y1, x2, y2)
                        
                        for panel in page.panels:
                            if self.point_in_panel((x1, y1), panel.coords):
                                if not self.is_box_within_panel(box_coords, panel.coords):
                                    debug_messages.append(f"Warning: Character box [{x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f}] extends beyond panel boundaries")
                                    continue
                                    
                                normalized_box = self.normalize_box_to_panel(box_coords, panel.coords)
                                
                                # Check if this box is new
                                is_new = True
                                for existing_box in panel.ip_bbox:
                                    if all(abs(a - b) < 0.001 for a, b in zip(normalized_box, existing_box)):
                                        is_new = False
                                        break
                                
                                if is_new and current_char_idx is not None:
                                    if current_char_idx in self.char_map:
                                        panel.ip_images.append(self.char_map[current_char_idx])
                                        panel.ip_bbox.append(normalized_box)
                                    else:
                                        debug_messages.append(f"Warning: Invalid character index {current_char_idx}")
                                break
                except (ValueError, IndexError) as e:
                    debug_messages.append(f"Error processing character boxes: {e}")
            
            # Process dialog boxes
            if dialog_canvas and dialog_canvas.get("points"):
                try:
                    dialog_boxes = self.process_box_points(dialog_canvas.get("points", []))
                    
                    for x1, y1, x2, y2 in dialog_boxes:
                        box_coords = (x1, y1, x2, y2)
                        
                        for panel in page.panels:
                            if self.point_in_panel((x1, y1), panel.coords):
                                if not self.is_box_within_panel(box_coords, panel.coords):
                                    debug_messages.append(f"Warning: Dialog box [{x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f}] extends beyond panel boundaries")
                                    continue
                                    
                                normalized_box = self.normalize_box_to_panel(box_coords, panel.coords)
                                
                                # Check if this box is new
                                is_new = True
                                for existing_box in panel.dialog_bbox:
                                    if all(abs(a - b) < 0.001 for a, b in zip(normalized_box, existing_box)):
                                        is_new = False
                                        break
                                
                                if is_new:
                                    panel.dialog_bbox.append(normalized_box)
                                break
                except (ValueError, IndexError) as e:
                    debug_messages.append(f"Error processing dialog boxes: {e}")
            
            self.current_page = page
            return page, debug_messages
            
        except Exception as e:
            print(f"Unexpected error in process_panel_boxes: {e}")
            return Page(0), [f"Error: {str(e)}"]
    
    def draw_box_with_character(self, canvas_img):
        """Draws a visual representation of the current Page state"""
        new_image = Image.new('RGB', (PAGE_WIDTH, PAGE_HEIGHT), 'white')
        draw = ImageDraw.Draw(new_image)
        page = self.current_page
        # First draw panel boxes (blue)
        for panel in page.panels:
            x1, y1, x2, y2 = panel.coords
            draw.rectangle([(x1, y1), (x2, y2)], outline=PANEL_COLOR, width=3)
            
            # Draw character boxes for this panel
            for char_path, bbox in zip(panel.ip_images, panel.ip_bbox):
                page_coords = self.denormalize_box_from_panel(bbox, panel.coords)
                cx1, cy1, cx2, cy2 = page_coords
                
                # Get character index from path
                char_idx = self.reverse_char_map.get(char_path)  # Simple dictionary lookup
                color = CHARACTER_COLORS[char_idx % len(CHARACTER_COLORS)]
                draw.rectangle([(cx1, cy1), (cx2, cy2)], outline=color, width=3)
                draw.text((cx1+5, cy1+5), f"Char {char_idx}", fill=color)
            
            # Draw dialog boxes
            for bbox in panel.dialog_bbox:
                page_coords = self.denormalize_box_from_panel(bbox, panel.coords)
                dx1, dy1, dx2, dy2 = page_coords
                
                draw.ellipse([(dx1, dy1), (dx2, dy2)], fill='white', outline=DIALOG_COLOR, width=2)
        return {"image": new_image, "points": canvas_img.get("points", [])}
    
  
    def create_character_selector(self, ip_images):
        """Creates radio buttons for character selection"""
        if not ip_images:
            return gr.Radio(choices=[], value=None, visible=False)
        num_images = len(ip_images)
        return gr.Radio(
            choices=[f"Character {i+1}" for i in range(num_images)],
            value=f"Character 1",
            visible=True,
            label="Select character before drawing box"
        )    

    
    def copy_canvas_with_boxes(self, source_img, color="blue", store_points=False):
        """Copies the source canvas and draws all boxes in specified color"""
        if source_img is None:
            return self.create_blank_image_dict()
        
        new_image = Image.new('RGB', (PAGE_WIDTH, PAGE_HEIGHT), 'white')
        draw = ImageDraw.Draw(new_image)
        
        if source_img.get("points"):
            points = np.array(source_img["points"]).reshape((-1, 2, 3))
            for start, end in points:
                x1, y1 = start[0], start[1]
                x2, y2 = end[0], end[1]
                draw.rectangle([(x1, y1), (x2, y2)], outline=color, width=3)
        
        return {"image": new_image, "points": source_img.get("points", []) if store_points else []}
    
    def process_new_dialog_box(self, dialog, panels, chars, imgs):
        """Process new dialog box and update page state"""
        if dialog and dialog.get("points"):
            dialog_boxes = self.process_box_points(dialog.get("points", []))
            for x1, y1, x2, y2 in dialog_boxes:
                box_coords = (x1, y1, x2, y2)
                for panel in self.current_page.panels:
                    if self.point_in_panel((x1, y1), panel.coords):
                        normalized_box = self.normalize_box_to_panel(box_coords, panel.coords)
                        panel.dialog_bbox.append(normalized_box)
   
    def get_char_idx_from_selector(self, selector_value):
        """Get character index from selector string"""
        if not selector_value:
            return None
        try:
            return int(selector_value.split()[-1]) - 1
        except (ValueError, IndexError):
            return None
    
    def debug_state(self):
        """Print debug info to console"""
        try:
            print("\n" + "="*50)
            print("DEBUG STATE")
            print("="*50)
            
            if self.current_page is None:
                print("No page initialized")
                return
            
            print(f"Number of character images: {self.current_page.num_images}")
            #print(f"Current character: {current_char_idx}")
            print(f"Total panels: {len(self.current_page.panels)}")
            
            for panel_idx, panel in enumerate(self.current_page.panels):
                print(f"\nPANEL {panel_idx}")
                print("-"*20)
                print(f"Coordinates: ({panel.coords[0]:.1f}, {panel.coords[1]:.1f}) -> ({panel.coords[2]:.1f}, {panel.coords[3]:.1f})")
                print(f"\nPrompt: {panel.prompt}")
                print("\nCharacter boxes:")
                if not panel.ip_images:
                    print("  None")
                for path, bbox in zip(panel.ip_images, panel.ip_bbox):
                    print(f"  Path: {path}")
                    print(f"  Box: [{bbox[0]:.3f}, {bbox[1]:.3f}, {bbox[2]:.3f}, {bbox[3]:.3f}]")
                
                print("\nDialog boxes:")
                if not panel.dialog_bbox:
                    print("  None")
                for bbox in panel.dialog_bbox:
                    print(f"  [{bbox[0]:.3f}, {bbox[1]:.3f}, {bbox[2]:.3f}, {bbox[3]:.3f}]")
            
            print("\n" + "="*50 + "\n")
            
        except Exception as e:
            print(f"\nERROR in debug_state: {str(e)}\n")
    
    def create_ui(self, pipeline=None):
        """Create and configure the Gradio interface"""
        with gr.Blocks(title="Manga Page Layout Test") as self.demo:
            with gr.Row():
                with gr.Column(scale=2):
                    with gr.Column():
                        gr.Markdown("""
                        ## Manga Page Layout Test Interface
                        1. Upload character images (max 4)
                        2. Draw panels
                        3. Select character, then draw their boxes
                        4. Add dialog boxes
                        5. Generate page
                        """)
                        
                        prompt = gr.Textbox(label="Prompt", lines=1, value="Test prompt")
                        ip_images = gr.File(
                            label="Upload Character Images (max 4)", 
                            file_count="multiple", 
                            type="filepath"
                        )
                        
                        # Character selector
                        char_selector = gr.Radio(
                            choices=[], 
                            label="Select character before drawing box",
                            visible=False
                        )
                        
                        # Panel layout canvas
                        panel_canvas = gr_ext.ImagePrompter(
                            label="1. Draw Panels (Blue)",
                            value=self.create_blank_image_dict(),
                            width=PAGE_WIDTH//3,
                            height=PAGE_HEIGHT//3
                        )
                        finish_panels_btn = gr.Button("Finish Drawing Panels")
                        
                        # Character placement canvas
                        char_canvas = gr_ext.ImagePrompter(
                            label="2. Place Characters (Select character first!)",
                            value=self.create_blank_image_dict(),
                            width=PAGE_WIDTH//3,
                            height=PAGE_HEIGHT//3
                        )
                        draw_char_box_btn = gr.Button("Draw Character Box")
                        finish_chars_btn = gr.Button("Finish Placing Characters")
                        
                        # Dialog placement canvas
                        dialog_canvas = gr_ext.ImagePrompter(
                            label="3. Place Dialogs",
                            value=self.create_blank_image_dict(),
                            width=PAGE_WIDTH//3,
                            height=PAGE_HEIGHT//3
                        )
                        draw_dialog_box_btn = gr.Button("Draw Dialog Box")
                        
                        generate_btn = gr.Button("Generate Page")
                        output_image = gr.Image(label="Generated Page", width=PAGE_WIDTH//3, height=PAGE_HEIGHT//3)
                
                with gr.Column(scale=1):
                    pass

                        
             # Event handlers with debug prints
            ip_images.change(
                fn=lambda imgs, prompt: (
                    #self.create_character_selector(imgs),
                    self.init_page(imgs if imgs else None, prompt),
                    self.debug_state(),
                    self.create_character_selector(imgs)  # Return value for UI
                )[-1],
                inputs=[ip_images,prompt],
                outputs=[char_selector]
            )

            
            finish_panels_btn.click(
                fn=lambda canvas, imgs: (
                    self.process_panel_boxes(canvas, None, None, imgs),
                    self.debug_state(),
                    self.copy_canvas_with_boxes(canvas, store_points=False)
                )[-1],
                inputs=[panel_canvas, ip_images],
                outputs=[char_canvas]
            )
            
            draw_char_box_btn.click(
                fn=lambda canvas, char_sel, panels, imgs: (
                    self.process_panel_boxes(None, canvas, None, imgs, self.get_char_idx_from_selector(char_sel)),
                    self.debug_state(),
                    self.draw_box_with_character(canvas)
                )[-1],
                inputs=[char_canvas, char_selector, panel_canvas, ip_images],
                outputs=[char_canvas]
            )
            
            finish_chars_btn.click(
                fn=lambda panels, chars: (
                   
                    self.debug_state(),
                    self.copy_canvas_with_boxes({
                        "image": panels["image"],
                        "points": panels["points"] + chars["points"]  # Combine panel and character boxes
                    }, store_points=False)
                )[-1],
                inputs=[panel_canvas, char_canvas],
                outputs=[dialog_canvas]
            )
            
            draw_dialog_box_btn.click(
                fn=lambda dialog, panels, chars, imgs: (
                    self.process_panel_boxes(None, None, dialog),
                    self.debug_state(),
                    self.draw_box_with_character(dialog)
                )[-1],
                inputs=[dialog_canvas, panel_canvas, char_canvas, ip_images],
                outputs=[dialog_canvas]
            )
                        
            # Add output gallery
            with gr.Column():
                output_gallery = gr.Gallery(
                    label="Generated Panels",
                    show_label=True,
                    elem_id="output_gallery",
                    columns=1,
                    rows=None,
                    height=None
                )
            
            # Connect generate button
            generate_btn.click(
                fn=lambda prompt: self.generate_panels(
                    prompt=prompt,
                    pipeline=pipeline
                ),
                inputs=[prompt],
                outputs=[output_gallery]
            )
    
    def launch(self, pipeline=None):
        """Launch the Gradio interface"""
        if self.demo is None:
            self.create_ui(pipeline)
        self.demo.launch(share=True)
    
    def generate_panels(self, prompt, pipeline):
        """Generate images for all panels in the current page"""
        if not self.current_page or not self.current_page.panels:
            print("No panels to generate!")
            return None
        
        # Create a list to store generated images
        generated_images = []
        
        # Default generation parameters
        num_samples = 1
        seed = 0
        num_inference_steps = 30
        guidance_scale = 7.5
        negative_prompt = ""
        ip_scale = 0.7

        for i, panel in enumerate(self.current_page.panels):
            print(f"\nGenerating panel {i+1}")
            print(f"Original panel coordinates: {panel.coords}")
            print(f"Using panel prompt: {panel.prompt}")
            
            # Extract panel dimensions
            x1, y1, x2, y2 = panel.coords
            width = int(x2 - x1)
            height = int(y2 - y1)
            
            # Process character boxes and images for this panel
            ip_images = panel.ip_images
            ip_bbox = panel.ip_bbox  # Already normalized
            dialog_bbox = panel.dialog_bbox  # Already normalized
            
            try:
                # Call the generation function
                results = result_generation(
                    pipeline=pipeline,
                    prompt=panel.prompt or prompt,  # Use panel prompt if available, else use global prompt
                    height=height,
                    width=width,
                    num_samples=num_samples,
                    seed=seed,
                    ip_images=ip_images,
                    ip_bbox=ip_bbox,
                    dialog_bbox=dialog_bbox,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    negative_prompt=negative_prompt,
                    ip_scale=ip_scale
                )
                
                if results:
                    generated_images.extend(results)
                    print(f"Successfully generated image for panel {i+1}")
                else:
                    print(f"No results generated for panel {i+1}")
                
            except Exception as e:
                print(f"Error generating panel {i+1}: {str(e)}")
                continue
        
        return generated_images

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
    image_encoder = CLIPVisionModelWithProjection.from_pretrained(
        os.path.join(args.ckpt_path, "image_generator", "clip_image_encoder"), 
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

    return pipeline
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, required=True)
    parser.add_argument("--inference_config_path", type=str, required=True)
    parser.add_argument("--ckpt_path", type=str, required=True)
    args = parser.parse_args()
    
    # Load models and pipeline
    pipeline = load_models(args)
    
    # Create and launch app
    app = MangaPageApp()
    app.launch(pipeline)

if __name__ == "__main__":
    main()