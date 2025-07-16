import io
import os
import sys
import yaml
import torch
import base64
import numpy as np
from collections import deque
from datetime import datetime
from typing import Optional
from contextlib import asynccontextmanager
from fastapi import FastAPI, File, UploadFile, HTTPException
from PIL import Image, ImageDraw
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from torchvision.transforms.functional import to_pil_image

# Add project directories to Python path
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'train')))

from deployment.src.utils import load_model, transform_images, to_numpy
from vint_train.training.train_utils import get_action

# --- Constants ---
CKPT_PATH = "/home/jeong/visualnav-transformer/nomad.pth"
CONFIG_PATH = "train/config/nomad.yaml"

# --- FastAPI App Setup ---


MAX_V = 0.4
FRAME_RATE = 4

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load resources on startup
    print("Loading model and resources...")
    app.state.device = torch.device("cpu")
    
    # Load config
    if not os.path.exists(CONFIG_PATH):
        raise FileNotFoundError(f"Config file not found at {CONFIG_PATH}. Please update the path.")
    with open(CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)
    app.state.cfg = cfg

    # Load model
    app.state.model = load_model(CKPT_PATH, app.state.cfg, app.state.device)
    app.state.model.eval()

    # Initialize noise scheduler
    app.state.noise_sched = DDPMScheduler(
        num_train_timesteps=app.state.cfg["num_diffusion_iters"],
        beta_schedule="squaredcos_cap_v2",
        clip_sample=True,
        prediction_type="epsilon",
    )
    
    # Set maxlen for observation queue based on config
    app.state.obs_queue = deque(maxlen=app.state.cfg['context_size'] + 1)
    
    print("Model and resources loaded successfully.")
    yield
    # Clean up resources on shutdown if needed
    print("Shutting down...")

app = FastAPI(lifespan=lifespan)

# --- In-memory Storage ---
app.state.goal_image = None
app.state.goal_image_tensor = None
app.state.latest_waypoints = None
app.state.topomap = None
app.state.closest_node = 0

# --- Core Waypoint Generation Logic ---

def draw_waypoints_on_image(image: Image.Image, trajectories: np.ndarray):
    """Draws multiple local waypoint paths on the camera frame with faked perspective."""
    if trajectories.size == 0:
        return image

    draw = ImageDraw.Draw(image)
    
    img_width, img_height = image.size

    vanishing_point = (img_width / 2, img_height * 0.55)
    horizontal_stretch = 120
    vertical_stretch = 100

    path_origin = (img_width / 2, img_height)
    
    for waypoints in trajectories:
        path = [path_origin]

        for point in waypoints:
            robot_x, robot_y = point[0], point[1]
            if robot_x <= 0.1:
                continue

            perspective_scale = 1 / robot_x
            pixel_x = vanishing_point[0] - (robot_y * horizontal_stretch * perspective_scale)
            pixel_y = vanishing_point[1] + (vertical_stretch * perspective_scale)
            
            if pixel_y > img_height:
                continue
            path.append((pixel_x, pixel_y))

        if len(path) > 1:
            draw.line(path, fill="green", width=3)
    
    return image

def generate_waypoints(model, noise_sched, obs_queue, goal_img_tensor, cfg, device, visualize=True):
    """
    Generates waypoint predictions given observations and a goal image.
    """
    num_samples = 8
    horizon = cfg['len_traj_pred']
    
    # Prepare observation tensors
    obs_images_tensors = transform_images(list(obs_queue), cfg['image_size'], center_crop=False)
    obs_images_list = torch.split(obs_images_tensors, 3, dim=1)

    if visualize:
        testdata_dir = "testdata/obs"
        if not os.path.exists(testdata_dir):
            os.makedirs(testdata_dir)
        # Save original observation images
        for i, pil_img in enumerate(list(obs_queue)):
            pil_img.save(os.path.join(testdata_dir, f"original_obs_{i}.png"))

    obs_images_tensors = torch.cat(obs_images_list, dim=1).to(device)

    # Get conditioning vector
    mask = torch.zeros(1).long().to(device)
    with torch.no_grad():
        obs_cond = model('vision_encoder', obs_img=obs_images_tensors, goal_img=goal_img_tensor, input_goal_mask=mask)

    # Repeat for batch sampling
    if len(obs_cond.shape) == 2:
        obs_cond = obs_cond.repeat(num_samples, 1)
    else:
        obs_cond = obs_cond.repeat(num_samples, 1, 1)

    # Generate trajectories
    with torch.no_grad():
        noisy_action = torch.randn((num_samples, horizon, 2), device=device)
        naction = noisy_action
        noise_sched.set_timesteps(cfg['num_diffusion_iters'])
        for k in noise_sched.timesteps[:]:
            noise_pred = model('noise_pred_net', sample=naction, timestep=k, global_cond=obs_cond)
            naction = noise_sched.step(model_output=noise_pred, timestep=k, sample=naction).prev_sample
        
        waypoints = to_numpy(get_action(naction))
    
    scaled_waypoints = waypoints * MAX_V / FRAME_RATE

    if visualize and len(obs_queue) > 0:
        # Get the last observation image
        last_obs_img = list(obs_queue)[-1].copy()
        
        # Draw all trajectories on the image
        img_with_waypoints = draw_waypoints_on_image(last_obs_img, scaled_waypoints)
        
        # Save the image
        testdata_dir = "testdata/obs" # Ensure directory exists
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        img_with_waypoints.save(os.path.join(testdata_dir, f"last_obs_with_waypoints_{timestamp}.png"))

    return scaled_waypoints

def generate_waypoints_from_map(
    model, noise_sched, obs_queue, topomap, cfg, device, 
    closest_node, radius, close_threshold, num_samples,
    visualize=True
):
    """
    Generates waypoint predictions using a topological map.
    """
    horizon = cfg['len_traj_pred']
    goal_node = len(topomap) - 1

    # Prepare observation tensors
    obs_images_tensors = transform_images(list(obs_queue), cfg['image_size'], center_crop=False)
    obs_images_list = torch.split(obs_images_tensors, 3, dim=1)
    obs_images_tensors = torch.cat(obs_images_list, dim=1).to(device)

    # Get conditioning vector from observation and goal
    mask = torch.zeros(1).long().to(device)
    with torch.no_grad():
        start = max(closest_node - radius, 0)
        end = min(closest_node + radius + 1, goal_node)
        
        goal_images = [
            transform_images(g_img, cfg["image_size"], center_crop=False).to(device) 
            for g_img in topomap[start:end + 1]
        ]
        goal_images_tensor = torch.cat(goal_images, dim=0)

        obs_img_batch = obs_images_tensors.repeat(len(goal_images), 1, 1, 1)
        mask_batch = mask.repeat(len(goal_images))

        obsgoal_cond = model('vision_encoder', obs_img=obs_img_batch, goal_img=goal_images_tensor, input_goal_mask=mask_batch)
        
        dists = model("dist_pred_net", obsgoal_cond=obsgoal_cond)
        dists = to_numpy(dists.flatten())
        
        min_dist_idx = np.argmin(dists)
        new_closest_node = min_dist_idx + start
        
        # Choose subgoal and get obs_cond
        if dists[min_dist_idx] > close_threshold:
            sg_idx = min_dist_idx
        else:
            sg_idx = min(min_dist_idx + 1, len(obsgoal_cond) - 1)
        
        obs_cond = obsgoal_cond[sg_idx].unsqueeze(0)

    # Repeat for batch sampling
    if len(obs_cond.shape) == 2:
        obs_cond = obs_cond.repeat(num_samples, 1)
    else:
        obs_cond = obs_cond.repeat(num_samples, 1, 1)

    # Generate trajectories
    with torch.no_grad():
        noisy_action = torch.randn((num_samples, horizon, 2), device=device)
        naction = noisy_action
        noise_sched.set_timesteps(cfg['num_diffusion_iters'])
        for k in noise_sched.timesteps[:]:
            noise_pred = model('noise_pred_net', sample=naction, timestep=k, global_cond=obs_cond)
            naction = noise_sched.step(model_output=noise_pred, timestep=k, sample=naction).prev_sample
        
        waypoints = to_numpy(get_action(naction))
    
    scaled_waypoints = waypoints * MAX_V / FRAME_RATE

    if visualize and len(obs_queue) > 0:
        last_obs_img = list(obs_queue)[-1].copy()
        img_with_waypoints = draw_waypoints_on_image(last_obs_img, scaled_waypoints)
        
        testdata_dir = "testdata/map_obs"
        if not os.path.exists(testdata_dir):
            os.makedirs(testdata_dir)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        img_with_waypoints.save(os.path.join(testdata_dir, f"map_obs_with_waypoints_{timestamp}.png"))

    return scaled_waypoints, int(new_closest_node)

# --- API Endpoints ---

def pil_to_base64(pil_image: Image.Image) -> str:
    """Converts a PIL image to a base64 encoded string."""
    buffered = io.BytesIO()
    pil_image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return img_str

@app.post("/set_goal_image/")
async def set_goal_image(
    goal_image: UploadFile = File(..., description="The goal image.")
):
    try:
        contents = await goal_image.read()
        pil_image = Image.open(io.BytesIO(contents))
        app.state.goal_image = pil_image
        app.state.goal_image_tensor = transform_images(
            [pil_image], app.state.cfg['image_size'], center_crop=False
        ).to(app.state.device)
        return {"message": "Goal image set successfully."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to set goal image: {e}")

@app.post("/add_observation_image/")
async def add_observation_image(
    observation_image: UploadFile = File(..., description="A single observation image.")
):
    if app.state.goal_image is None:
        raise HTTPException(status_code=400, detail="Goal image not set.")
    try:
        contents = await observation_image.read()
        pil_image = Image.open(io.BytesIO(contents))
        app.state.obs_queue.append(pil_image)
        return {"message": f"Observation added. Queue size: {len(app.state.obs_queue)}."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to add observation: {e}")

@app.post("/clear_observations/")
async def clear_observations():
    app.state.obs_queue.clear()
    return {"message": "Observation queue cleared."}

@app.post("/generate_waypoints/")
async def generate_waypoints_endpoint(use_map: Optional[bool] = False, 
                                    topomap_dir: Optional[str] = "testdata/recording_20250716_174201",
                                    radius: Optional[int] = 8,
                                    close_threshold: Optional[int] = 3,
                                    num_samples: Optional[int] = 8):
    if not use_map:
        if app.state.goal_image_tensor is None:
            raise HTTPException(status_code=400, detail="Goal image not set.")
        if len(app.state.obs_queue) == 0:
            raise HTTPException(status_code=400, detail="Observation queue is empty.")
        
        try:
            waypoints = generate_waypoints(
                model=app.state.model,
                noise_sched=app.state.noise_sched,
                obs_queue=app.state.obs_queue.copy(),
                goal_img_tensor=app.state.goal_image_tensor,
                cfg=app.state.cfg,
                device=app.state.device
            )
            app.state.latest_waypoints = waypoints
            return {"waypoints": waypoints.tolist()}
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to generate waypoints: {e}")
    else:
        if len(app.state.obs_queue) == 0:
            raise HTTPException(status_code=400, detail="Observation queue is empty.")
        
        # Load topomap if not already loaded
        if app.state.topomap is None:
            if topomap_dir is None or not os.path.exists(topomap_dir) or not os.path.isdir(topomap_dir):
                raise HTTPException(status_code=400, detail=f"Topomap directory not found at {topomap_dir}")
            
            try:
                image_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.gif')
                image_files = [
                    f for f in os.listdir(topomap_dir) 
                    if f.lower().endswith(image_extensions)
                ]
                
                # Sort numerically based on filename, assuming format like '1.png', '2.png', etc.
                topomap_filenames = sorted(image_files, key=lambda x: int(x.split(".")[0]))
                
                app.state.topomap = [Image.open(os.path.join(topomap_dir, fn)) for fn in topomap_filenames]
                app.state.closest_node = 0
            except (ValueError, IndexError):
                raise HTTPException(status_code=400, detail=f"Filenames in directory {topomap_dir} are not in the expected numeric format (e.g., '1.png').")
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Failed to load topomap: {e}")

        try:
            waypoints, new_closest_node = generate_waypoints_from_map(
                model=app.state.model,
                noise_sched=app.state.noise_sched,
                obs_queue=app.state.obs_queue.copy(),
                topomap=app.state.topomap,
                cfg=app.state.cfg,
                device=app.state.device,
                closest_node=app.state.closest_node,
                radius=radius,
                close_threshold=close_threshold,
                num_samples=num_samples
            )
            app.state.latest_waypoints = waypoints
            app.state.closest_node = new_closest_node
            print(f"Closest node: {new_closest_node}")
            return {"waypoints": waypoints.tolist(), "closest_node": new_closest_node}
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to generate waypoints from map: {e}")

@app.post("/set_topomap/")
async def set_topomap(topomap_dir: str):
    """
    Creates and sets a topological map from a directory of images.
    Images are sorted alphanumerically.
    """
    if not os.path.exists(topomap_dir) or not os.path.isdir(topomap_dir):
        raise HTTPException(status_code=404, detail=f"Topomap directory not found: {topomap_dir}")
    
    try:
        image_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.gif')
        image_files = sorted([
            f for f in os.listdir(topomap_dir) 
            if f.lower().endswith(image_extensions)
        ])

        if not image_files:
            raise HTTPException(status_code=400, detail=f"No compatible images found in {topomap_dir}")

        app.state.topomap = [Image.open(os.path.join(topomap_dir, fn)) for fn in image_files]
        app.state.closest_node = 0
        app.state.latest_waypoints = None

        return {"message": f"Topomap set successfully with {len(app.state.topomap)} images from {topomap_dir}."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create topomap: {str(e)}")

@app.get("/get_latest_info/")
async def get_latest_info():
    """
    Returns the latest goal image, observation queue, and generated waypoints.
    Images are returned as base64 encoded strings.
    """
    return {
        "goal_image": pil_to_base64(app.state.goal_image) if app.state.goal_image else None,
        "observation_queue": [pil_to_base64(img) for img in app.state.obs_queue],
        "latest_waypoints": app.state.latest_waypoints.tolist() if app.state.latest_waypoints is not None else None,
        "closest_node": app.state.closest_node
    }

@app.get("/")
def read_root():
    return {"message": "Welcome to the Visual Navigation Waypoint Generation API"} 

# uvicorn main:app --host 0.0.0.0 --port 8001