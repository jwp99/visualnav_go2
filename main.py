import io
import os
import sys
import yaml
import torch
import numpy as np
from collections import deque
from contextlib import asynccontextmanager
from fastapi import FastAPI, File, UploadFile, HTTPException
from PIL import Image
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


MAX_V = 0.6
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

# --- Core Waypoint Generation Logic ---
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
        for i, img_tensor in enumerate(obs_images_list):
            pil_img = to_pil_image(img_tensor.squeeze(0).cpu())
            pil_img.save(os.path.join(testdata_dir, f"transformed_obs_{i}.png"))

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
    
    return waypoints * MAX_V / FRAME_RATE

# --- API Endpoints ---

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
async def generate_waypoints_endpoint():
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
        return {"waypoints": waypoints.tolist()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate waypoints: {e}")

@app.get("/")
def read_root():
    return {"message": "Welcome to the Visual Navigation Waypoint Generation API"} 

# uvicorn main:app --host 0.0.0.0 --port 8001