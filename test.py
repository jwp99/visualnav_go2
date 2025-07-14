import os
import yaml
import torch
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
from PIL import Image
from matplotlib.widgets import Slider
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
import pickle

# Make sure these imports are resolvable in your environment
# You might need to adjust sys.path if running from a different directory
from deployment.src.utils import load_model, transform_images, to_numpy
from vint_train.training.train_utils import get_action

# --- Data Loading ---

def load_traj_data(traj_dir: str):
    """Load traj_data.pkl and return position (T x 2) and yaw (T,) arrays."""
    with open(os.path.join(traj_dir, "traj_data.pkl"), "rb") as f:
        data = pickle.load(f)
    positions = np.asarray(data["position"], dtype=float)
    yaw = np.asarray(data["yaw"], dtype=float)
    return positions, yaw

def load_image(traj_dir: str, idx: int):
    """Load idx.jpg from traj_dir."""
    img_path = os.path.join(traj_dir, f"{idx}.jpg")
    if not os.path.exists(img_path):
        raise FileNotFoundError(f"Image {img_path} not found")
    return Image.open(img_path)

# --- Core Model and Plotting Functions ---

def generate_waypoints(model, noise_sched, obs_queue, goal_img_tensor, cfg, device):
    """
    Generates waypoint predictions given observations and a goal image.

    Args:
        model: The trained NoMaD model.
        noise_sched: The DDPMScheduler.
        obs_queue: A deque containing the sequence of observation images (PIL.Image).
        goal_img_tensor: The pre-processed goal image tensor.
        cfg: The model configuration dictionary.
        device: The torch device.

    Returns:
        A numpy array of shape (num_samples, horizon, 2) representing predicted waypoints.
    """
    num_samples = 8
    horizon = cfg['len_traj_pred']
    
    # 1. Prepare observation tensors
    obs_images_tensors = transform_images(list(obs_queue), cfg['image_size'], center_crop=False)
    obs_images_tensors = torch.split(obs_images_tensors, 3, dim=1)
    obs_images_tensors = torch.cat(obs_images_tensors, dim=1).to(device)

    # 2. Get the conditioning vector from the vision encoder
    mask = torch.zeros(1).long().to(device)
    with torch.no_grad():
        obs_cond = model('vision_encoder', obs_img=obs_images_tensors, goal_img=goal_img_tensor, input_goal_mask=mask)

    # 3. Repeat conditioning vector for batch sampling
    if len(obs_cond.shape) == 2:
        obs_cond = obs_cond.repeat(num_samples, 1)
    else:
        obs_cond = obs_cond.repeat(num_samples, 1, 1)

    # 4. Generate trajectories using the diffusion model
    with torch.no_grad():
        noisy_action = torch.randn((num_samples, horizon, 2), device=device)
        naction = noisy_action
        noise_sched.set_timesteps(cfg['num_diffusion_iters'])
        for k in noise_sched.timesteps[:]:
            noise_pred = model('noise_pred_net', sample=naction, timestep=k, global_cond=obs_cond)
            naction = noise_sched.step(model_output=noise_pred, timestep=k, sample=naction).prev_sample
        
        waypoints = to_numpy(get_action(naction))
    
    return waypoints

def setup_plot(pos_data, traj_dir_name):
    """
    Initializes the matplotlib figure and axes for visualization.

    Returns:
        A tuple of (fig, artists) where artists is a dictionary of plot elements.
    """
    fig, (ax_path, ax_img, ax_goal) = plt.subplots(1, 3, figsize=(15, 5), gridspec_kw={'width_ratios': [2, 1, 1]})
    fig.suptitle(os.path.basename(traj_dir_name))
    plt.subplots_adjust(bottom=0.2)

    ax_path.plot(pos_data[:, 0], pos_data[:, 1], "b-", lw=0.8, label="Ground Truth Path")
    ax_path.set_aspect('equal', adjustable='box')
    ax_path.legend()

    pt, = ax_path.plot([], [], "ro")
    arrow = ax_path.quiver([], [], [], [], color="r", scale_units="xy", scale=1, headwidth=5)
    img_ax = ax_img.imshow(Image.new("RGB", (10, 10)))
    goal_ax_img = ax_goal.imshow(Image.new("RGB", (10, 10)))
    
    for ax in [ax_img, ax_goal]:
        ax.axis("off")

    artists = {
        'fig': fig, 'ax_path': ax_path, 'ax_img': ax_img, 'ax_goal': ax_goal,
        'pt': pt, 'arrow': arrow, 'img_ax': img_ax, 'goal_ax_img': goal_ax_img,
        'trajectory_plots': []
    }
    return fig, artists

def update_plot(artists, robot_pos, robot_yaw, obs_img, goal_img, waypoints):
    """
    Updates the plot with new data for a single timestep.
    """
    # Update robot position and orientation
    artists['pt'].set_data([robot_pos[0]], [robot_pos[1]])
    dx, dy = np.cos(robot_yaw), np.sin(robot_yaw)
    artists['arrow'].set_offsets(robot_pos)
    artists['arrow'].set_UVC(dx, dy)

    # Update images
    artists['img_ax'].set_data(obs_img)
    artists['goal_ax_img'].set_data(goal_img)

    # Clear and redraw waypoint trajectories
    for p in artists['trajectory_plots']:
        p.remove()
    artists['trajectory_plots'].clear()

    rot_matrix = np.array([
        [np.cos(robot_yaw), -np.sin(robot_yaw)],
        [np.sin(robot_yaw),  np.cos(robot_yaw)]
    ])

    for i in range(waypoints.shape[0]):
        local_traj = np.vstack([[0, 0], waypoints[i]])
        world_traj = local_traj @ rot_matrix.T + robot_pos
        line, = artists['ax_path'].plot(world_traj[:, 0], world_traj[:, 1], color="orange", alpha=0.5, lw=1.2)
        artists['trajectory_plots'].append(line)

    artists['fig'].canvas.draw_idle()


if __name__ == "__main__":
    # -------- User Paths and Config --------
    traj_dir = "testdata/go_stanford/no1vc_7_0"
    ckpt_path = "/home/jeong/visualnav-transformer/nomad.pth"
    config_path = "train/config/nomad.yaml"
    device = torch.device("cpu")

    # -------- Load Model and Config --------
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    model = load_model(ckpt_path, cfg, device)
    model.eval()

    noise_sched = DDPMScheduler(
        num_train_timesteps=cfg["num_diffusion_iters"],
        beta_schedule="squaredcos_cap_v2",
        clip_sample=True,
        prediction_type="epsilon",
    )
    
    # -------- Load Trajectory Data --------
    pos_data, yaw_data = load_traj_data(traj_dir)
    T = len(pos_data)
    
    # -------- Main Visualization Setup --------
    start_t = cfg['context_size'] + 1
    goal_t = T - 1
    
    goal_image = load_image(traj_dir, goal_t)
    goal_image_tensor = transform_images([goal_image], cfg['image_size'], center_crop=False).to(device)
    
    fig, artists = setup_plot(pos_data, traj_dir)
    artists['ax_img'].set_title(f"Observation (t={start_t})")
    artists['ax_goal'].set_title(f"Goal (t={goal_t})")

    # -------- Interactive Slider Logic --------
    def on_slider_change(t):
        t = int(t)
        
        # Get robot pose and observation queue for the current timestep
        robot_pos = pos_data[t]
        robot_yaw = yaw_data[t]
        current_obs_image = load_image(traj_dir, t)
        
        obs_queue = deque(maxlen=cfg['context_size'] + 1)
        start_idx_for_queue = max(0, t - cfg['context_size'])
        for i in range(start_idx_for_queue, t + 1):
            obs_queue.append(load_image(traj_dir, i))
            
        # Generate waypoints using the modular function
        waypoints = generate_waypoints(model, noise_sched, obs_queue, goal_image_tensor, cfg, device)
        
        # Update the plot using the modular function
        update_plot(artists, robot_pos, robot_yaw, current_obs_image, goal_image, waypoints)
        artists['ax_img'].set_title(f"Observation (t={t})")

    # Create and connect the slider
    ax_slider = plt.axes([0.2, 0.1, 0.65, 0.03])
    slider = Slider(
        ax=ax_slider, label='Time',
        valmin=start_t, valmax=goal_t - 1, valinit=start_t, valstep=1
    )
    slider.on_changed(on_slider_change)

    # Initial call to draw the first frame
    on_slider_change(start_t)

    plt.show() 