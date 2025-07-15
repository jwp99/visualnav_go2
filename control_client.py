import time
import requests
import numpy as np
from typing import Optional, cast, List, Tuple

from minimal_mppi_py import MinimalMPPI, Pose2D, Twist2D, Costmap2D, MPPIConfig  # type: ignore
import matplotlib.pyplot as plt
import os
import base64

# -----------------------------------------------------------------------------
# External AI-server configuration
# -----------------------------------------------------------------------------

# URL of the server that receives observation images
AI_SERVER_URL = "http://127.0.0.1:8001"

# Local goal image to upload to the AI server at startup
GOAL_IMAGE_PATH = "testdata/go2_dataset_20250714_191924/90.jpg"


class ControllerWrapper :
    def __init__(self):

        config = MPPIConfig()

        config.batch_size = 2000
        config.time_steps = 30
        config.iteration_count = 1
        config.obstacle_weight = 0.004
        config.goal_dist_weight = 5
        config.path_dist_weight = 0
        config.path_align_weight = 15
        config.path_follow_weight = 0
        config.path_angle_weight = 2
        config.offset_from_furthest = 30 
        config.prefer_forward_weight = 4
        config.gamma = 0.015
        config.prune_distance = 2.5

        config.shift_control_sequence = True
        config.seed = 41
        config.temperature = 0.5
        config.vx_std = 0.7
        config.vy_std = 0.25

        self.controller = MinimalMPPI()
        self.controller.set_config(config)
        self.controller.set_debug_level(0)

        

    def get_action(self, pose, vel, traj, angles, ranges, plot = True):
        # Convert pose tuple to Pose2D
        current_pose = Pose2D()
        current_pose.x, current_pose.y, current_pose.theta = pose

        # Convert velocity tuple to Twist
        current_velocity = Twist2D() 
        current_velocity.vx, current_velocity.vy, current_velocity.vtheta = vel

        # Convert trajectory points to list of Pose2D
        global_plan = []
        for point in traj:
            pose = Pose2D()
            pose.x, pose.y, pose.theta = point
            global_plan.append(pose)
            
        costmap = self.controller.get_costmap()
        new_origin_x = current_pose.x - (costmap.get_size_x() * costmap.get_resolution()) / 2.0
        new_origin_y = current_pose.y - (costmap.get_size_y() * costmap.get_resolution()) / 2.0
        costmap.updateOrigin(new_origin_x, new_origin_y)
        costmap.update_with_scan(current_pose, ranges, angles)
        costmap.inflate(0.5, 0.2, 10.0)


        best_velocity = self.controller.compute_velocity_commands(
            current_pose,
            current_velocity, 
            global_plan,
        )

        if plot:
            optimal_traj = self.controller.get_optimal_trajectory()
            if optimal_traj is not None and len(optimal_traj.x) > 0:
                opt_np = np.concatenate([optimal_traj.x, optimal_traj.y, optimal_traj.yaws]).T
            self.plot((current_pose.x, current_pose.y, current_pose.theta), title=f"local costmap {time.strftime('%H:%M:%S')}",trajectory=opt_np, save=True)


        return (best_velocity.vx, best_velocity.vy, best_velocity.vtheta)


    def plot(self, pose, title="Costmap", trajectory=None, save = True):
        """Helper function to plot a costmap object from the C++ bindings."""
      
        # Plot the raw costmap matrix
        fig = plt.figure()
        plt.title(title)
        plt.quiver(pose[0], pose[1], np.cos(pose[2]), np.sin(pose[2]), color='r', scale=10, width=0.01)

        costmap = self.controller.get_costmap()
        self.origin_x = pose[0] - (costmap.get_size_x() * costmap.get_resolution()) / 2.0
        self.origin_y = pose[1] - (costmap.get_size_y() * costmap.get_resolution()) / 2.0

        extent = (self.origin_x, self.origin_x + costmap.get_size_x() * costmap.get_resolution(),
                    self.origin_y, self.origin_y + costmap.get_size_y() * costmap.get_resolution())
        plt.imshow(costmap.get_costmap_numpy(), extent=extent, origin='lower', cmap='gray_r', vmin=0, vmax=255)
        
        if trajectory is not None:
            traj_x = [p[0] for p in trajectory]
            traj_y = [p[1] for p in trajectory]
            plt.plot(traj_x, traj_y, 'y-')
            
        plt.colorbar(label="Cost")
        plt.xlabel("X (grid cells)")
        plt.ylabel("Y (grid cells)")

        if save:
            timestamp_dir = time.strftime('%Y%m%d_%H%M%S')
            data_dir = f'data/{timestamp_dir}'
            if not os.path.exists(data_dir):
                os.makedirs(data_dir)
            filename = f"{data_dir}/{title.replace(' ', '_')}.png"
            plt.savefig(filename)
            plt.close(fig)
        else: 
            plt.show()


            




# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

# Base URL of the FastAPI server (adjust if the server is running elsewhere)
BASE_URL = "http://127.0.0.1:8000"

# Number of control iterations to perform (-1 for infinite loop)
MAX_STEPS = 5  # set to a positive integer for a finite run

# Time to wait between control iterations (seconds)
CONTROL_PERIOD = 0.1

# -----------------------------------------------------------------------------
# Helper functions for HTTP requests
# -----------------------------------------------------------------------------

def fetch_pose():
    """Retrieve the robot's current pose (x, y, theta)."""
    resp = requests.get(f"{BASE_URL}/pose", timeout=2.0)
    resp.raise_for_status()
    odo = resp.json()["odometry"]
    return odo["x"], odo["y"], odo["theta"]


def fetch_scan():
    """Retrieve the latest laser scan (angles, ranges) as NumPy arrays."""
    resp = requests.get(f"{BASE_URL}/scan", timeout=2.0)
    resp.raise_for_status()
    data = resp.json()
    angles = np.array(data["angles"], dtype=np.float32)
    ranges = np.array(data["ranges"], dtype=np.float32)
    return angles, ranges


def send_step(vx: float, vy: float, vw: float, dt: float = CONTROL_PERIOD):
    """Send a velocity command to the robot."""
    payload = {"vx": vx, "vy": vy, "vw": vw, "dt": dt}
    resp = requests.post(f"{BASE_URL}/step", json=payload, timeout=2.0)
    resp.raise_for_status()
    return resp.json()


# -----------------------------------------------------------------------------
# Image handling helpers
# -----------------------------------------------------------------------------


def fetch_latest_frame(n: int = 1):
    """Fetch the most recent camera frame from the /frames endpoint.

    Returns
    -------
    str | None
        Base-64 encoded JPEG string if available, otherwise None.
    """
    try:
        resp = requests.get(f"{BASE_URL}/frames?n={n}", timeout=2.0)
        resp.raise_for_status()
        frames = resp.json().get("frames", [])
        if not frames:
            return None
        return frames[-1]["image"]
    except requests.RequestException as exc:
        print(f"[WARN] Failed to fetch frame: {exc}")
        return None


def send_observation_image(image_b64: str):
    """Send a base-64 JPEG string to the AI server by uploading it as a file.

    The AI server expects a multipart/form-data POST at /add_observation_image/
    with the field name `observation_image`.
    """
    if image_b64 is None:
        return

    try:
        img_bytes = base64.b64decode(image_b64)
    except Exception as exc:
        print(f"[WARN] Could not decode image for forwarding: {exc}")
        return

    files = {
        "observation_image": ("frame.jpg", img_bytes, "image/jpeg"),
    }

    try:
        resp = requests.post(
            f"{AI_SERVER_URL}/add_observation_image/",  # trailing slash matters
            files=files,
            timeout=4.0,
        )
        resp.raise_for_status()
    except requests.RequestException as exc:
        print(f"[WARN] Failed to forward image to AI server: {exc}")


# -----------------------------------------------------------------------------
# Waypoint generation & visualization helpers
# -----------------------------------------------------------------------------


def fetch_waypoints():
    """Request waypoint predictions from the AI server and return as np.ndarray."""
    try:
        resp = requests.post(f"{AI_SERVER_URL}/generate_waypoints/", timeout=6.0)
        resp.raise_for_status()
        waypoints = np.array(resp.json().get("waypoints", []), dtype=float)
        return waypoints
    except requests.RequestException as exc:
        print(f"[WARN] Failed to fetch waypoints: {exc}")
        return None


def plot_trajectory_and_waypoints(path_xy: np.ndarray, yaw_arr: np.ndarray, waypoints: Optional[np.ndarray], save_dir: str = "testdata"):
    """Plot the robot path and predicted waypoints.

    Parameters
    ----------
    path_xy : (T,2) array of robot positions in world frame
    yaw_arr : (T,) array of yaws (rad)
    waypoints : (N,2) array in robot local frame
    save_dir : directory where the plot will be saved
    """
    if path_xy.size == 0:
        print("[WARN] No trajectory to plot.")
        return

    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)

    fig, ax_arr = plt.subplots(1, 1, figsize=(6, 6))
    ax = cast(plt.Axes, ax_arr)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title("Trajectory with Predicted Waypoints")

    # Plot path
    ax.plot(path_xy[:, 0], path_xy[:, 1], "b-", lw=0.8, label="Robot Path")

    # Plot current robot pose
    robot_pos = path_xy[-1]
    robot_yaw = yaw_arr[-1]
    ax.plot(robot_pos[0], robot_pos[1], "ro", label="Current Pose")
    dx, dy = np.cos(robot_yaw), np.sin(robot_yaw)
    ax.quiver(robot_pos[0], robot_pos[1], dx, dy, color="r", scale_units="xy", scale=1, headwidth=5)

    # Plot waypoints if available
    if waypoints is not None and waypoints.size > 0:
        R = np.array([[np.cos(robot_yaw), -np.sin(robot_yaw)], [np.sin(robot_yaw), np.cos(robot_yaw)]])
        for wp in waypoints:
            local_traj = np.vstack([[0, 0], wp])
            world_traj = local_traj @ R.T + robot_pos
            ax.plot(world_traj[:, 0], world_traj[:, 1], color="orange", alpha=0.7)

    ax.legend()

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(save_dir, f"waypoints_plot_{timestamp}.png")
    plt.savefig(out_path)
    plt.close(fig)
    print(f"[INFO] Waypoints plot saved to {out_path}")


# -----------------------------------------------------------------------------
# Goal image provisioning
# -----------------------------------------------------------------------------


def send_goal_image_once():
    """Upload the predefined goal image to the AI server.

    Safe to call multiple times—the server just overwrites the old goal image.
    """
    if not os.path.exists(GOAL_IMAGE_PATH):
        print(f"[ERROR] Goal image file not found at {GOAL_IMAGE_PATH}.")
        return

    try:
        with open(GOAL_IMAGE_PATH, "rb") as f:
            files = {"goal_image": (os.path.basename(GOAL_IMAGE_PATH), f, "image/jpeg")}
            resp = requests.post(
                f"{AI_SERVER_URL}/set_goal_image/",
                files=files,
                timeout=6.0,
            )
            resp.raise_for_status()
            print("[INFO] Goal image successfully uploaded to AI server.")
    except requests.RequestException as exc:
        print(f"[WARN] Could not upload goal image: {exc}")


# -----------------------------------------------------------------------------
# Geometry helpers
# -----------------------------------------------------------------------------


def local_to_global(local_xy: np.ndarray, robot_pose: Tuple[float, float, float]) -> np.ndarray:
    """Convert a set of (x,y) points expressed in the robot frame to world frame."""
    x, y, theta = robot_pose
    R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    return local_xy @ R.T + np.array([x, y])


# -----------------------------------------------------------------------------
# Modular control loop
# -----------------------------------------------------------------------------


def run_control_loop(
    controller: "ControllerWrapper",
    trajectory: List[Tuple[float, float, float]],
    max_steps: int,
    initial_vel: Tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> Tuple[Tuple[float, float, float], np.ndarray, np.ndarray]:
    """Run closed-loop control for *max_steps* iterations.

    Returns
    -------
    final_pose : (x,y,theta) of last reported odometry
    path_hist  : (T,2) numpy array of positions
    yaw_hist   : (T,) numpy array of headings
    """

    path_hist: List[List[float]] = []
    yaw_hist: List[float] = []

    current_vel = initial_vel
    step_count = 0

    while max_steps < 0 or step_count < max_steps:
        try:
            pose = fetch_pose()
            angles, ranges = fetch_scan()
        except requests.RequestException as exc:
            print(f"[WARN] Communication error: {exc}. Retrying...")
            time.sleep(1.0)
            continue

        # Send latest camera frame to AI server (observation collection)
        img_b64 = fetch_latest_frame()
        if img_b64 is not None:
            send_observation_image(img_b64)

        # Compute control
        vx, vy, vw = controller.get_action(
            pose, current_vel, trajectory, angles, ranges, plot=False
        )

        # Send command
        try:
            result = send_step(vx, vy, vw)
            path_hist.append([result["odometry"]["x"], result["odometry"]["y"]])
            yaw_hist.append(result["odometry"]["theta"])
        except requests.RequestException as exc:
            print(f"[WARN] Failed to send step: {exc}")

        current_vel = (vx, vy, vw)
        step_count += 1
        time.sleep(CONTROL_PERIOD)

    final_pose_tuple: Tuple[float, float, float] = (
        float(path_hist[-1][0]),
        float(path_hist[-1][1]),
        float(yaw_hist[-1]),
    )
    return final_pose_tuple, np.array(path_hist, dtype=float), np.array(yaw_hist, dtype=float)


# -----------------------------------------------------------------------------
# Main control loop
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    controller = ControllerWrapper()

    # Ensure the AI server has the goal image
    send_goal_image_once()

    # --------------------------------------------------------------
    # 1. FIRST LOOP – simple two-point trajectory to gather images
    # --------------------------------------------------------------

    try:
        start_pose = fetch_pose()
    except requests.RequestException as exc:
        print(f"[ERROR] Cannot fetch initial pose: {exc}")
        exit(1)

    simple_target = (start_pose[0] + 0.5, start_pose[1], start_pose[2])
    first_traj = [start_pose, simple_target]
    print(f"[INFO] Starting first control loop with trajectory: {first_traj}")

    final_pose1, path1, yaw1 = run_control_loop(controller, first_traj, 5)

    # --------------------------------------------------------------
    # 2. Fetch waypoints, convert to global, build new trajectory
    # --------------------------------------------------------------

    print("[INFO] Fetching waypoints from AI server for second loop...")
    local_wps_raw = fetch_waypoints()

    # Prepare a 2-D array of (M,2) points
    if local_wps_raw is not None and local_wps_raw.size > 0:
        if local_wps_raw.ndim == 3:  # (num_samples, horizon, 2)
            local_wps = local_wps_raw[3]  # take first sample
        elif local_wps_raw.ndim == 2 and local_wps_raw.shape[1] == 2:
            local_wps = local_wps_raw
        else:
            print(f"[WARN] Unexpected waypoint shape {local_wps_raw.shape}; skipping.")
            local_wps = None
    else:
        local_wps = None

    if local_wps is not None:
        global_wps_xy = local_to_global(local_wps, final_pose1)
        second_traj: List[Tuple[float, float, float]] = [final_pose1] + [
            (pt[0], pt[1], final_pose1[2]) for pt in global_wps_xy
        ]
        print(f"[INFO] Second trajectory (global waypoints): {second_traj}")

        # ----------------------------------------------------------
        # 3. SECOND LOOP – follow waypoint trajectory
        # ----------------------------------------------------------

        final_pose2, path2, yaw2 = run_control_loop(controller, second_traj, 10)

        # Combine paths for visualization
        full_path = np.vstack([path1, path2])
        full_yaw = np.hstack([yaw1, yaw2])
    else:
        print("[WARN] No waypoints received – skipping second loop.")
        full_path = path1
        full_yaw = yaw1
        local_wps = None

    # --------------------------------------------------------------
    # 4. Plot results with waypoints
    # --------------------------------------------------------------

    plot_trajectory_and_waypoints(full_path, full_yaw, local_wps)

