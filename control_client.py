import time
import requests
import numpy as np
from typing import Optional, cast, List, Tuple

from minimal_mppi_py import MinimalMPPI, Pose2D, Twist2D, Costmap2D, MPPIConfig  # type: ignore
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
import base64
import io  # added for live visualization image decoding
from itertools import cycle  # for consistent color cycling

# -----------------------------------------------------------------------------
# External AI-server configuration
# -----------------------------------------------------------------------------
local = False
# URL of the server that receives observation images
AI_SERVER_URL = "http://127.0.0.1:8001" if local else "http://192.168.200.130:8001"
BASE_URL = "http://127.0.0.1:8000"# if local else "http://192.168.200.130:8000"
live_visualize = True

# Global context for live visualisation so we reuse the same figure across
# successive control-loop invocations.
viz_context: dict = {}
viz_context['loop_count'] = 0  # track number of loops visualised

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

    def get_costmap(self):
        """Pass-through to get the internal MPPI costmap object."""
        return self.controller.get_costmap()

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

    # --------------------------------------------------------------
    # Live visualization setup / reuse                             
    # --------------------------------------------------------------
    if live_visualize:
        global viz_context
        if 'fig' not in viz_context:
            # First-time creation of the figure and axes
            plt.ion()
            fig = plt.figure(figsize=(14, 7))
            gs = gridspec.GridSpec(2, 2, figure=fig, width_ratios=[1, 2])
 
            ax_frame = fig.add_subplot(gs[0, 0])
            ax_goal = fig.add_subplot(gs[1, 0])
            ax_traj = fig.add_subplot(gs[:, 1])
 
            # Show goal image once
            if os.path.exists(GOAL_IMAGE_PATH):
                try:
                    goal_img = plt.imread(GOAL_IMAGE_PATH)
                    ax_goal.imshow(goal_img)
                    ax_goal.axis('off')
                    ax_goal.set_title("Goal Image")
                except Exception as exc:
                    print(f"[WARN] Could not load goal image for viz: {exc}")
            else:
                ax_goal.text(0.5, 0.5, 'Goal image not found', ha='center', va='center')
                ax_goal.axis('off')
 
            img_handle = None
            arrow_handle = None
             
            ax_frame.set_title("Latest Camera Frame")
            ax_frame.axis("off")
            ax_traj.set_title("Robot Trajectory")
            ax_traj.set_aspect("equal", "box")
 
             # Cumulative line (black)
            cum_line, = ax_traj.plot([], [], 'k-', lw=1, label='Cumulative Path')
 
             # Costmap image background
            costmap_handle = ax_traj.imshow(np.zeros((1, 1), dtype=np.uint8), extent=(0, 1, 0, 1),
                                             origin='lower', cmap='gray_r', vmin=0, vmax=255, zorder=0)
 
            cum_path = []
            viz_context.update({
                "fig": fig,
                "ax_frame": ax_frame,
                "ax_traj": ax_traj,
                "ax_goal": ax_goal,
                "img_handle": img_handle,
                "arrow_handle": arrow_handle,
                "color_cycler": cycle(plt.rcParams['axes.prop_cycle'].by_key()['color']),
                "cum_line": cum_line,
                "cum_path": cum_path,
                "costmap_handle": costmap_handle,
            })
        else:
            fig = viz_context["fig"]
            ax_frame = viz_context["ax_frame"]
            ax_traj = viz_context["ax_traj"]
            ax_goal = viz_context["ax_goal"]
            img_handle = viz_context.get("img_handle")
            arrow_handle = viz_context.get("arrow_handle")
            cum_line = viz_context["cum_line"]
            cum_path = viz_context["cum_path"]
            costmap_handle = viz_context["costmap_handle"]
 
        # Choose a new colour for this loop and create a new line for its path
        loop_color = next(viz_context['color_cycler'])
        traj_line, = ax_traj.plot([], [], marker='.', lw=1, color=loop_color,
                                   label=f"Path loop {viz_context['loop_count'] + 1}")
 
        # Remove any existing arrow so we can redraw cleanly
        if arrow_handle is not None:
            arrow_handle.remove()
            arrow_handle = None
 
        # (Re)plot the planned trajectory for this run
        try:
            traj_xy = np.array([(pt[0], pt[1]) for pt in trajectory[1:]])  # skip current pose
            if traj_xy.size > 0:
                ax_traj.plot(traj_xy[:, 0], traj_xy[:, 1], linestyle='--', marker='.', lw=1,
                             color=loop_color, alpha=0.7, label=f"Planned Traj {viz_context['loop_count'] + 1}")
 
        except Exception:
            pass
 
        ax_traj.legend(loc="upper right", fontsize='x-small')
 
        # Stash updated handles back into the context
        viz_context["img_handle"] = img_handle
        viz_context["arrow_handle"] = arrow_handle
        viz_context["traj_line_current"] = traj_line
        viz_context['loop_count'] += 1
        viz_context['cum_line'] = cum_line
        viz_context['cum_path'] = cum_path

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

        # Record current pose for history (before issuing new command)
        path_hist.append([pose[0], pose[1]])
        yaw_hist.append(pose[2])

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
            # Previously we appended odometry from result – keeping for debugging if needed
            # path_hist.append([result["odometry"]["x"], result["odometry"]["y"]])
            # yaw_hist.append(result["odometry"]["theta"])
        except requests.RequestException as exc:
            print(f"[WARN] Failed to send step: {exc}")

        # ----------------------------------------------------------
        # Live visualization update                                 
        # ----------------------------------------------------------
        if live_visualize:
            fig = viz_context["fig"]
            ax_frame = viz_context["ax_frame"]
            ax_traj = viz_context['ax_traj']
            costmap_handle = viz_context['costmap_handle']
            # ---- Update costmap background ----
            costmap = controller.get_costmap()
            if costmap and costmap.get_size_x() > 0:
                costmap_np = costmap.get_costmap_numpy()
                origin_x = pose[0] - (costmap.get_size_x() * costmap.get_resolution()) / 2.0
                origin_y = pose[1] - (costmap.get_size_y() * costmap.get_resolution()) / 2.0
                extent = (origin_x, origin_x + costmap.get_size_x() * costmap.get_resolution(),
                          origin_y, origin_y + costmap.get_size_y() * costmap.get_resolution())

                costmap_handle.set_data(costmap_np)
                costmap_handle.set_extent(extent)

            # Update camera frame on the left
            if img_b64 is not None:
                try:
                    img_np = plt.imread(io.BytesIO(base64.b64decode(img_b64)), format="jpeg")
                    img_handle = viz_context.get("img_handle")
                    if img_handle is None:
                        viz_context["img_handle"] = ax_frame.imshow(img_np)
                    else:
                        img_handle.set_data(img_np)
                except Exception as exc:
                    print(f"[WARN] Failed to update frame visualization: {exc}")

            # Update trajectory on the right
            if path_hist:
                path_arr = np.array(path_hist)
                traj_line = viz_context["traj_line_current"]
                traj_line.set_data(path_arr[:, 0], path_arr[:, 1])
                ax_traj.relim()
                ax_traj.autoscale_view(True, True, True)

                # Update robot orientation arrow
                try:
                    cur_x, cur_y = path_arr[-1]
                    cur_theta = yaw_hist[-1] if yaw_hist else 0.0
                    arrow_handle = viz_context.get("arrow_handle")
                    # Remove previous arrow
                    if arrow_handle is not None:
                        arrow_handle.remove()
                    scale_len = 0.25  # shorter arrow
                    arrow_handle = ax_traj.quiver(cur_x, cur_y,
                                                  np.cos(cur_theta) * scale_len,
                                                  np.sin(cur_theta) * scale_len,
                                                  color='r', scale_units='xy', scale=1, width=0.004)
                    viz_context["arrow_handle"] = arrow_handle
 
                    # -- cumulative path update --
                    cum_path = viz_context['cum_path']
                    cum_line = viz_context['cum_line']
                    cum_path.append([cur_x, cur_y])
                    cum_arr = np.array(cum_path)
                    cum_line.set_data(cum_arr[:, 0], cum_arr[:, 1])
                except Exception as exc:
                    print(f"[WARN] Failed to update robot orientation: {exc}")
 
            fig.canvas.draw_idle()
            plt.pause(0.001)

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

    simple_target = (start_pose[0] + 0, start_pose[1], start_pose[2])
    first_traj = [start_pose, simple_target]
    print(f"[INFO] Starting first control loop with trajectory: {first_traj}")

    final_pose1, path1, yaw1 = run_control_loop(controller, first_traj, 5)

    # --------------------------------------------------------------
    # 2. REPEATED LOOPS USING AI-PREDICTED WAYPOINTS (5 iterations)
    # --------------------------------------------------------------

    loops_to_run = 5
    current_pose = final_pose1
    full_path = path1
    full_yaw = yaw1
    last_local_wps = None

    for loop_idx in range(loops_to_run):
        print(f"[INFO] Loop {loop_idx + 1}/{loops_to_run}: Fetching waypoints from AI server…")
        local_wps_raw = fetch_waypoints()

        # Prepare usable (N,2) waypoint array
        if local_wps_raw is not None and local_wps_raw.size > 0:
            if local_wps_raw.ndim == 3:
                local_wps = local_wps_raw[0]  # first sample in batch
            elif local_wps_raw.ndim == 2 and local_wps_raw.shape[1] == 2:
                local_wps = local_wps_raw
            else  :
                print(f"[WARN] Unexpected waypoint shape {local_wps_raw.shape}; skipping.")
                local_wps = None
        else:
            local_wps = None

        if local_wps is not None and local_wps.size > 0:
            global_wps_xy = local_to_global(local_wps, current_pose)
            traj = [current_pose] + [(pt[0], pt[1], current_pose[2]) for pt in global_wps_xy]
            last_local_wps = local_wps  # remember last successful prediction
        else:
            # Fallback: simple forward motion if no waypoints
            print("[WARN] No waypoints received – using fallback target.")
            traj = [current_pose, (current_pose[0] + 0.5, current_pose[1], current_pose[2])]

        print(f"[INFO] Executing control loop for trajectory with {len(traj)} points…")
        current_pose, path_seg, yaw_seg = run_control_loop(controller, traj, 3)

        full_path = np.vstack([full_path, path_seg])
        full_yaw = np.hstack([full_yaw, yaw_seg])

    # --------------------------------------------------------------
    # 3. Plot combined results with the last batch of waypoints
    # --------------------------------------------------------------

    plot_trajectory_and_waypoints(full_path, full_yaw, last_local_wps)

