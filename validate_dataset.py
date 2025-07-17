import os
import pickle
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import argparse

def visualize_trajectory(dataset_dir):
    """
    Visualizes the collected trajectory data.
    - Loads trajectory data from a pickle file.
    - Displays images from the dataset.
    - Shows the robot's position on the trajectory plot.
    - A slider allows navigating through the frames.
    """
    pkl_path = os.path.join(dataset_dir, "traj_data.pkl")
    if not os.path.exists(pkl_path):
        print(f"Error: traj_data.pkl not found in '{dataset_dir}'")
        return

    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)

    positions = np.array(data['position'])
    yaws = np.array(data['yaw'])
    num_frames = len(positions)

    image_files = sorted([f for f in os.listdir(dataset_dir) if f.endswith('.jpg')], 
                         key=lambda x: int(os.path.splitext(x)[0]))

    if not image_files:
        print(f"Error: No jpg images found in '{dataset_dir}'")
        return

    fig, (ax_traj, ax_img) = plt.subplots(1, 2, figsize=(12, 6))
    plt.subplots_adjust(bottom=0.25)

    # Initial plot
    ax_traj.plot(positions[:, 0], positions[:, 1], 'b-', label='Full Trajectory')
    current_pos_plot, = ax_traj.plot(positions[0, 0], positions[0, 1], 'ro', markersize=10, label='Current Position')
    ax_traj.set_xlabel("X Position (m)")
    ax_traj.set_ylabel("Y Position (m)")
    ax_traj.set_title("Robot Trajectory")
    ax_traj.legend()
    ax_traj.axis('equal')
    ax_traj.grid(True)

    img_path = os.path.join(dataset_dir, image_files[0])
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_display = ax_img.imshow(img)
    ax_img.set_title(f"Frame 0 / {num_frames - 1}")
    ax_img.axis('off')

    # Slider
    ax_slider = plt.axes([0.25, 0.1, 0.65, 0.03])
    slider = Slider(ax_slider, 'Frame', 0, num_frames - 1, valinit=0, valstep=1)

    def update(val):
        frame_idx = int(slider.val)
        
        # Update trajectory plot
        current_pos_plot.set_data(positions[frame_idx, 0], positions[frame_idx, 1])
        
        # Update image
        img_path = os.path.join(dataset_dir, image_files[frame_idx])
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_display.set_data(img)
        
        ax_img.set_title(f"Frame {frame_idx} / {num_frames - 1}")
        fig.canvas.draw_idle()

    slider.on_changed(update)

    plt.show()


if __name__ == "__main__":
    # parser = argparse.ArgumentParser(description="Visualize a collected trajectory dataset.")
    # parser.add_argument("dataset_dir", type=str, help="Path to the dataset directory.")
    # args = parser.parse_args()

    dataset_dir = "traindata/go2_dataset_20250714_191924"

    visualize_trajectory(dataset_dir) 