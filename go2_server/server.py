from fastapi import FastAPI  # type: ignore
from pydantic import BaseModel  # type: ignore
from typing import List, Tuple, Optional

from env import Robot
from fastapi.responses import StreamingResponse  # type: ignore
import io
import cv2  # type: ignore
import base64
import time
from collections import deque
import threading
from aiortc import MediaStreamTrack  # type: ignore

# Instantiate FastAPI application
app = FastAPI()

# Create a single shared Robot instance
# Adjust the IP address if your robot uses a different one
# robot = Robot(ip="172.30.1.80")
robot = Robot(ip = "192.168.200.157")
# robot = Robot(ip = "192.168.200.255")

# ---------------------------------------------------------------------------
# Video frame buffering
# ---------------------------------------------------------------------------


class FrameBuffer:
    """Thread-safe circular buffer storing recent camera frames with metadata."""

    def __init__(self, maxlen: int = 30):
        self._frames = deque(maxlen=maxlen)
        self._lock = threading.Lock()

    def add(self, img, ts: float, pose):
        with self._lock:
            self._frames.append({"image": img, "timestamp": ts, "pose": pose})

    def last_n(self, n: int = 1):
        if n <= 0:
            return []
        with self._lock:
            return list(self._frames)[-n:]

    def latest_image(self):
        with self._lock:
            if not self._frames:
                return None
            return self._frames[-1]["image"]


frame_buffer = FrameBuffer(maxlen=30)


# Asynchronous callback for aiortc track
async def _recv_camera_stream(track: MediaStreamTrack):
    while True:
        try:
            frame = await track.recv()
            img = frame.to_ndarray(format="bgr24")
            ts = time.time()
            pose_snapshot = robot.pose  # (x, y, theta)

            frame_buffer.add(img, ts, pose_snapshot)
        except Exception:
            break


# Register callback & request video stream
try:
    robot._conn.video.add_track_callback(_recv_camera_stream)  # type: ignore
    robot._conn.video.switchVideoChannel(True)  # type: ignore
except Exception as exc:
    print(f"Video stream activation failed: {exc}")


# @app.on_event("startup")
# async def startup_event():
#     """Connect to the robot when the server starts."""
#     # The Robot class now needs an explicit connection call within the running loop
#     await robot.connect_in_loop()


# Global variables
global_path: List[Tuple[float, float, float]] = []
global_goal_pose: Optional[Tuple[float, float, float]] = None


class VelocityCmd(BaseModel):
    """Velocity command for one control step."""

    vx: float = 0.0  # forward/backward (m/s), +x forward
    vy: float = 0.0  # left/right (m/s),  +y left
    vw: float = 0.0  # yaw rate (rad/s),  +z CCW
    dt: float = 0.1  # optional blocking delay after sending the command


# ------------------------------------------------------------
# Helper utilities
# ------------------------------------------------------------

def _pose_dict(x: float, y: float, theta: float):
    """Return a dictionary with odometry components."""
    return {"x": x, "y": y, "theta": theta}


@app.get("/pose")
async def get_pose():
    """Return the latest robot odometry (x, y, yaw)."""
    x, y, theta = robot.pose
    return {"odometry": {"x": x, "y": y, "theta": theta}}


@app.get("/scan")
async def get_latest_scan():
    """Return the most recent laser scan (angles & ranges)."""
    angles, ranges = robot.scan_polar
    return {"angles": angles.tolist(), "ranges": ranges.tolist()}


@app.get("/video")
async def get_video_feed():
    """Return the latest video frame as a JPEG image."""
    frame = frame_buffer.latest_image()

    if frame is None:
        return {"error": "no frame available"}

    # Encode frame as JPEG
    _, img_encoded = cv2.imencode(".jpg", frame)
    return StreamingResponse(io.BytesIO(img_encoded.tobytes()), media_type="image/jpeg")


@app.post("/step")
async def step_once(cmd: VelocityCmd):
    """Send a single velocity command, then return updated pose & scan."""

    # Send command to robot
    # Cap vx to [-0.5, 0.5] and vy to [-0.2, 0.2]
    vx = max(-0.5, min(0.5, cmd.vx))
    vy = max(-0.2, min(0.2, cmd.vy))
    robot.step(vx, vy, cmd.vw, dt=cmd.dt)

    # After motion delay, fetch latest state
    x, y, theta = robot.pose
    angles, ranges = robot.scan_polar

    return {
        "collision": False,  # placeholder until collision detection implemented
        "odometry": _pose_dict(x, y, theta),
        "angles": angles.tolist(),
        "ranges": ranges.tolist(),
    }


# ---------------------------------------------------------------------------
# New endpoint: fetch recent camera frames with timestamp & pose
# ---------------------------------------------------------------------------

@app.get("/frames")
async def get_recent_frames(n: int = 1):
    """Return the last *n* camera frames as base64 JPEG along with timestamp and pose."""

    # Sanity check
    if n < 1:
        return {"frames": []}

    frames = frame_buffer.last_n(n)
    results = []

    for entry in frames:
        img = entry.get("image")
        ts = entry.get("timestamp")
        pose = entry.get("pose", (None, None, None))

        if img is None:
            continue

        # JPEG encode
        success, img_encoded = cv2.imencode(".jpg", img)
        if not success:
            continue

        img_b64 = base64.b64encode(img_encoded).decode("utf-8")

        results.append(
            {
                "timestamp": ts,
                "pose": {"x": pose[0], "y": pose[1], "theta": pose[2]},
                "image": img_b64,
            }
        )

    return {"frames": results}


class PathPoint(BaseModel):
    """Single point in a path, with position and orientation."""
    x: float
    y: float 
    theta: float

class Path(BaseModel):
    """A sequence of path points."""
    points: List[PathPoint]

class GoalPose(BaseModel):
    """A single goal pose with position and orientation."""
    x: float
    y: float 
    theta: float

@app.post("/path")
async def set_global_path(path: Path):
    """Store a new global path for the robot to follow."""
    global global_path
    global_path = [(p.x, p.y, p.theta) for p in path.points]
    return {
        "status": "received",
        "path_length": len(global_path),
    }

@app.get("/path")
async def get_global_path():
    """Retrieve the stored global path."""
    # Return only x and y for visualization
    path_xy = [[x, y] for x, y, theta in global_path]
    return {"path": path_xy}
    
@app.post("/goal_pose")
async def set_goal_pose(goal: GoalPose):
    """Store a new goal pose."""
    global global_goal_pose
    global_goal_pose = (goal.x, goal.y, goal.theta)
    return {"status": "received", "goal_pose": global_goal_pose}

@app.get("/goal_pose")
async def get_goal_pose():
    """Retrieve the stored goal pose."""
    if global_goal_pose:
        return {"goal_pose": {"x": global_goal_pose[0], "y": global_goal_pose[1], "theta": global_goal_pose[2]}}
    return {"goal_pose": None}


# ---------------------------------------------------------------------------
# Graceful shutdown handling
# ---------------------------------------------------------------------------

@app.on_event("shutdown")
async def shutdown_event():
    """Ensure the WebRTC connection is closed when the server stops."""
    try:
        # The new close method handles shutting down the loop
        robot.close()
    except Exception:
        pass

#uvicorn env_server:app --host 0.0.0.0 --port 8000
