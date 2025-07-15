import asyncio
import threading
import time
from typing import Tuple

import numpy as np  # type: ignore


import matplotlib.pyplot as plt  # type: ignore
from scipy.ndimage import distance_transform_edt  # type: ignore


from go2_webrtc_driver.webrtc_driver import Go2WebRTCConnection, WebRTCConnectionMethod
from go2_webrtc_driver.constants import RTC_TOPIC, SPORT_CMD

# Raw ULIDAR coordinates are reported in 5 centimetres (empirically); convert to
# metres by multiplying by 0.05.  Adjust this value if your hardware firmware
# uses a different unit.
_LIDAR_UNIT_SCALE = 0.05
MAX_RANGE = 15.0

# Angle offset so that bin 0 corresponds to robot-forward (+Y axis).
# Unitree base frame: +X right, +Y forward ⇒ arctan2 gives 0° at +X.
# To rotate so 0° = +Y we subtract 90°.
_ANGLE_OFFSET_DEG = 90  # rotate base-frame so +Y (forward) becomes 0°

class Robot:
    """High-level wrapper around the Unitree Go2 WebRTC interface.

    The class starts its own asynchronous event loop in a background thread so that
    the public API (`step`, `pose`, `lasers`) can be used from ordinary synchronous
    code.  Under the hood we subscribe to the relevant RTC topics so that pose and
    LIDAR data are continuously updated while velocity commands are sent on demand.
    """

    # Maximum number of lidar snapshots to keep in memory
    _MAX_SCAN_HISTORY = 5

    def __init__(self, ip: str = "172.30.1.80") -> None:
        # Robot state
        self.x: float = 0.0
        self.y: float = 0.0
        self.theta: float = 0.0  # yaw (rad)

        # Data buffers
        self._scans = []  # list of recent scans for optional post-processing
        self._latest_scan: np.ndarray | None = None  # 1-D distance vector (len 360)
        self._latest_pts: np.ndarray | None = None   # Raw XY points (N×2)

        # Thread-safety
        self._pose_lock = threading.Lock()
        self._scan_lock = threading.Lock()

        # Async event loop running in a separate thread
        self._loop = asyncio.new_event_loop()
        self._loop_thread = threading.Thread(target=self._run_loop, daemon=True)
        self._loop_thread.start()

        # Establish the WebRTC connection synchronously from caller's perspective
        fut = asyncio.run_coroutine_threadsafe(self._async_init(ip), self._loop)
        fut.result()  # block until connected

    # ---------------------------------------------------------------------
    # Public API
    # ---------------------------------------------------------------------

    def step(self, cmd_vx: float, cmd_vy: float, cmd_vw: float, dt: float = 0.02) -> None:
        """Send velocity command and (optionally) wait *dt* seconds.

        Parameters
        ----------
        cmd_vx : float
            Forward/backward linear velocity in m/s (+x forward)
        cmd_vy : float
            Left/right linear velocity in m/s (+y left)
        cmd_vw : float
            Yaw rate in rad/s (+z CCW)
        dt : float, optional
            Blocking delay applied after sending the command.  Useful when this
            method is called from a control loop running at fixed period.
        """

        async def _send():
            await self._conn.datachannel.pub_sub.publish_request_new(
                RTC_TOPIC["SPORT_MOD"],
                {
                    "api_id": SPORT_CMD["Move"],
                    "parameter": {"x": cmd_vx, "y": cmd_vy, "z": cmd_vw},
                },
            )

        asyncio.run_coroutine_threadsafe(_send(), self._loop)

        if dt is not None and dt > 0:
            time.sleep(dt)

    # ------------------------------------------------------------------
    # Sensors – pose & 2-D laser scans
    # ------------------------------------------------------------------

    @property
    def pose(self) -> Tuple[float, float, float]:
        """Return the latest (x, y, yaw) estimate provided by the robot."""
        with self._pose_lock:
            return self.x, self.y, self.theta

    @property
    def lasers(self) -> np.ndarray:
        """Return the latest laser scan as a 1-D distance vector.

        The vector has length 360 (one cell per degree).  Element *i* stores the
        distance (in metres) to the closest return whose bearing rounds to *i*
        degrees in the robot base frame (0–359, CCW, where 0° is the +x axis).
        Cells without a valid return are set to 0.  A copy is returned so the
        caller is free to modify it.
        """
        with self._scan_lock:
            if hasattr(self, "_latest_scan") and self._latest_scan is not None:
                return self._latest_scan.copy()
            return np.full(360, np.nan, dtype=np.float32)

    @property
    def scan_polar(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return tuple (angles_rad, ranges).

        * *angles_rad* – numpy array of shape (360,) with bearings in radians
          (0–2π, CCW, 0 rad = +x).
        * *ranges* – distance vector identical to *lasers*.
        """
        angles_rad = np.deg2rad(np.arange(360, dtype=np.float32))
        return angles_rad, self.lasers

    # ------------------------------------------------------------------
    # Convenience view: 180° forward field-of-view (−90…+90°)
    # ------------------------------------------------------------------

    @property
    def lasers_fov180(self) -> np.ndarray:
        """Return 180-element distance vector covering −90°…+90° around front.

        Index 0   → −90° (right side)
        Index 90  →   0° (straight ahead)
        Index 179 → +89° (left side)
        """
        full = self.lasers  # length 360
        if full.size != 360:
            return full  # fallback – unexpected size

        right_half = full[270:360]  # −90…−1°
        left_half  = full[0:90]     #   0…+89°
        return np.concatenate((right_half, left_half))  # shape (180,)

    @property
    def scan_polar_fov180(self) -> Tuple[np.ndarray, np.ndarray]:
        """Tuple (angles_rad, ranges) for −90…+90° FOV.

        Angles are centred on 0 rad straight ahead.
        """
        angles_deg = np.arange(-90, 90, 1, dtype=np.float32)
        angles_rad = np.deg2rad(angles_deg)
        return angles_rad, self.lasers_fov180

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _run_loop(self):
        """Target for the background thread: run the asyncio loop forever."""
        asyncio.set_event_loop(self._loop)
        self._loop.run_forever()
    async def _async_init(self, ip: str):
        """Asynchronous part of the constructor – executed in the background loop."""
        # 1. Connect via WebRTC
        self._conn = Go2WebRTCConnection(WebRTCConnectionMethod.LocalSTA, ip=ip)
        await self._conn.connect()

        # 2. (Optional) disable traffic saving so we get full LIDAR throughput
        await self._conn.datachannel.disableTrafficSaving(True)

        # Ensure we are using the voxel decoder for compressed voxel maps
        self._conn.datachannel.set_decoder(decoder_type="libvoxel")

        # 3. Turn LIDAR on
        self._conn.datachannel.pub_sub.publish_without_callback(
            RTC_TOPIC["ULIDAR_SWITCH"], "on"
        )

        # 4. Subscribe to pose updates
        self._conn.datachannel.pub_sub.subscribe(
            RTC_TOPIC["LF_SPORT_MOD_STATE"], self._pose_callback
        )

        # 5. Subscribe to LIDAR point cloud messages
        self._conn.datachannel.pub_sub.subscribe(
            RTC_TOPIC["ULIDAR_ARRAY"], self._lidar_callback
        )

    # ------------------------------------------------------------------
    # Callbacks (run in the async event loop context)
    # ------------------------------------------------------------------

    def _pose_callback(self, message: dict):
        """Parse position & orientation from LF_SPORT_MOD_STATE."""
        try:
            data = message.get("data", {})
            position = data.get("position", [0.0, 0.0, 0.0])
            imu_state = data.get("imu_state", {})
            rpy = imu_state.get("rpy", [0.0, 0.0, 0.0])

            with self._pose_lock:
                self.x, self.y = float(position[0]), float(position[1])
                self.theta = float(rpy[2])  # yaw
        except Exception:
            # Silently ignore malformed messages
            pass


    def voxel_positions_to_scan(
            self,
            positions,
            width=(128, 128, 38),        # Unitree's default grid size
            resolution=0.05,             # metres per voxel
            angle_res_deg=1.0,           # desired angular resolution
            z_min_m: float = -0.20,      # metres from lidar plane
            z_max_m: float = 0.05,       # metres from lidar plane
            yaw_rad: float | None = None  # robot yaw (world → robot frame)
    ):
        """
        Convert Unitree voxel map 'positions' into a planar laser-scan.

        Parameters
        ----------
        positions : list[int]
            Flat [x0, y0, z0, x1, y1, z1, …] list from the WebRTC packet.
        width : tuple[int,int,int], optional
            Grid dimensions (Nx, Ny, Nz).  Defaults to 128×128×38.
        resolution : float, optional
            Edge length of one voxel in metres.  Defaults to 0.05 m.
        angle_res_deg : float, optional
            Angular bucket size of the resulting scan (1 deg → 360 bins).
        z_min_m : float, optional
            Minimum z value (metres) for a point to be included.
        z_max_m : float, optional
            Maximum z value (metres) for a point to be included.
        yaw_rad : float | None, optional
            Current robot yaw (heading) in radians.  If provided, the returned
            scan is rotated so that 0 ° corresponds to the robot's +x axis.

        Returns
        -------
        scan : np.ndarray
            Array of length 360/angle_res_deg; `scan[i]` is the minimum range
            in metres in the bucket whose centre is `i*angle_res_deg` degrees
            **in the robot frame**.  Cells with no returns contain max_range.
        """

        # 1. reshape positions → (N,3) int array
        pts_idx = np.asarray(positions, dtype=np.int32).reshape(-1, 3)

        # 2. convert to lidar-centred metric coordinates
        half_w = np.asarray(width, dtype=np.float32) / 2.0
        pts_rel = (pts_idx - half_w) * float(resolution)   # (N,3)  metres

        # 3. keep points close to the horizontal plane
        z_coords = pts_rel[:, 2]
        mask = (z_coords >= z_min_m) & (z_coords <= z_max_m)
        horiz  = pts_rel[mask, :2]                         # (M,2)  x,y only
        if horiz.size == 0:
            nbins = int(round(360 / angle_res_deg))
            return np.full(nbins, MAX_RANGE, dtype=np.float32)

        # 4. polar coordinates
        dx, dy   = horiz[:, 0], horiz[:, 1]
        ranges   = np.hypot(dx, dy)                        # distance
        bearings = (np.degrees(np.arctan2(dy, dx)) + 360) % 360

        # ------------------------------------------------------------------
        # Rotate scan into robot frame if yaw is provided
        # ------------------------------------------------------------------
        if yaw_rad is not None:
            # Subtract yaw so that when the robot rotates CCW by +yaw the scan
            # rotates accordingly (bearings decrease).
            bearings = (bearings - np.degrees(yaw_rad)) % 360

        # 5. bucket by bearing and keep the nearest hit per bucket
        nbins  = int(round(360 / angle_res_deg))
        scan   = np.full(nbins, MAX_RANGE, dtype=np.float32)

        bins = (bearings / angle_res_deg).astype(int)
        for b, r in zip(bins, ranges):
            if r < scan[b]:
                scan[b] = r

        return scan


    def _lidar_callback(self, message: dict):
        """Convert point cloud to 2-D laser scan and store the most recent one."""
        try:
            # Retrieve current robot yaw (protected by lock)
            with self._pose_lock:
                yaw = float(self.theta)

            # Convert flat list [x1, y1, z1, x2, y2, z2, ...] → scan in robot frame
            scan = self.voxel_positions_to_scan(
                message.get("data", {}).get("data", {}).get("positions", []),
                yaw_rad=yaw
            )

            # Get additional LIDAR metadata
            resolution = message.get("data", {}).get("resolution")
            src_size = message.get("data", {}).get("src_size") 
            origin = message.get("data", {}).get("origin")
            width = message.get("data", {}).get("width")
            
            print(f"LIDAR metadata: resolution={resolution}, src_size={src_size}, origin={origin}, width={width}")

            # Atomically update buffers
            with self._scan_lock:
                self._latest_scan = scan
                self._scans.append(scan)
                if len(self._scans) > self._MAX_SCAN_HISTORY:
                    self._scans.pop(0)
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Shutdown utilities
    # ------------------------------------------------------------------

    async def _async_close(self):
        if hasattr(self, "_conn") and self._conn:
            await self._conn.disconnect()

    def close(self):
        """Synchronously close WebRTC connection and stop background loop."""
        try:
            fut = asyncio.run_coroutine_threadsafe(self._async_close(), self._loop)
            fut.result(timeout=5)
        except Exception:
            pass
        finally:
            if self._loop.is_running():
                self._loop.call_soon_threadsafe(self._loop.stop)
            if self._loop_thread.is_alive():
                self._loop_thread.join(timeout=1)


def generate_robot_frame_map(map_size_m: float,
                             resolution: float,
                             angles: np.ndarray,
                             ranges: np.ndarray,
                             max_range: float = 8.0,
                             inflate: bool = False,
                             inflation_radius: float = 0.2):

    cells = int(map_size_m / resolution)
    grid = np.zeros((cells, cells), dtype=np.uint8)

    # Robot is at the centre of the grid
    cx = cy = cells // 2

    valid = (ranges < max_range) & np.isfinite(ranges)
    r = ranges[valid]
    a = angles[valid]

    # Cartesian end-points in robot frame
    end_x = r * np.cos(a)
    end_y = r * np.sin(a)

    xi = (end_x / resolution).astype(int) + cx
    yi = (end_y / resolution).astype(int) + cy

    inside = (xi >= 0) & (xi < cells) & (yi >= 0) & (yi < cells)
    grid[yi[inside], xi[inside]] = 1

    # Optional obstacle inflation
    # if inflate and distance_transform_edt is not None:
    #     d = distance_transform_edt(grid == 0) * resolution
    #     grid[d < inflation_radius] = 1

    return grid

# ------------------------------------------------------------------
# World-frame map generation
# ------------------------------------------------------------------


def generate_world_frame_map(
    map_size_m: float,
    resolution: float,
    robot_pose: Tuple[float, float, float],
    angles: np.ndarray,
    ranges: np.ndarray,
    max_range: float = 8.0,
    inflate: bool = False,
    inflation_radius: float = 0.2,
):
   
    # Grid allocation
    cells = int(map_size_m / resolution)
    grid = np.zeros((cells, cells), dtype=np.uint8)

    # --- Pose unpacking ----------------------------------------------------
    rx, ry, rtheta = robot_pose  # world-frame position & yaw

    # Origin (world coordinates) of the bottom-left corner of the grid so that
    # the robot is located at the grid centre just as in the robot-frame map.
    origin_x = rx - (map_size_m / 2.0)
    origin_y = ry - (map_size_m / 2.0)

    # --- Filter and transform laser returns -------------------------------
    valid = (ranges < max_range) & np.isfinite(ranges)
    if not np.any(valid):
        return grid  # nothing to mark

    r = ranges[valid]
    a = angles[valid]

    # End-points in WORLD frame
    world_x = rx + r * np.cos(a + rtheta)
    world_y = ry + r * np.sin(a + rtheta)

    # Grid indices (integer)
    xi = ((world_x - origin_x) / resolution).astype(int)
    yi = ((world_y - origin_y) / resolution).astype(int)

    inside = (xi >= 0) & (xi < cells) & (yi >= 0) & (yi < cells)
    grid[yi[inside], xi[inside]] = 1

    # Optional obstacle inflation (requires SciPy)
    # if inflate and distance_transform_edt is not None:
    #     d = distance_transform_edt(grid == 0) * resolution
    #     grid[d < inflation_radius] = 1

    return grid


def live_robot_frame_visualization(
    robot: "Robot",
    map_size_m: float = 30.0,
    resolution: float = 0.1,
    max_range: float = 8.0,
    plot_interval: float = 0.2,
    world_frame: bool = True
):


    if plt is None:
        raise ImportError("matplotlib is required for live visualization → install matplotlib")

    # Initial map from the current scan so that we have something to show
    angles, ranges = robot.scan_polar
    if world_frame:
        grid = generate_world_frame_map(map_size_m, resolution, robot.pose, angles, ranges, max_range)
    else:
        grid = generate_robot_frame_map(map_size_m, resolution, angles, ranges, max_range)
    # Figure setup
    plt.ion()
    fig, ax = plt.subplots()
    im = ax.imshow(grid, origin="lower", cmap="Greys", vmin=0, vmax=1)

    # Draw an arrow representing the robot's +x (forward) axis.
    # In world-frame mode we rotate the arrow by the robot's yaw so that it
    # points in the correct absolute direction.
    cells = grid.shape[0]
    centre = cells / 2.0
    arrow_len_cells = 0.5 / resolution  # 0.5 m arrow length

    def _draw_direction_arrow(yaw_rad: float):
        """Helper: draw arrow originating at grid centre with given yaw."""
        dx = np.cos(yaw_rad) * arrow_len_cells
        dy = np.sin(yaw_rad) * arrow_len_cells
        return ax.arrow(
            centre,
            centre,
            dx,
            dy,
            head_width=arrow_len_cells * 0.3,
            head_length=arrow_len_cells * 0.3,
            fc="red",
            ec="red",
        )

    initial_yaw = robot.pose[2] if world_frame else 0.0
    arrow = _draw_direction_arrow(initial_yaw)

    ax.set_title("Robot-frame occupancy grid (live)")

    fig.canvas.draw_idle()
    plt.show(block=False)

    try:
        while plt.fignum_exists(fig.number):
            # Retrieve the latest laser scan
            angles, ranges = robot.scan_polar
            if world_frame:
                grid = generate_world_frame_map(map_size_m, resolution, robot.pose, angles, ranges, max_range)
            else:
                grid = generate_robot_frame_map(map_size_m, resolution, angles, ranges, max_range)

            # Update plot data
            im.set_data(grid)

            # Update arrow to reflect current orientation in world frame
            if world_frame:
                try:
                    # Remove previous arrow
                    arrow.remove()
                except Exception:
                    pass

                # Draw new arrow with updated yaw
                arrow = _draw_direction_arrow(robot.pose[2])

            # Force a redraw and wait a bit
            fig.canvas.draw_idle()
            plt.pause(plot_interval)
    except KeyboardInterrupt:
        robot.close()
        # Gracefully exit on Ctrl+C
        pass
    finally:
        plt.ioff()
        plt.close(fig)

if __name__ == "__main__":
    """Simple manual test: move gently forward and print state."""
    r = Robot(ip="172.30.1.80")
    live_robot_frame_visualization(r)

    try:
        while True:
            # Send a gentle forward command (0.05 m/s)
            r.step(0.0, 0.1, 0.0, dt=0.1)
            pose = r.pose
            scan = r.lasers
            num_valid = int(np.isfinite(scan).sum())

            time.sleep(0.2)
    except KeyboardInterrupt:
        print("\nUser interrupted, exiting.")
        r.close()
   