import gymnasium as gym
from gymnasium import spaces
import numpy as np
import time
from coppeliasim_zmqremoteapi_client import RemoteAPIClient
from ultralytics import YOLO
import cv2

class DroneTrackingEnv(gym.Env):
    def __init__(self, model_path="best.pt", dt=0.02):
        super().__init__()

        # -------------------------------
        # Connect to CoppeliaSim
        client = RemoteAPIClient()
        self.sim = client.getObject('sim')

        # -------------------------------
        # Niryo One base handle (usually 15 in scene)
        self.base_handle = 15

        # -------------------------------
        # Fetch all 6 joints
        JOINT_TYPE = 1  # revolute joints
        all_joints = self.sim.getObjectsInTree(self.base_handle, JOINT_TYPE, 1)
        if len(all_joints) < 6:
            raise RuntimeError(f"Expected 6 joints, found {len(all_joints)}")
        self.joint_handles = all_joints[:6]  # take first 6 joints for full control

        # -------------------------------
        # Vision sensor
        self.camera_handle = self.sim.getObject('/NiryoOne/visionSensor')
        if self.camera_handle < 0:
            raise RuntimeError("Vision sensor not found!")

        # -------------------------------
        # Drone handle
        self.drone = self.sim.getObject('/target')

        # -------------------------------
        # Load YOLO model
        self.model = YOLO(model_path)
        self.model.to("cuda" if torch.cuda.is_available() else "cpu")

        # -------------------------------
        # Step settings
        self.dt = dt
        self.max_steps = 400
        self.current_step = 0
        self.speed_scale = 0.05  # small increments for smooth motion

        # -------------------------------
        # Spaces
        self.action_space = spaces.Box(
            low=-self.speed_scale,
            high=self.speed_scale,
            shape=(6,),  # all 6 joints
            dtype=np.float32
        )

        # Observation: dx, dy, area_norm, aspect_ratio, visible_flag + 6 joint positions
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(5 + 6,), dtype=np.float32
        )

        # YOLO cache
        self.last_bbox = None

        # Enable synchronous stepping
        self.sim.setStepping(True)

    # -------------------------------
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0

        # Reset drone
        self.sim.setObjectPosition(self.drone, -1, [0, 0, 0.5])

        # Reset joints
        for joint in self.joint_handles:
            self.sim.setJointTargetPosition(joint, 0.0)

        time.sleep(self.dt)
        obs = self._get_obs()
        return obs, {}

    # -------------------------------
    def step(self, action):
        self.current_step += 1
        action = np.clip(action, -self.speed_scale, self.speed_scale)

        # Apply action smoothly
        for idx, joint in enumerate(self.joint_handles):
            current_pos = self.sim.getJointPosition(joint)
            self.sim.setJointTargetPosition(joint, current_pos + action[idx])

        # Let simulation catch up
        time.sleep(self.dt)
        self.sim.step()

        # Observation
        obs = self._get_obs()
        dx, dy, area, aspect, visible = obs[:5]

        # Reward: smaller error = higher reward
        reward = - (dx**2 + dy**2)

        # Bonus for being well-centered
        if abs(dx) < 0.05 and abs(dy) < 0.05 and visible > 0.5:
            reward += 2.0

        # Penalize loss of visibility
        if visible < 0.5:
            reward -= 5.0

        # Small penalty for big actions
        reward -= 0.001 * np.sum(np.square(action))

        # Termination conditions
        terminated = False
        if visible < 0.5 or self.current_step >= self.max_steps:
            terminated = True
        truncated = False

        return obs, reward, terminated, truncated, {}

    # -------------------------------
    def _get_obs(self):
        # Get vision sensor image
        img, resolution = self.sim.getVisionSensorImg(self.camera_handle)
        resX, resY = resolution
        frame = np.frombuffer(img, dtype=np.uint8).reshape((resY, resX, 3))

        # YOLO detection
        results = self.model(frame, verbose=False)
        boxes = results[0].boxes

        dx, dy, area_norm, aspect_ratio, visible_flag = 0, 0, 0, 0, 0
        if len(boxes) > 0:
            areas = (boxes.xyxy[:,2]-boxes.xyxy[:,0])*(boxes.xyxy[:,3]-boxes.xyxy[:,1])
            idx = areas.argmax()
            x1, y1, x2, y2 = boxes.xyxy[idx].cpu().numpy()
            self.last_bbox = (x1, y1, x2, y2)

            # Center offsets
            cx, cy = (x1 + x2)/2, (y1 + y2)/2
            fx, fy = resX/2, resY/2
            dx = (cx - fx)/resX
            dy = (cy - fy)/resY

            # Normalized area
            area_norm = (areas[idx]/(resX*resY)).item()

            # Aspect ratio
            aspect_ratio = ((x2-x1)/(y2-y1+1e-6))

            visible_flag = 1.0
        else:
            self.last_bbox = None

        # Joint positions
        joint_positions = [self.sim.getJointPosition(j) for j in self.joint_handles]

        return np.array([dx, dy, area_norm, aspect_ratio, visible_flag] + joint_positions, dtype=np.float32)

    # -------------------------------
    def visualize_step(self):
        img, resolution = self.sim.getVisionSensorImg(self.camera_handle)
        resX, resY = resolution
        frame = np.frombuffer(img, dtype=np.uint8).reshape((resY, resX, 3))
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        if self.last_bbox is not None:
            x1, y1, x2, y2 = map(int, self.last_bbox)
            cv2.rectangle(frame, (x1,y1),(x2,y2),(0,255,0),2)
            cx, cy = (x1+x2)//2, (y1+y2)//2
            cv2.circle(frame, (cx,cy), 5, (0,0,255), -1)

        # Center target
        fx, fy = resX//2, resY//2
        cv2.circle(frame, (fx, fy), 5, (255,0,0), -1)

        cv2.imshow("Drone Tracking", frame)
        cv2.waitKey(1)
