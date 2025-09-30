import time
import numpy as np
import cv2
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from drone_tracking_env import DroneTrackingEnv  # your fixed env

# ------------------------
# TQDM Callback for progress bar
# ------------------------
from tqdm import tqdm

class TQDMCallback(BaseCallback):
    def __init__(self, total_timesteps, verbose=0):
        super().__init__(verbose)
        self.total_timesteps = total_timesteps
        self.pbar = tqdm(total=total_timesteps, desc="Training PPO", ncols=100, colour="green")

    def _on_step(self) -> bool:
        self.pbar.update(1)
        # ✅ Stop training exactly at requested timesteps
        if self.num_timesteps >= self.total_timesteps:
            return False
        return True

    def _on_training_end(self) -> None:
        self.pbar.close()

# ------------------------
# Standalone visualization function
# ------------------------
def visualize_step(env, obs, bbox=None):
    img, resolution = env.sim.getVisionSensorImg(env.camera_handle)
    resX, resY = resolution

    if isinstance(img, bytes):
        frame = np.frombuffer(img, dtype=np.uint8)
    else:
        frame = np.array(img, dtype=np.uint8)
    frame = frame.reshape((resY, resX, 3))
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    if bbox is not None:
        x1, y1, x2, y2 = map(int, bbox)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cx, cy = (x1 + x2)//2, (y1 + y2)//2
        cv2.circle(frame, (cx, cy), 5, (0,0,255), -1)
    else:
        fx, fy = resX//2, resY//2
        dx = int(obs[0]*resX)
        dy = int(obs[1]*resY)
        cx, cy = fx + dx, fy + dy
        cv2.circle(frame, (cx, cy), 5, (0,0,255), -1)

    fx, fy = resX//2, resY//2
    cv2.circle(frame, (fx, fy), 5, (255,0,0), -1)
    cv2.line(frame, (fx, fy), (cx, cy), (255,255,0), 1)

    cv2.imshow("RL Drone Training", frame)
    cv2.waitKey(1)

# ------------------------
# Initialize environment
# ------------------------
env = DroneTrackingEnv("best.pt")

# ------------------------
# Initialize PPO
# ------------------------
model = PPO(
    "MlpPolicy",
    env,
    verbose=1,
    n_steps=512,
    batch_size=64,
    learning_rate=3e-4,
    tensorboard_log="./ppo_logs",
    device="cpu"
)

# ------------------------
# Train PPO with TQDM progress bar
# ------------------------
total_timesteps = 10000
callback = TQDMCallback(total_timesteps=total_timesteps)
model.learn(total_timesteps=total_timesteps, callback=callback)

# ------------------------
# ✅ Save the trained model immediately
# ------------------------
model.save("drone_tracker_rl_second")
print("✅ Training complete. Model saved as 'drone_tracker_rl'.")

# ------------------------
# Optional: run trained policy with visualization
# ------------------------
obs, _ = env.reset()
while True:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    visualize_step(env, obs, bbox=env.last_bbox)
    if terminated or truncated:
        obs, _ = env.reset()

cv2.destroyAllWindows()

