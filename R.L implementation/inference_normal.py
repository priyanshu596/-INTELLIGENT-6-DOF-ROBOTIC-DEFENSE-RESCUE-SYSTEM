import time
import cv2
from stable_baselines3 import PPO
from drone_tracking_env import DroneTrackingEnv   # make sure path is correct

if __name__ == "__main__":
    # ✅ load trained PPO
    model = PPO.load("drone_tracker_rl_second")

    # ✅ inference environment (slow + with visualization)
    env = DroneTrackingEnv(model_path="best.pt")

    obs, _ = env.reset()
    done, truncated = False, False

    while True:
        # get action from trained model
        action, _ = model.predict(obs, deterministic=True)

        # step in environment
        obs, reward, done, truncated, _ = env.step(action)

        # visualize
        env.visualize_step(obs)

        if done or truncated:
            obs, _ = env.reset()
            time.sleep(0.5)   # short pause before restarting
