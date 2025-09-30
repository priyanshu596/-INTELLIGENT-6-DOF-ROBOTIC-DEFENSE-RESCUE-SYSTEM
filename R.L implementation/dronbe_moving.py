from coppeliasim_zmqremoteapi_client import RemoteAPIClient
import math
import time
import random

# -------------------------------
# Connect to CoppeliaSim
client = RemoteAPIClient()
sim = client.getObject('sim')

# -------------------------------
# Drone handle
drone = sim.getObject('/target')  # keep as '/target' if that’s the name

# -------------------------------
# Workspace bounds [xmin, xmax], [ymin, ymax], [zmin, zmax]
bounds = [-0.6, 0.6, -0.6, 0.6, 0.3, 0.9]

# Starting position
pos = sim.getObjectPosition(drone, -1)

# Pick first waypoint
def random_waypoint():
    return [
        random.uniform(bounds[0], bounds[1]),
        random.uniform(bounds[2], bounds[3]),
        random.uniform(bounds[4], bounds[5])
    ]

target = random_waypoint()
speed = random.uniform(0.1, 0.4)   # m/s
dt = 0.05  # 50ms update step

print("🚁 Drone random motion started...")

# Run indefinitely
while True:
    # Vector towards target
    direction = [target[i] - pos[i] for i in range(3)]
    dist = math.sqrt(sum(d*d for d in direction))

    if dist < 0.05:  # reached target → choose new one
        target = random_waypoint()
        speed = random.uniform(0.1, 0.4)
    else:
        direction = [d / dist for d in direction]  # normalize
        pos = [pos[i] + direction[i] * speed * dt for i in range(3)]
        sim.setObjectPosition(drone, -1, pos)

    time.sleep(dt)
