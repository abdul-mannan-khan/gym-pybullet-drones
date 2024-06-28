from gym_pybullet_drones.envs.FlyThruObstaclesAviary import FlyThruObstaclesAviary

try:
    env = FlyThruObstaclesAviary()
    print("Environment instantiated successfully.")
except TypeError as e:
    print("TypeError during instantiation:", e)
except Exception as e:
    print("Other error during instantiation:", e)
