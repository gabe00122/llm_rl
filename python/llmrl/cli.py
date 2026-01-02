from typing_extensions import NamedTuple
from llmrl.env.make import make_env
from llmrl.config import load_config
from llmrl.experiement import Experiment
import numpy as np

class Test(NamedTuple):
    max_x: int
    max_y: int


def main():
    # experiment = Experiment.from_config_file("configs/test.json", create_directories=False)
    # print(experiment.config)
    env = make_env("arithmetic", 1, 0, Test(max_x=10, max_y=10))
    batch_indices = np.array([0], dtype=np.int32)
    
    print(env.instructions())
    obs = env.reset(batch_indices)

    while True:
        print(obs[0])
        action = input("Action: ")
        obs, reward, done = env.step(batch_indices, [action])
        print(f"Reward: {reward[0]}")


if __name__ == "__main__":
    main()
