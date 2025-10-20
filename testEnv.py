from monopolyEnv import MonopolyEnv

env = MonopolyEnv()
obs, info = env.reset()
for i in range(1000):
    action = env.action_space.sample()
    actions = {f"agent_{i+1}": action for i in range(4)}
    obs, reward, terminated, truncated, info = env.step(actions)
    if terminated["agent_1"]:
        print(terminated["agent_1"])
        obs, info = env.reset()
