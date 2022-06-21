import gym

env = gym.make('Humanoid-v2')

print(env.action_space)

env.reset()

for _ in range(10000):
    env.render()
    env.step(env.action_space.sample())

env.close()