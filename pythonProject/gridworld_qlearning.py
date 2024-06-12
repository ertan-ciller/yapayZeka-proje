import numpy as np
import random
import matplotlib.pyplot as plt
import time

class GridWorld:
    def __init__(self, size=(4, 4)):
        self.size = size
        self.state = None
        self.reset()

    def reset(self):
        self.state = (0, 0)
        return self.state

    def step(self, action):
        x, y = self.state
        if action == 0:  # Up
            x = max(x - 1, 0)
        elif action == 1:  # Down
            x = min(x + 1, self.size[0] - 1)
        elif action == 2:  # Left
            y = max(y - 1, 0)
        elif action == 3:  # Right
            y = min(y + 1, self.size[1] - 1)

        self.state = (x, y)

        reward = 0
        done = False

        if self.state == (3, 3):  # Cheese
            reward = 1
            done = True
        elif self.state in [(1, 1), (2, 2)]:  # Cats
            reward = -1
            done = True

        return self.state, reward, done

    def render(self):
        grid = np.zeros(self.size)
        x, y = self.state
        grid[x, y] = 1
        print(grid)


class QLearningAgent:
    def __init__(self, state_space, action_space, alpha=0.1, gamma=0.99, epsilon=1.0, epsilon_decay=0.995, min_epsilon=0.01):
        self.state_space = state_space #durum uzayının boyutunu belirler.
        self.action_space = action_space #Eylem uzayının boyutunu belirler.
        self.alpha = alpha #Öğrenme oranı (learning rate), Q-değerlerinin güncellenme hızını kontrol eder.
        self.gamma = gamma # Gelecekteki ödüllerin mevcut değelere katkısını kontrol eder.
        self.epsilon = epsilon #başlangıç keşif oranı, epsilon-greedy stratejisi için kullanılır.
        self.epsilon_decay = epsilon_decay #Her adımda epsilonu azaltmak için kullanılan oran
        self.min_epsilon = min_epsilon #epsilonun alabileceği minimum değer
        self.q_table = np.zeros(state_space + (action_space,)) #Q değerlerini saklamak için kullanılan tabla, başlangıçta sıfırlarla doldurulur.

    def choose_action(self, state): #Eylem seçimi
        if random.random() < self.epsilon:
            return random.randint(0, self.action_space - 1) #Eğer rastgele bir sayı epsilonmdan küçükse, rastgele bir eylem(keşif yapar).
        else:
            return np.argmax(self.q_table[state]) #Eğer rastgele sayı epsilon'dan büyükse, Q-tablosuna göre en iyi eylem seçilir (sömürme yapar).

    def update_q_table(self, state, action, reward, next_state, done): #Q-Tablosunu Güncelleme:
        best_next_action = np.argmax(self.q_table[next_state])
        td_target = reward + self.gamma * self.q_table[next_state][best_next_action] * (not done)
        td_error = td_target - self.q_table[state][action]
        self.q_table[state][action] += self.alpha * td_error

    def decay_epsilon(self):
        if self.epsilon > self.min_epsilon:
            self.epsilon *= self.epsilon_decay

def simulate_agent(agent, env, episodes=1000):
    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        steps = 0

        while True:
            env.render()
            action = agent.choose_action(state)
            next_state, reward, done = env.step(action)
            state = next_state
            total_reward += reward
            steps += 1

            # Visualize the grid
            visualize_grid(env, total_reward, steps)

            if done:
                print(f"Episode: {episode+1}, Score: {total_reward}")
                break

def visualize_grid(env, total_reward, steps):
    grid = np.zeros(env.size)
    x, y = env.state
    grid[x, y] = 1

    fig, ax = plt.subplots()
    ax.imshow(grid, cmap='Greys')

    for i in range(env.size[0]):
        for j in range(env.size[1]):
            text = ax.text(j, i, 'M' if grid[i, j] == 1 else '',
                           ha="center", va="center", color="red" if grid[i, j] == 1 else "black")

    plt.title(f'Total Reward: {total_reward} Steps: {steps}')
    plt.show(block=False)
    plt.pause(0.3)
    input("Press Enter to continue...")  # Kullanıcıdan giriş bekleyin
    plt.close()


if __name__ == "__main__":
    env = GridWorld()
    agent = QLearningAgent((4, 4), 4)

    # Train the agent
    episodes = 5
    for episode in range(episodes):
        state = env.reset()
        total_reward = 0

        for time in range(100):
            action = agent.choose_action(state)
            next_state, reward, done = env.step(action)
            agent.update_q_table(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward

            if done:
                agent.decay_epsilon()
                print(f"Episode: {episode + 1}/{episodes}, Score: {total_reward}, Epsilon: {agent.epsilon:.2f}")
                break

        # Simulate the trained agent
        simulate_agent(agent, env)