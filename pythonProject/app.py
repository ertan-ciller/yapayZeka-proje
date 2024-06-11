import numpy as np
import random

#4X4 Ortam Tanımı
class GridWorld:
    def __init__(self, size=(4, 4)): #Constructor varsayılan olarak 4x4 boyutuna bir ızgara oluşturur.
        self.size = size
        self.state = None #Farenin mevcut konumunu tutar.
        self.reset() #metot çağrılarak başlangıç durumu ayarlanır.

    def reset(self): #Farenin başlangıç durumunu (0,0) olarak ayarlar.
        self.state = (0, 0)
        return self.state

    def step(self, action): # Adım atma fonksiyonu diyebilirz.
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

        if self.state == (3, 3):  # Peynire Ulaştığında
            reward = 10
            done = True
        elif self.state in [(1, 1), (2, 2)]:  # Kediye yakalandığında
            reward = -10
            done = True

        return self.state, reward, done

    def render(self):
        grid = np.zeros(self.size)
        x, y = self.state
        grid[x, y] = 1
        print(grid)

class QLearningAgent: #Q-learning algoritmasını kullanan ajanımız - fare
    def __init__(self, state_space, action_space, alpha=0.1, gamma=0.99, epsilon=1.0, epsilon_decay=0.995, min_epsilon=0.01):
        self.state_space = state_space
        self.action_space = action_space
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        self.q_table = np.zeros(state_space + (action_space,))

    def choose_action(self, state):
        if random.random() < self.epsilon: #Eğer rastgele bir sayı epsilon'dan küçükse, rastgele bir eylem seçilir (keşif yapar).
            return random.randint(0, self.action_space - 1)
        else: #Eğer rastgele sayı epsilon'dan büyükse, Q-tablosuna göre en iyi eylem seçilir (sömürme yapar).
            return np.argmax(self.q_table[state])

    def update_q_table(self, state, action, reward, next_state, done):
        best_next_action = np.argmax(self.q_table[next_state])
        td_target = reward + self.gamma * self.q_table[next_state][best_next_action] * (not done)
        td_error = td_target - self.q_table[state][action]
        self.q_table[state][action] += self.alpha * td_error

    def decay_epsilon(self): #Eğer epsilon minimum epsilon değerinden büyükse, epsilon_decay oranında azaltılır. Bu, ajan öğrenme süreci boyunca keşiften sömürüye geçiş yapmasını sağlar.
        if self.epsilon > self.min_epsilon:
            self.epsilon *= self.epsilon_decay

if __name__ == "__main__": #Programın çalışacağı main classı
    env = GridWorld() #enviroment in oluşturulabilmesi için gridWorld class ı  çağrılır.
    agent = QLearningAgent((4, 4), 4) #Durum uzayı (4x4) ve eylem uzayına (4 yön) sahip bir QLearningAgent ajanı oluşturur.

    episodes = 100
    for episode in range(episodes):  #Eğitim döngüsünde kaç epizod çalıştırılacağını belirtir (1000 epizod).
        state = env.reset() #Her epizod başında ortam sıfırlanır (state = env.reset()) ve toplam ödül (total_reward) sıfırlanır.
        total_reward = 0

        for time in range(100):
            action = agent.choose_action(state)
            next_state, reward, done = env.step(action)
            agent.update_q_table(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward

            if done:
                agent.decay_epsilon()
                print(f"Episode: {episode}/{episodes}, Score: {total_reward}, Epsilon: {agent.epsilon:.2f}")
                break

    # Final policy render
    state = env.reset()
    env.render()
    for _ in range(100):
        action = agent.choose_action(state)
        next_state, reward, done = env.step(action)
        env.render()
        state = next_state
        if done:
            break
    print(agent.q_table)
