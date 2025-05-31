# Street Fighter II Reinforcement Learning

This project demonstrates the application of Reinforcement Learning to the classic fighting game, Street Fighter II Special Champion Edition. Utilizing the Proximal Policy Optimization (PPO) algorithm, implemented through the Stable Baselines3 library, an AI agent was trained to learn how to play the game. The agent learns by interacting with the game environment, receiving rewards based on its actions, and iteratively refining its policy to maximize its performance.

[Watch the Gameplay Video](videos/StreetFighterII-1.mp4)

A significant challenge during development involved ensuring consistent and correct resetting of the observation space across different scenarios and game stages. This was crucial for stable learning and required careful handling of the environment's state. The resulting learned behavior of the PPO agent is visualized through recorded gameplay footage, showcasing its evolving ability to execute moves, react to opponents, and engage in combat.

Looking ahead, I aim to explore the capabilities of Reinforcement Learning in more complex environments. I'm particularly interested in environments that offer a richer set of agent actions and interactions, which could lead to the emergence of even more sophisticated and nuanced strategies.

---
