---
layout: default
title:  Proposal
---

# Summary of Project
The goal is to train the AI to shoot any hostile mobs and prevent them from attacking the AI. Mobs in minecraft are textured differently, ranging from vibrant green to a dark hue brown. The AI will learn which mobs are hostile through their textures and use their bow to prevent them from coming close to them. Additionally, the AI will prioritize targets that are closer to them.  

# AI/ML Algorithm
We anticipate using computer vision and reinforcement learning to complete this project.  

# Evaluation Plan
We expect our AI to shoot a hostile mob 90+% of the time. Before our AI runs, we will feed the AI 50 images of each mob to help it recognize hostile and friendly mobs. During the run, the AI should prioritize shooting hostile mobs that are at a closer distance over hostile mobs that are farther away. We will use the distance formula $$d = \sqrt{(x_2 - x_1)^2 + (y_2 - y_1)^2}$$ to determine the distance between the AI character and surrounding mobs. 

To verify that the project works, we have 5 test scenarios in mind. We will start with 1 hostile mob (creepers) and 2 friendly mobs (cows, pigs). At the most basic level, the AI will face an environment that only contains static/non-mobile hostile mobs. The next level will build upon the previous level to include static friendly mobs. The third level will introduce randomness. In this level, the AI will face randomly spawned and randomly moving hostile mobs, while the friendly mobs remain static. In the final level, both the hostile mobs and friendly mobs will be spawning and moving randomly. Our moonshot case will introduce more types of mobs, specifically at least 2 types of hostile mobs and 4 types of friendly mobs.

# Appointment with Instructor
October 21, 2021 at 2:30 - 2:45 PM
