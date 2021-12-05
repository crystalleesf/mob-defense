---
layout: default
title: Final Report
---

## Video Summary


## Project Summary
<img src="./images/mob-defense-screenshot-2.png" alt="Mob Defense Screenshot" width="500"/>

The goal of Mob Defense is to train the agent to efficiently and effectively kill Zombies. We've placed the agent in a 20x20 arena with 4 Zombies and 50 sheep. The agent is equipped with an enchanted diamond sword and night vision. We initially considered respawning Zombies after the agent kills a Zombie, however we felt that this may convolute our data. We chose to only spawn 4 Zombies per episode to maintain a consistent learning environment, which will ensure that the agent's ability to attack and survive is actually improving. We also placed 50 sheep in the arena as obstacles for the agent. As sheep are friendly mobs in Minecraft, we wanted the agent to learn to avoid attacking and killing the sheep. Each episode spans two minutes and concludes if time runs out or the agent gets killed by the Zombies. Its mission is to kill all 4 Zombies before time runs out while incurring minimal damage to health.  

This task requires AI/ML techniques to achieve because in order for the agent to survive it needs to make strategic decisions on whether it should run away from the Zombies (fight or flight), which Zombie to attack first, and the unintended damage it may cause to surrounding sheep. For example, if the agent exists in a state where a Zombie is surrounded by a lot of sheep, it may be more ideal for the agent to move away and wait for the sheep to disperse, instead of attacking the Zombie at that moment and possibly racking up negative reward for unintentionally damaging the surrounding sheep. 

## Approaches


## Evaluation


## References