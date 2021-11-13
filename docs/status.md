---
layout: default
title: Status
---

## Project Summary

Our goal for the project has somewhat changed from our initial proposal. Our original idea was to train the agent to differentiate between hostile mobs (e.g., zombies) and friendly mobs (e.g., pigs) by positively rewarding it for killing hostile mobs and negatively rewarding it for killing friendly mobs. This turned out to be more challenging than expected, so we've decided to focus on training the agent to damage and kill a fixed number of zombies in a closed arena. The optimal goal for the agent is to kill all of the zombies in the arena without incurring damage to health or dying. 

## Approach

The actions of the agent include turning, moving, and attacking with the sword. We used reinforcement learning to train our agent. We reward the agent (+1) whenever it damages a zombie. We are actively trying to add a negative reward (-1) whenever the agent incurs damage from the zombies. In our code, the agent would attack whenever a zombie enters its line of vision and detect where the zombies are and moves and turns towards them.

## Evaluation
### Qualitative

### Quantitative

## Remaining Goals and Challenges

We still need to figure out how to add a negative reward for when the agent incurs damage. Next, we're also planning to add a friendly mob that the agent SHOULD NOT attack. Lastly, we need to respawn zombies after they are killed. 

## Resources Used
- [Malmo XML Schema Documentation](https://microsoft.github.io/malmo/0.21.0/Schemas/MissionHandlers.html)
- [Minecraft Enchantment Command](https://www.digminecraft.com/game_commands/enchant_command.php)
- hit_test.py from Malmo's Python Examples