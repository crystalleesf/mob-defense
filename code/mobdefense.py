# Rllib docs: https://docs.ray.io/en/latest/rllib.html
# Malmo XML docs: https://docs.ray.io/en/latest/rllib.html

########### TO DO ###########
# reward for agent health   #
#############################

try:
    from malmo import MalmoPython
except:
    import MalmoPython

import sys
import time
import json
import numpy as np
from numpy.random import randint
from past.utils import old_div
import gym, ray
from gym.spaces import Discrete, Box
from ray.rllib.agents import ppo
import random
import math
import matplotlib.pyplot as plt
# Task parameters:
ARENA_WIDTH = 100
ARENA_BREADTH = 100

class MobDefense(gym.Env):

    def __init__(self, env_config):  
        # Static Parameters
        self.size = 50
        self.obs_size = 5
        self.max_episode_steps = 100
        self.log_frequency = 1
      
        self.episode_num = 0
        self.total_reward_arr = []

        # Rllib Parameters
        self.action_space = Box(low= np.array([-1.0,-1.0,-1.0]), high=np.array([1.0,1.0,1.0]), dtype=np.float32)
        self.observation_space = Box(0, 1, shape=(2 * self.obs_size * self.obs_size, ), dtype=np.float32)

        # Malmo Parameters
        self.agent_host = MalmoPython.AgentHost()
        try:
            self.agent_host.parse( sys.argv )
        except RuntimeError as e:
            print('ERROR:', e)
            print(self.agent_host.getUsage())
            exit(1)

        # MobDefense Parameters
        self.obs = None
        self.allow_break_action = False
        self.episode_return = 0
        self.returns = []
        self.steps = []

    def reset(self):
        """
        Resets the environment for the next episode.

        Returns
            observation: <np.array> flattened initial obseravtion
        """
        # Reset Malmo
        world_state = self.init_malmo()

        # Reset Variables
        self.returns.append(self.episode_return)
        self.episode_return = 0

        # Get Observation
        self.obs, self.allow_break_action = self.get_observation(world_state)

        return self.obs

    def step(self, action):
        # Get Observation
        world_state = self.agent_host.getWorldState()
        for error in world_state.errors:
            print("Error:", error.text)
        self.obs, self.allow_break_action = self.get_observation(world_state) 

        # Get Done
        done = not world_state.is_mission_running 

        # Get Reward
        reward = 0
        for r in world_state.rewards:
            reward += r.getValue()
        self.episode_return += reward

        return self.obs, reward, done, dict()

    def get_mission_xml(self):
        # Draw Arena
        arena_walls = ""
        for x in range(-10,11):
            arena_walls += "<DrawBlock x='{}' y='2' z='{}' type='grass' />".format(x, 10)
            arena_walls += "<DrawBlock x='{}' y='3' z='{}' type='grass' />".format(x, 10)
            arena_walls += "<DrawBlock x='{}' y='4' z='{}' type='grass' />".format(x, 10)
            arena_walls += "<DrawBlock x='{}' y='2' z='{}' type='grass' />".format(x, -10)
            arena_walls += "<DrawBlock x='{}' y='3' z='{}' type='grass' />".format(x, -10)
            arena_walls += "<DrawBlock x='{}' y='4' z='{}' type='grass' />".format(x, -10)
        
        glowstone_wall = ""
        glowstone_wall += "<DrawBlock x='{}' y='3' z='{}' type='glowstone' />".format(0, 10)
        glowstone_wall += "<DrawBlock x='{}' y='3' z='{}' type='glowstone' />".format(0, -10)
        glowstone_wall += "<DrawBlock x='{}' y='3' z='{}' type='glowstone' />".format(-10, 5)
        glowstone_wall += "<DrawBlock x='{}' y='3' z='{}' type='glowstone' />".format(10, 5)
            
        for z in range(-10,11):
            arena_walls += "<DrawBlock x='{}' y='2' z='{}' type='grass' />".format(-10, z)
            arena_walls += "<DrawBlock x='{}' y='3' z='{}' type='grass' />".format(-10, z)
            arena_walls += "<DrawBlock x='{}' y='4' z='{}' type='grass' />".format(-10, z)   
            arena_walls += "<DrawBlock x='{}' y='2' z='{}' type='grass' />".format(10, z) 
            arena_walls += "<DrawBlock x='{}' y='3' z='{}' type='grass' />".format(10, z)
            arena_walls += "<DrawBlock x='{}' y='4' z='{}' type='grass' />".format(10, z)
        
        # Draw Zombie
        hostile_mob_xml = "<DrawEntity x='{}' y='2' z='{}' type='Zombie' />".format(0, 5) 
        hostile_mob_xml += "<DrawEntity x='{}' y='2' z='{}' type='Zombie' />".format(-5, 2) 
        hostile_mob_xml += "<DrawEntity x='{}' y='2' z='{}' type='Zombie' />".format(3, 8) 

        return '''<?xml version="1.0" encoding="UTF-8" standalone="no" ?>
                <Mission xmlns="http://ProjectMalmo.microsoft.com" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">

                    <About>
                        <Summary>Mob Defense</Summary>
                    </About>

                    <ServerSection>
                        <ServerInitialConditions>
                            <Time>
                                <StartTime>14000</StartTime>
                                <AllowPassageOfTime>false</AllowPassageOfTime>
                            </Time>
                            <Weather>clear</Weather>
                        </ServerInitialConditions>
                        <ServerHandlers>
                            <FlatWorldGenerator generatorString="3;7,2;1;"/>
                            <DrawingDecorator>''' + \
                                "<DrawCuboid x1='{}' x2='{}' y1='2' y2='2' z1='{}' z2='{}' type='air'/>".format(-self.size, self.size, -self.size, self.size) + \
                                "<DrawCuboid x1='{}' x2='{}' y1='1' y2='1' z1='{}' z2='{}' type='stone'/>".format(-self.size, self.size, -self.size, self.size) + \
                                arena_walls + \
                                glowstone_wall + \
                                hostile_mob_xml + \
                                '''<DrawBlock x='0'  y='2' z='0' type='air' />
                                <DrawBlock x='0'  y='1' z='0' type='stone' />
                            </DrawingDecorator>
                            <ServerQuitWhenAnyAgentFinishes/>
                            <ServerQuitFromTimeUp timeLimitMs="120000"/>
                        </ServerHandlers>
                    </ServerSection>

                    <AgentSection mode="Survival">
                        <Name>CS175MobDefense</Name>
                        <AgentStart>
                            <Placement x="0" y="2" z="0" pitch="45" yaw="0"/>
                            <Inventory>
                                <InventoryItem slot="0" type="diamond_sword"/>
                            </Inventory>
                        </AgentStart>
                        <AgentHandlers>
                            <ChatCommands/>
                            <ContinuousMovementCommands/>
                            <ObservationFromFullStats/>
                            <ObservationFromRay/>
                            <RewardForDamagingEntity>
                                <Mob type="Zombie" reward="1"/>
                            </RewardForDamagingEntity>
                            <ObservationFromNearbyEntities>
                                <Range name="entities" xrange="'''+str(ARENA_WIDTH)+'''" yrange="2" zrange="'''+str(ARENA_BREADTH)+'''" />
                            </ObservationFromNearbyEntities>
                            <ObservationFromGrid>
                                <Grid name="floorAll">
                                    <min x="-'''+str(int(self.obs_size/2))+'''" y="-1" z="-'''+str(int(self.obs_size/2))+'''"/>
                                    <max x="'''+str(int(self.obs_size/2))+'''" y="0" z="'''+str(int(self.obs_size/2))+'''"/>
                                </Grid>
                            </ObservationFromGrid>
                            <AgentQuitFromTouchingBlockType>
                                <Block type="bedrock" />
                            </AgentQuitFromTouchingBlockType>
                        </AgentHandlers>
                    </AgentSection>
                </Mission>'''


    def init_malmo(self):
        """
        Initialize new malmo mission.
        """
        my_mission = MalmoPython.MissionSpec(self.get_mission_xml(), True)
        my_mission_record = MalmoPython.MissionRecordSpec()
        my_mission.requestVideo(800, 500)
        my_mission.setViewpoint(1)

        max_retries = 3
        my_clients = MalmoPython.ClientPool()
        my_clients.add(MalmoPython.ClientInfo('127.0.0.1', 10000)) # add Minecraft machines here as available

        for retry in range(max_retries):
            try:
                self.agent_host.startMission( my_mission, my_clients, my_mission_record, 0, 'MobDefense' )
                break
            except RuntimeError as e:
                if retry == max_retries - 1:
                    print("Error starting mission:", e)
                    exit(1)
                else:
                    time.sleep(2)

        world_state = self.agent_host.getWorldState()
        while not world_state.has_mission_begun:
            time.sleep(0.1)
            world_state = self.agent_host.getWorldState()
            # for error in world_state.errors:
            #     print("\nError:", error.text)

        # main loop:
        total_reward = 0
        Zombie_population = 0
        self_x = 0
        self_z = 0
        current_yaw = 0

        self.agent_host.sendCommand("chat /enchant CS175MobDefense minecraft:sharpness 5")
        
        while world_state.is_mission_running:
            world_state = self.agent_host.getWorldState()
            if world_state.number_of_observations_since_last_state > 0:
                msg = world_state.observations[-1].text
                ob = json.loads(msg)
                # Use the line-of-sight observation to determine when to hit and when not to hit:
                if u'LineOfSight' in ob:
                    los = ob[u'LineOfSight']
                    type=los["type"]
                    if type == "Zombie":
                        self.agent_host.sendCommand("attack 1")
                        time.sleep(.5)
                        self.agent_host.sendCommand("attack 0")
                # Get our position/orientation:
                if u'Yaw' in ob:
                    current_yaw = ob[u'Yaw']
                if u'XPos' in ob:
                    self_x = ob[u'XPos']
                if u'ZPos' in ob:
                    self_z = ob[u'ZPos']
                # Use the nearby-entities observation to decide which way to move, and to keep track
                # of population sizes - allows us some measure of "progress".
                if u'entities' in ob:
                    entities = ob["entities"]
                    num_Zombie = 0
                    x_pull = 0
                    z_pull = 0
                    for e in entities:
                        if e["name"] == "Zombie":
                            num_Zombie += 1
                            # Each Zombie contributes to the direction we should head in...
                            dist = max(0.0001, (e["x"] - self_x) * (e["x"] - self_x) + (e["z"] - self_z) * (e["z"] - self_z))
                            # Prioritise going after wounded Zombie. Max Zombie health is 20, according to Minecraft wiki...
                            weight = 21.0 - e["life"]
                            x_pull += weight * (e["x"] - self_x) / dist
                            z_pull += weight * (e["z"] - self_z) / dist
                    # Determine the direction we need to turn in order to head towards the "Zombieiest" point:
                    yaw = -180 * math.atan2(x_pull, z_pull) / math.pi
                    difference = yaw - current_yaw
                    while difference < -180:
                        difference += 360
                    while difference > 180:
                        difference -= 360
                    difference /= 180.0
                    self.agent_host.sendCommand("turn " + str(difference))
                    move_speed = 1.0 if abs(difference) < 0.5 else 0  # move slower when turning faster - helps with "orbiting" problem
                    self.agent_host.sendCommand("move " + str(move_speed))
                    if num_Zombie != Zombie_population:
                        # Print an update of our "progress":
                        Zombie_population = num_Zombie
                        tot = Zombie_population
                        if tot:
                            print("Zombie", end=' ')
                            r = old_div(40.0, tot)
                            print("Z:", num_Zombie)
                            
            if world_state.number_of_rewards_since_last_state > 0:
                # Keep track of our total reward:
                total_reward += world_state.rewards[-1].getValue()

        # mission has ended.
        for error in world_state.errors:
            print("Error:",error.text)
        if world_state.number_of_rewards_since_last_state > 0:
            # A reward signal has come in - see what it is:
            total_reward += world_state.rewards[-1].getValue()

        print()
        print("=" * 41)
        self.episode_num += 1
        self.total_reward_arr.append(total_reward)
        print("Reward Array: ", self.total_reward_arr)
        print("Episode number: ", self.episode_num)
        print("Total score this round:", total_reward)
        print("=" * 41)
        print()
        self.log_returns(self.episode_num, total_reward)
        time.sleep(1) # Give the mod a little time to prepare for the next mission.

        return world_state

    def get_observation(self, world_state):
        """
        Use the agent observation API to get a flattened 2 x 5 x 5 grid around the agent. 
        The agent is in the center square facing up.

        Args
            world_state: <object> current agent world state

        Returns
            observation: <np.array> the state observation
            allow_break_action: <bool> whether the agent is facing a diamond
        """
        obs = np.zeros((2 * self.obs_size * self.obs_size, ))
        allow_break_action = False

        while world_state.is_mission_running:
            time.sleep(0.1)
            world_state = self.agent_host.getWorldState()
            if len(world_state.errors) > 0:
                raise AssertionError('Could not load grid.')

            if world_state.number_of_observations_since_last_state > 0:
                # First we get the json from the observation API
                msg = world_state.observations[-1].text
                observations = json.loads(msg)

                # Get observation
                print(observations.keys())
                grid = observations['floorAll']
                for i, x in enumerate(grid):
                    obs[i] = x == 'Zombie'

                # Rotate observation with orientation of agent
                obs = obs.reshape((2, self.obs_size, self.obs_size))
                yaw = observations['Yaw']
                if yaw >= 225 and yaw < 315:
                    obs = np.rot90(obs, k=1, axes=(1, 2))
                elif yaw >= 315 or yaw < 45:
                    obs = np.rot90(obs, k=2, axes=(1, 2))
                elif yaw >= 45 and yaw < 135:
                    obs = np.rot90(obs, k=3, axes=(1, 2))
                obs = obs.flatten()

                allow_break_action = observations['LineOfSight']['type'] == 'Zombie'
                print(allow_break_action)
                break

        return obs, allow_break_action

    def log_returns(self, episode_num, total_reward):
        """
        Log the current returns as a graph and text file

        Args:
            steps (list): list of global steps after each episode
            returns (list): list of total return of each episode
        """
        plt.clf()
        plt.plot(range(1,episode_num + 1), self.total_reward_arr)
        plt.title('Mob Defense')
        plt.ylabel('Reward')
        plt.xlabel('Episode')
        plt.savefig('returns.png')

        with open('returns.txt', 'a') as f:
            f.write("{}\t{}\n".format(episode_num, total_reward)) 


if __name__ == '__main__':
    ray.init()
    trainer = ppo.PPOTrainer(env=MobDefense, config={
        'env_config': {},           # No environment parameters to configure
        'framework': 'torch',       # Use pyotrch instead of tensorflow
        'num_gpus': 0,              # We aren't using GPUs
        'num_workers': 0            # We aren't using parallelism
    })

    while True:
        print(trainer.train())