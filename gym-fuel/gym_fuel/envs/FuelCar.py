
import math
from typing import Optional

import numpy as np

from random import randrange
import gym
from gym import spaces
from gym.utils import seeding
from random import randrange


class FuelCarEnv(gym.Env):
    """
    Description:
        The agent (a car) is started at the start of a circuit. For any given
        state the agent may choose to accelerate,  don't accelerate, deaccelerate, up a gear or
        down a gear
    Source:
        The environment is proposed by Hugo Jiménez
    Observation:
        Type: MultiDiscrete(6)
        Num    Observation               Min            Max
        0      Car Position              0              10.0000
        1      Car speed                 0              180
        2      Car Gear                  1              6
        3      Car Revolutions           0              60000
        4      Consume                   0              30
        5      Circuit Limit             40             150
    Actions:
        Type: Discrete(5)
        Num    Action
        0      Deccelerate
        1      Don't accelerate
        2      Accelerate
        3      Down gear
        4      Up gear

    Reward:
         Reward of speed is awarded in each step.
         Reward of -1 is awarded if the position of the agent is less than 0.5.
    Starting State:
         The position of the car is fixed [0, 0].
         The starting speed of the car is always assigned to 0.
         The starting gear is 1
         The starting car revolutions are 0.
         The limit speed is random between 50 and 150
    Episode Termination:
         The car arrives at the end of the circuit.
         The episode length is higher than 40000
    """

    def __init__(self):
        self.min_position = 0
        self.max_position = 50000
        self.min_speed = 0
        self.max_speed = 180
        self.min_gear = 1
        self.max_gear = 6
        self.min_limit = 40
        self.max_limit = 150
        self.min_revolutions = 0
        self.max_revolutions = 6000
        self.min_consume = 0
        self.max_consume = 30
        self.gear_r = [2.5, 1.4, 0.9, 0.7, 0.55, 0.45]
        self.alpha = 1.8
        self.beta1 = 100
        self.beta2 = 1.22

        self.low = np.array([self.min_position, self.min_speed, self.min_gear,
                             self.min_revolutions, self.min_consume, self.min_limit], dtype=np.float32)
        self.high = np.array([self.max_position, self.max_speed, self.max_gear,
                              self.max_revolutions, self.max_consume, self.max_limit], dtype=np.float32)

        self.action_space = spaces.Discrete(5)
        self.observation_space = spaces.Box(self.low, self.high, dtype=np.float32)

    def step(self, action):
        assert self.action_space.contains(
            action
        ), f"{action!r} ({type(action)}) invalid"

        position, speed, gear, revolutions, consume, limit = self.state

        if action in [0, 1, 2]:      # action regarding speed
            speed += (action - 1) * (6)

        elif action in [3, 4]:      # action regarding gears
            # we sum -1 or 1 to the gear depending on the action
            gear += 2/1*(action-4)+1
            gear = np.clip(gear, self.min_gear, self.max_gear)

        revolutions = (self.gear_r[int(gear-1)]*4.64*speed*1000)/(1.83*60)
        revolutions = np.clip(revolutions, self.min_revolutions, self.max_revolutions)
        speed = np.clip(speed, self.min_speed, self.max_speed)

        # conversion from km/h to m/s as the position is in meters
        position += speed*0.2778
        position = np.clip(position, self.min_position, self.max_position)

        # speed limits
        if position > 3000:
            limit = 120
        if position > 7000:
            limit = 100
        if position > 20000:
            limit = 70
        if position > 30000:
            limit = 140
        if position > 35000:
            limit = 80
        if position > 40000:
            limit = 90

        # consume calculation depending on the gear
        if gear == 1:
            consume = 0.0059*speed**2-0.1857*speed+11.0892
        elif gear == 2:
            consume = 0.0012*speed**2-0.0574*speed+6.6088
        elif gear == 3:
            consume = 0.0005*speed**2-0.0207*speed+4.5582
        elif gear == 4:
            consume = 0.0004*speed**2-0.0216*speed+4.1160
        elif gear == 5:
            consume = 0.0004*speed**2-0.0183*speed+3.6267
        elif gear == 6:
            consume = 0.0004*speed**2-0.0198*speed+3.4347

        consume = np.clip(consume, self.min_consume, self.max_consume)

        # we finish when
        done = bool(position >= self.max_position)

        # reward function calculation
        reward_vel = self.alpha*speed-consume
        reward_pen = 0

        if speed > limit:
            reward_pen = -speed-limit
        rev_pen = 0

        if revolutions > 3000 or revolutions < 2000:
            rev_pen = -revolutions/self.beta1-speed*self.beta2

        reward = reward_vel + reward_pen + rev_pen

        self.state = (position, speed, gear, revolutions, consume, limit)
        return np.array(self.state, dtype=np.float32), reward, done, {}

    def reset(
        self,
        return_info: bool = False,
        options: Optional[dict] = None,
    ):
        self.state = np.array([0, 0, 1, 0, 0, randrange(50, 150)])
        if not return_info:
            return np.array(self.state, dtype=np.float32)
        else:
            return np.array(self.state, dtype=np.float32), {}
