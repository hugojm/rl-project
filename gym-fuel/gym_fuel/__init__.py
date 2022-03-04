import logging
from gym.envs.registration import register

logger = logging.getLogger(__name__)

register(
    id='Fuel-v0',
    entry_point='gym_fuel.envs:FuelCarEnv',

)
