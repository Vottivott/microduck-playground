"""Conservative PPO continuation without restoring a stale optimizer learning rate."""
from dataclasses import asdict
import math

from mjlab.rl import MjlabOnPolicyRunner
from mjlab.tasks.registry import load_rl_cfg
from mjlab_microduck.basketball_distillation import actor_model_cfg
from mjlab_microduck.tasks.microduck_basketball_env_cfg import make_microduck_basketball_env_cfg


def set_fixed_learning_rate(algorithm, learning_rate):
    if not math.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError('learning_rate must be finite and positive')
    algorithm.schedule = 'fixed'
    algorithm.learning_rate = learning_rate
    for group in algorithm.optimizer.param_groups:
        group['lr'] = learning_rate


class BasketballContinuationRunner(MjlabOnPolicyRunner):
    def load(self, path, load_cfg=None, strict=True, map_location=None):
        requested_rate = self.cfg['algorithm']['learning_rate']
        result = super().load(path, load_cfg, strict, map_location)
        # Adam.load_state_dict restores param-group LR, overriding the CLI/config.
        # Keep its moment estimates, but explicitly install the continuation's LR.
        set_fixed_learning_rate(self.alg, requested_rate)
        return result


def continuation_configs(checkpoint, *, learning_rate=2e-5, action_rate_weight=-.2,
                         command_scale=1., episode_seconds=10., seed=42, push_interval_s=None):
    if not math.isfinite(action_rate_weight) or action_rate_weight > 0:
        raise ValueError('action_rate_weight must be finite and non-positive')
    if not all(math.isfinite(x) and x > 0 for x in (learning_rate, command_scale, episode_seconds)):
        raise ValueError('learning rate, command scale and episode seconds must be positive')
    cfg = make_microduck_basketball_env_cfg(blind=True, history=1)
    cfg.seed = seed
    if push_interval_s is not None:
        if len(push_interval_s) != 2 or not all(math.isfinite(v) and v > 0 for v in push_interval_s) or push_interval_s[0] > push_interval_s[1]:
            raise ValueError('push interval must be two positive ordered values')
        cfg.events['push_robot'].interval_range_s = tuple(push_interval_s)
    cfg.episode_length_s = episode_seconds
    cfg.actions['ball_hold'].levels = (0.,)
    cfg.curriculum.pop('basketball_hold', None)
    cfg.rewards['action_rate_l2'].weight = action_rate_weight
    cfg.curriculum.pop('action_rate_weight', None)
    cfg.commands['twist'].ranges.lin_vel_x = (-.15*command_scale, .15*command_scale)
    cfg.commands['twist'].ranges.lin_vel_y = (-.10*command_scale, .10*command_scale)
    cfg.commands['twist'].ranges.ang_vel_z = (-.50*command_scale, .50*command_scale)
    agent = asdict(load_rl_cfg('Mjlab-Basketball-MicroDuck'))
    agent['actor'] = actor_model_cfg(checkpoint)
    if agent['actor']['class_name'] != 'RNNModel':
        raise ValueError('Expected a recurrent basketball source policy')
    agent['seed'] = seed
    agent['algorithm']['learning_rate'] = learning_rate
    agent['algorithm']['schedule'] = 'fixed'
    return cfg, agent
