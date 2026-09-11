from types import SimpleNamespace
import torch
from mjlab_microduck.basketball_finetune import set_fixed_learning_rate


def test_lr_override_preserves_restored_adam_moments():
    p = torch.nn.Parameter(torch.tensor([1.]))
    optimizer = torch.optim.Adam([p], lr=.001)
    p.sum().backward()
    optimizer.step()
    state = {k:v.clone() for k,v in optimizer.state[p].items()}
    alg = SimpleNamespace(optimizer=optimizer, learning_rate=.001, schedule='adaptive')
    set_fixed_learning_rate(alg, 2e-5)
    assert alg.schedule == 'fixed' and alg.learning_rate == 2e-5
    assert optimizer.param_groups[0]['lr'] == 2e-5
    assert all(torch.equal(v, optimizer.state[p][k]) for k,v in state.items())


def test_continuation_keeps_blind_observations_and_free_ball():
    from mjlab_microduck.basketball_finetune import continuation_configs
    from mjlab_microduck.tasks import mdp
    checkpoint = {'actor_state_dict': {
        'mlp.0.weight':torch.empty(512,256), 'mlp.2.weight':torch.empty(256,512),
        'mlp.4.weight':torch.empty(128,256), 'mlp.6.weight':torch.empty(14,128),
        'rnn.rnn.weight_ih_l0':torch.empty(1024,61),
        'rnn.rnn.weight_hh_l0':torch.empty(1024,256),
    }}
    cfg, agent = continuation_configs(checkpoint, action_rate_weight=-.3, push_interval_s=(1.5,3.))
    assert cfg.observations['actor'].terms['body_command'].func is mdp.basketball_body_pad
    assert cfg.observations['critic'].terms['body_command'].func is mdp.basketball_state
    assert cfg.actions['ball_hold'].levels == (0.,)
    assert cfg.events['push_robot'].interval_range_s == (1.5,3.)
    assert 'basketball_hold' not in cfg.curriculum and 'action_rate_weight' not in cfg.curriculum
    assert cfg.rewards['action_rate_l2'].weight == -.3
    assert agent['actor']['class_name'] == 'RNNModel'
    assert agent['algorithm']['schedule'] == 'fixed'
    assert agent['algorithm']['learning_rate'] == 2e-5
