"""Independent, fail-closed validation of the fixed-input evaluation contract."""
import numpy as np
import torch


def audit_inputs(trace, common):
    if trace['command_input_mode'].item() != 'source_held_before_observation':
        raise ValueError('Expected source-held commands')
    c = {str(k): i for i, k in enumerate(trace['columns'])}
    data = trace['data']
    get = lambda name: data[:, c[name]]
    n = len(data)
    for prefix in ('observation', 'actor', 'environment'):
        np.testing.assert_allclose(get(prefix + '_cmd_vx'), common['pin_vx'], atol=1e-7, rtol=0)
        np.testing.assert_array_equal(get(prefix + '_cmd_vy'), 0)
    for prefix in ('observation', 'environment'):
        np.testing.assert_array_equal(get(prefix + '_standing'), 0)
    for name in ('actor_cmd_vyaw', 'environment_cmd_vyaw'):
        np.testing.assert_allclose(get(name), get('cmd_vyaw'), atol=6e-8, rtol=1e-7)
    np.testing.assert_allclose(get('observation_cmd_vyaw'), np.r_[0., get('cmd_vyaw')[:-1]], atol=6e-8, rtol=1e-7)
    # Use the same float32 time convention, independently of recorded phase.
    phase = (torch.arange(n) * float(common['dt'])) % 0.6 / 0.6
    expected = torch.stack((torch.sin(phase * torch.pi * 2), torch.cos(phase * torch.pi * 2)), dim=1).numpy()
    actual = np.column_stack((get('phase_sin'), get('phase_cos')))
    np.testing.assert_allclose(actual, expected, atol=3e-6, rtol=0)
    norm = np.linalg.norm(actual, axis=1)
    np.testing.assert_allclose(norm, 1, atol=2e-7, rtol=0)
    return {'input_checked_samples': n, 'zero_phase_samples': int(np.sum(norm < .1)),
            'max_phase_absolute_error': float(np.max(np.abs(actual - expected)))}
