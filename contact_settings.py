"""Explicit G1 foot-ground friction configuration and compiled/runtime provenance.

Legacy TerrainEntityCfg.static_friction/dynamic_friction assignments do not
reach this installed mjlab terrain builder. Foot geoms have higher priority
than terrain, so their sliding coefficient determines dynamic foot contacts.
"""
import hashlib
import math
import re
import numpy as np

FOOT = re.compile(r'(?:robot/)?(?:left|right)_foot[1-7]_collision$')

def configure_contact_friction(scene_cfg, value):
    if value is None:
        return
    if not math.isfinite(value) or value <= 0:
        raise ValueError('contact_friction must be positive and finite')
    original = scene_cfg.spec_fn
    def edit(spec):
        if original is not None:
            original(spec)
        feet = [g for g in spec.geoms if FOOT.fullmatch(g.name)]
        terrain = [g for g in spec.geoms if g.name == 'terrain']
        if len(feet) != 14 or len(terrain) != 1 or len(spec.pairs):
            raise ValueError('Explicit friction override requires the verified flat G1 geometry')
        if any(g.priority <= terrain[0].priority for g in feet):
            raise ValueError('Foot priority no longer determines contact friction')
        for g in feet:
            g.friction[0] = value
    scene_cfg.spec_fn = edit

def contact_record(sim, requested):
    m = sim.mj_model
    gpu = sim.model.geom_friction.detach().cpu().numpy()
    if gpu.ndim == 3:
        if len(gpu) != 1:
            raise ValueError('Contact record currently requires one environment')
        gpu = gpu[0]
    feet = [i for i in range(m.ngeom) if FOOT.fullmatch(m.geom(i).name or '')]
    terrain = [i for i in range(m.ngeom) if m.geom(i).name == 'terrain']
    if requested is not None:
        if len(feet) != 14 or len(terrain) != 1 or m.npair:
            raise ValueError('Unexpected compiled collision geometry')
        np.testing.assert_allclose(m.geom_friction[feet,0],requested,rtol=0,atol=1e-12)
        np.testing.assert_allclose(gpu[feet,0],requested,rtol=0,atol=1e-7)
        if np.any(m.geom_priority[feet] <= m.geom_priority[terrain[0]]):
            raise ValueError('Unexpected collision priorities')
    return dict(requested_contact_friction=requested,
        recording='compiled MuJoCo and active GPU model before stepping',
        legacy_terrain_friction_request_effect='not consumed by installed TerrainEntityCfg',
        npair=int(m.npair),compiled_friction_sha256=hashlib.sha256(m.geom_friction.tobytes()).hexdigest(),
        geoms=[dict(name=m.geom(i).name,priority=int(m.geom_priority[i]),condim=int(m.geom_condim[i]),
            compiled_friction=m.geom_friction[i].tolist(),active_gpu_friction=gpu[i].tolist()) for i in terrain+feet])
