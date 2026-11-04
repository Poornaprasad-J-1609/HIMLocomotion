"""Read-only verification of standing-derived warm-start provenance."""
import hashlib
from pathlib import Path
import torch

PROJECT = Path(__file__).resolve().parents[4]
SOURCE = PROJECT / 'IsaacLab/logs/rsl_rl/grallator_him_standing/2026-11-03_21-06-02_middle_pose_038_resume/model_1700.pt'
SOURCE_SHA256 = '10c9f75518359d9206c66c42f0a51ba6a68f84fcadd4f26c2b323dfd142c8ced'
WARM_START = PROJECT / 'grallator_rl/checkpoints/walking_from_standing.pt'


def verify_warm_start(path=WARM_START):
    digest = hashlib.sha256(SOURCE.read_bytes()).hexdigest()
    if digest != SOURCE_SHA256:
        raise ValueError('Trusted standing checkpoint checksum mismatch')
    source = torch.load(SOURCE, map_location='cpu', weights_only=True)
    warm = torch.load(path, map_location='cpu', weights_only=True)
    for group in ('actor_state_dict', 'critic_state_dict'):
        if source[group].keys() != warm[group].keys():
            raise ValueError(f'{group} tensor keys differ')
        for name, value in source[group].items():
            candidate = warm[group][name]
            if candidate.shape != value.shape:
                raise ValueError(f'Shape differs: {group}/{name}')
            if group == 'actor_state_dict' and name == 'distribution.std_param':
                if not torch.allclose(candidate, torch.full_like(candidate, .12)):
                    raise ValueError('Warm-start exploration std must be 0.12')
            elif not torch.equal(candidate, value):
                raise ValueError(f'Warm-start changed pretrained tensor: {group}/{name}')
    if warm['iter'] != 0:
        raise ValueError('Warm-start iteration must be zero')
    for name in ('optimizer_state_dict', 'hio_optimizer_state_dict'):
        if warm[name]['state']:
            raise ValueError(f'Warm-start {name} is not fresh')
    return {'source': str(SOURCE), 'source_sha256': digest, 'warm_start': str(Path(path).resolve()),
            'original_std_mean': source['actor_state_dict']['distribution.std_param'].mean().item(),
            'warm_std_mean': warm['actor_state_dict']['distribution.std_param'].mean().item(),
            'iteration': warm['iter']}
