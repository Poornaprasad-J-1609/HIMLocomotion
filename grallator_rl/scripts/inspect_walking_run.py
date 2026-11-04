"""Read native TensorBoard events and compare trained weights to warm-start."""
import argparse
import json
from pathlib import Path
import torch
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from grallator_rl.him.walking_checkpoint import WARM_START, verify_warm_start


def inspect_run(path):
    events=EventAccumulator(str(path),size_guidance={'scalars':0}).Reload()
    tags=events.Tags()['scalars']
    for tag in ('Reward_Rate/feet_air_time','Loss/value','Loss/surrogate','Loss/entropy',
                'Loss/ppo_kl','Loss/post_hio_kl'):
        assert tag in tags,tag
    scalars={tag:[{'step':e.step,'value':e.value} for e in events.Scalars(tag)] for tag in tags}
    assert all(torch.isfinite(torch.tensor([e['value'] for e in points])).all()
               for points in scalars.values()), 'Nonfinite event values'
    checkpoints=sorted(path.glob('model_*.pt'),key=lambda p:int(p.stem.split('_')[-1]))
    assert checkpoints,'No saved checkpoint'
    model=torch.load(checkpoints[-1],map_location='cpu',weights_only=True)
    warm=torch.load(WARM_START,map_location='cpu',weights_only=True)
    changes={group:[name for name,value in warm[group].items() if not torch.equal(value,model[group][name])]
             for group in ('actor_state_dict','critic_state_dict')}
    return dict(log_dir=str(path.resolve()),checkpoint=str(checkpoints[-1].resolve()),
                iteration=model['iter'],changed_tensors=changes,
                optimizer_state_count=len(model['optimizer_state_dict']['state']),
                hio_optimizer_state_count=len(model['hio_optimizer_state_dict']['state']),
                scalar_count=len(tags),scalars=scalars)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--log-dir',type=Path,required=True)
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    result=inspect_run(args.log_dir)
    result['provenance']=verify_warm_start()
    encoded=json.dumps(result,indent=2)+'\n'
    if args.output: args.output.write_text(encoded)
    print(json.dumps({k:v for k,v in result.items() if k!='scalars'},indent=2))
    for tag in ('Loss/ppo_kl','Loss/post_hio_kl','Reward_Rate/feet_air_time','Episode_Reward/feet_air_time'):
        print(tag,result['scalars'].get(tag,'NO COMPLETED EPISODE IN THIS RUN'))


if __name__=='__main__': main()
