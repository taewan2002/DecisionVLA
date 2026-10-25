import copy
import json
import subprocess
import sys

import numpy as np
import torch

from decisionvla.continuation import learning_rate_at
from decisionvla.train import restore_checkpoint, save_checkpoint


def test_optimizer_and_rng_survive_a_process_boundary(tmp_path, monkeypatch):
    # Exercise the actual serialization used by train.py, with a small CPU policy.
    monkeypatch.setattr(torch.cuda, 'get_rng_state', lambda: torch.tensor([1, 2], dtype=torch.uint8))
    monkeypatch.setattr(torch.cuda, 'set_rng_state', lambda value: None)
    config={'steps':50000,'schedule_start_step':10000,'warmup_steps':500,
            'learning_rate':5e-5,'warmup_start_learning_rate':5e-6}
    def initialize():
        torch.manual_seed(42)
        model=torch.nn.Linear(3, 2)
        optimizer=torch.optim.AdamW(model.parameters(),lr=5e-5)
        return model,optimizer,np.random.default_rng(42),torch.Generator().manual_seed(43)
    def update(bundle,step):
        model,optimizer,sampler,noise=bundle
        optimizer.param_groups[0]['lr']=learning_rate_at(step,config)
        key=int(sampler.integers(1000))
        x=torch.randn(4,3,generator=noise)+torch.rand(4,3)
        optimizer.zero_grad()
        model(x).square().mean().backward()
        optimizer.step()
        return key
    uninterrupted=initialize()
    keys=[update(uninterrupted,s) for s in range(10001,10007)]
    expected=copy.deepcopy(uninterrupted[0].state_dict())
    expected_opt=copy.deepcopy(uninterrupted[1].state_dict())
    split=initialize()
    first_keys=[update(split,s) for s in range(10001,10004)]
    checkpoint=save_checkpoint(split[0],split[1],tmp_path,10003,config,split[2],split[3],{})
    restarted=initialize()
    assert restore_checkpoint(restarted[0],restarted[1],checkpoint,restarted[2],restarted[3]) == 10003
    assert first_keys+[update(restarted,s) for s in range(10004,10007)] == keys
    for name,value in restarted[0].state_dict().items():
        torch.testing.assert_close(value,expected[name],atol=0,rtol=0)
    for key,state in restarted[1].state_dict()['state'].items():
        for name,value in state.items():
            torch.testing.assert_close(value,expected_opt['state'][key][name],atol=0,rtol=0)


def test_release_contains_only_adapter_config_card_and_hashes(tmp_path):
    from decisionvla.checkpoints import save_adapter
    checkpoint=tmp_path/'checkpoint'
    checkpoint.mkdir()
    save_adapter(torch.nn.Linear(3,2),checkpoint)
    (checkpoint/'adapter_config.json').write_text('{}')
    (checkpoint/'training_metadata.json').write_text('{"private":"excluded"}')
    (checkpoint/'training_state.pt').write_text('excluded optimizer')
    card=tmp_path/'MODEL_CARD.md'
    card.write_text('# Experimental adapter\n')
    output=tmp_path/'assets'
    subprocess.run([sys.executable,'scripts/package_adapter.py',str(checkpoint),'--model-card',str(card),
                    '--output',str(output),'--sharded-release'],check=True,capture_output=True)
    assert {p.name for p in output.iterdir()} == {'adapter.safetensors','adapter_config.json','MODEL_CARD.md','sha256.json'}
    assert set(json.loads((output/'sha256.json').read_text())) == {'adapter.safetensors','adapter_config.json','MODEL_CARD.md'}
