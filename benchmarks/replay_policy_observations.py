"""Reconstruct exact recorded LIBERO query images by replaying saved controls.

No new policy is sampled. Direct state restoration can alter renderer timing,
so the original settling/control sequence and every query-image hash are used.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from decisionvla.rollout import camera_image, digest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    from libero.libero import benchmark
    from libero.libero.envs import OffScreenRenderEnv
    contract = json.loads((args.results/'contract.json').read_text())
    config = contract['config']
    args.output.mkdir(parents=True, exist_ok=False)
    suite = benchmark.get_benchmark_dict()[config['suite']](task_order_index=0)
    records, episodes, controls = [], [], 0
    max_state_error = 0.0
    for task in config['task_ids']:
        env = OffScreenRenderEnv(bddl_file_name=suite.get_task_bddl_file_path(task),
                                camera_heights=128, camera_widths=128, camera_names=['agentview'],
                                control_freq=20, render_gpu_device_id=-1)
        try:
            for initial in config['initial_state_ids']:
                folder = args.results/f'task-{task:02d}-init-{initial:02d}'
                result = json.loads((folder/'result.json').read_text())
                trace_sha = digest(folder/'trace.npz')
                assert trace_sha == result['artifacts_sha256']['trace.npz']
                with np.load(folder/'trace.npz', allow_pickle=False) as trace:
                    env.seed(config['environment_seed'])
                    env.reset()
                    observation = env.set_init_state(trace['initial_state'])
                    for _ in range(config['settle_steps']):
                        observation, _, _, _ = env.step([0, 0, 0, 0, 0, 0, -1])
                    query_at = {int(step): q for q, step in enumerate(trace['query_steps'])}
                    for step, action in enumerate(trace['actions']):
                        error = float(np.max(np.abs(env.get_sim_state()-trace['states'][step])))
                        max_state_error = max(max_state_error, error)
                        assert error <= 1e-12
                        if step in query_at:
                            q = query_at[step]
                            image = camera_image(observation)
                            sha = hashlib.sha256(np.asarray(image).tobytes()).hexdigest()
                            assert sha == trace['image_sha256'][q], (task, initial, q)
                            name = f'task-{task:02d}-init-{initial:02d}-query-{q:03d}.png'
                            image.save(args.output/name)
                            records.append({'file': name, 'task': task, 'initial': initial, 'query': q,
                                            'seed': int(trace['query_seeds'][q]), 'instruction': result['instruction'],
                                            'image_sha256': sha, 'trace_sha256': trace_sha})
                        observation, _, _, _ = env.step(action.tolist())
                        controls += 1
                    np.testing.assert_allclose(env.get_sim_state(), trace['states'][-1], atol=1e-12, rtol=0)
                    assert bool(env.check_success()) == result['success']
                    episodes.append({'task': task, 'initial': initial, 'queries': len(query_at),
                                     'terminal_success_matches': True})
                print(json.dumps(episodes[-1]), flush=True)
        finally:
            env.close()
    (args.output/'records.json').write_text(json.dumps(records, indent=2)+'\n')
    report = {'status': 'completed', 'episodes': len(episodes), 'queries': len(records),
              'all_query_images_exact': True, 'state_restore_max_error': max_state_error,
              'replayed_controls': controls, 'new_policy_calls': 0,
              'interpretation': 'Replay of existing trajectories; not an additional success cohort',
              'source_contract_sha256': digest(args.results/'contract.json')}
    (args.output/'completion.json').write_text(json.dumps(report, indent=2)+'\n')


if __name__ == '__main__':
    main()
