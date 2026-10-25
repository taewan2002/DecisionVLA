"""Post-hoc geometry/confidence diagnostics from saved LIBERO states; no policy or stepping."""
import argparse
import json
from pathlib import Path

import numpy as np

from decisionvla.rollout import digest, write_json


def consecutive(values, count=3):
    return len(values) >= count and bool(np.any(np.convolve(np.asarray(values, int), np.ones(count, int), 'valid') == count))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--task-id', type=int, default=0)
    parser.add_argument('--episodes', type=int, default=50)
    args = parser.parse_args()
    from libero.libero import benchmark
    from libero.libero.envs import OffScreenRenderEnv

    contract = json.loads((args.results / 'contract.json').read_text())
    suite = benchmark.get_benchmark_dict()[contract['config']['suite']](task_order_index=0)
    env = OffScreenRenderEnv(bddl_file_name=suite.get_task_bddl_file_path(args.task_id),
                            camera_heights=128, camera_widths=128, camera_names=['agentview'],
                            control_freq=20, render_gpu_device_id=-1)
    env.seed(0)
    env.reset()
    target_name, goal_name = env.obj_of_interest
    target_body, goal_body = env.env.obj_body_id[target_name], env.env.obj_body_id[goal_name]
    gripper = env.robots[0].gripper
    left = {env.sim.model.geom_name2id(n) for n in gripper.important_geoms['left_finger']}
    right = {env.sim.model.geom_name2id(n) for n in gripper.important_geoms['right_finger']}
    target = {env.sim.model.geom_name2id(n) for n in env.env.objects_dict[target_name].contact_geoms}
    rows, source_hashes, details = [], {}, {}
    state_restore_max_abs = 0.0
    try:
        for initial in range(args.episodes):
            folder = args.results / f'task-{args.task_id:02d}-init-{initial:02d}'
            result = json.loads((folder / 'result.json').read_text())
            assert digest(folder / 'trace.npz') == result['artifacts_sha256']['trace.npz']
            source_hashes[folder.name] = {'result': digest(folder / 'result.json'),
                                        'trace': result['artifacts_sha256']['trace.npz']}
            with np.load(folder / 'trace.npz', allow_pickle=False) as trace:
                positions, distances, goal_distances, contacts = [], [], [], []
                for state in trace['states']:
                    env.set_state(state)
                    env.sim.forward()
                    restored = env.get_sim_state()
                    state_restore_max_abs = max(state_restore_max_abs, float(np.max(np.abs(restored - state))))
                    # MuJoCo forward may normalize quaternion components at machine precision.
                    np.testing.assert_allclose(restored, state, atol=1e-12, rtol=0)
                    position = env.sim.data.body_xpos[target_body].copy()
                    goal = env.sim.data.body_xpos[goal_body].copy()
                    eef = env.sim.data.site_xpos[env.robots[0].eef_site_id]
                    positions.append(position)
                    distances.append(np.linalg.norm(eef - position))
                    goal_distances.append(np.linalg.norm(position[:2] - goal[:2]))
                    touching = set()
                    for contact in env.sim.data.contact[:env.sim.data.ncon]:
                        if contact.geom1 in target:
                            touching.add(contact.geom2)
                        if contact.geom2 in target:
                            touching.add(contact.geom1)
                    contacts.append(bool(touching & left) and bool(touching & right))
                assert bool(env.check_success()) == result['success']
                positions = np.asarray(positions)
                height = positions[:, 2] - positions[0, 2]
                bilateral = consecutive(contacts)
                lift = consecutive(height > 0.03)
                category = ('success' if result['success'] else 'no_sustained_bilateral_contact' if not bilateral
                            else 'contact_without_3cm_lift' if not lift else 'lift_without_task_success')
                scores = trace['ordered_confidence']
                row = {'initial_id': initial, 'success': result['success'], 'category': category,
                       'policy_steps': result['policy_steps'], 'min_eef_target_m': float(min(distances)),
                       'maximum_target_height_gain_m': float(max(height)),
                       'target_displacement_final_m': float(np.linalg.norm(positions[-1] - positions[0])),
                       'final_target_goal_xy_m': float(goal_distances[-1]),
                       'sustained_bilateral_contact': bilateral, 'sustained_lift_3cm': lift,
                       'first_chunk_confidence': float(trace['chunk_confidence'][0]),
                       'all_query_mean_confidence': float(trace['chunk_confidence'].mean()),
                       'first_confidence_by_axis': scores[0].mean(0).tolist(),
                       'first_peak_probability_by_axis': trace['selected_probability'][0].mean(0).tolist(),
                       'first_min_component_confidence': float(scores[0].min()),
                       'gripper_close_fraction': float(np.mean(trace['actions'][:, 6] > 0)),
                       'action_abs_mean_by_axis': np.abs(trace['actions']).mean(0).tolist()}
                rows.append(row)
                details[f'init_{initial:02d}_height_m'] = height
                details[f'init_{initial:02d}_eef_distance_m'] = np.asarray(distances)
                details[f'init_{initial:02d}_bilateral_contact'] = np.asarray(contacts)
                details[f'init_{initial:02d}_query_steps'] = trace['query_steps']
                details[f'init_{initial:02d}_confidence'] = trace['chunk_confidence']
    finally:
        env.close()
    failures = [r for r in rows if not r['success']]
    report = {'kind': 'posthoc_saved_state_diagnostic_not_new_rollout',
              'task_id': args.task_id, 'episodes': len(rows), 'new_policy_calls': 0, 'new_simulation_steps': 0,
              'state_restore_max_abs': state_restore_max_abs, 'state_restore_atol': 1e-12,
              'geometry': {'target': target_name, 'goal': goal_name,
                           'bilateral_contact': 'target contact with each finger for >=3 consecutive saved states',
                           'lift': 'target center >3cm above its own settled initial height for >=3 saved states',
                           'limitations': 'post-hoc geometric proxies; contact/lift do not prove a stable grasp'},
              'categories': {key: sum(r['category'] == key for r in rows) for key in sorted({r['category'] for r in rows})},
              'failures_first_confidence_by_axis': (
                  np.mean([r['first_confidence_by_axis'] for r in failures], 0).tolist() if failures else None),
              'failures_first_peak_probability_by_axis': (
                  np.mean([r['first_peak_probability_by_axis'] for r in failures], 0).tolist() if failures else None),
              'rows': rows, 'source_hashes': source_hashes, 'contract_sha256': digest(args.results / 'contract.json')}
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(args.output / 'geometry.json', report)
    np.savez_compressed(args.output / 'geometry-traces.npz', **details)
    print(json.dumps({k: v for k, v in report.items() if k not in ['rows', 'source_hashes']}))
    print(json.dumps({'examples': [r for r in rows if r['initial_id'] in [0, 1, 19, 25, 49]]}))


if __name__ == '__main__':
    main()
