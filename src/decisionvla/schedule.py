"""Fixed development-checkpoint selection, never an independent-test claim."""
import csv
import json
from pathlib import Path


def select_checkpoint(plan, reports):
    expected = plan['evaluation_steps']
    fixed_final = plan['selection'].get('policy') == 'fixed_final_checkpoint'
    if fixed_final and (expected != [plan['total_steps']]
                        or plan['selection'].get('fixed_step') != plan['total_steps']):
        raise ValueError('Fixed-final evaluation must use only the declared final checkpoint')
    if sorted(reports) != expected or plan['selection']['candidate_steps'] != expected:
        raise ValueError('Every planned checkpoint must finish before selection')
    rows = []
    for step in expected:
        report = reports[step]
        if (report['status'] != 'completed' or report['episodes'] != plan['episodes_per_checkpoint']
                or report['protocol']['checkpoint_step'] != step):
            raise ValueError('Incomplete or mismatched evaluation')
        tasks = report['tasks']
        if (sorted(t['task_id'] for t in tasks) != list(range(10))
                or any(t['episodes'] != 50 for t in tasks)
                or sum(t['episodes'] - t['failures'] for t in tasks) != report['successes']):
            raise ValueError('Expected ten tasks with fifty episodes each')
        # With 50 episodes in every task, this is exactly the macro rate.
        # Compute once from integers so a floating-point summation cannot break a tie.
        macro = report['successes'] / report['episodes']
        rows.append({'step':step, 'successes':report['successes'], 'episodes':report['episodes'],
                     'success_rate':macro, 'within_task_failure_auroc':report['primary_within_task_auroc'],
                     'auroc_ci95':report['primary_ci95']})
    best = rows[0] if fixed_final else max(rows, key=lambda r:(r['success_rate'], -r['step']))
    return {'status':'completed', 'selected_step':best['step'], 'selection':plan['selection'],
            'checkpoints':rows, 'scope':plan['scope'], 'independent_test':False}


def write_selection(plan, reports, output):
    result = select_checkpoint(plan, reports)
    output = Path(output)
    output.mkdir(exist_ok=True)
    (output / 'selection.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    filename = ('final-evaluation.csv' if plan['selection'].get('policy') == 'fixed_final_checkpoint'
                else 'learning-curve.csv')
    with (output / filename).open('w') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(result['checkpoints'][0]))
        writer.writeheader()
        writer.writerows(result['checkpoints'])
    return result
