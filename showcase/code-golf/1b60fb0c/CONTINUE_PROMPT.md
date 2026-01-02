# Task: Fix 1b60fb0c (Shape Mirroring) - ARC Code Golf

Working directory: `/Users/aerickson/Documents/Claude Code Projects/agentic-evolve/showcase/code-golf`

## Status
Train 1 and Train 2 pass, but Train 0 and Test 0 fail.

## The Pattern (discovered through analysis)

1. Find "spine" = most common leftmost column across rows
2. Find "bars" = contiguous rows at top/bottom with smallest leftmost value
3. "Corridor" = rows between bars that have 1s extending beyond spine
4. For corridor rows, mirror 1s from right of spine to left, placing 2s

## Key rules discovered

- When there's a bottom bar: mirror all 1s beyond spine (for continuous rows) or only 1s >= spine+2 (for gap rows)
- When no bottom bar: exclude spine-adjacent and edge cells from mirroring
- Fill gaps in 2-region for continuous rows
- Don't fill gaps for rows with gaps in the 1s
- Vertical connectivity: rows without direct mirrors get 2 at col 1 if adjacent to row with 2s

## Exact mirror relationships from training data

```
Train 0 (spine=4, NO bottom bar):
  Row 4: 1s_right=[5,6,7,8], 2s=[1,2] - mirrors 7,6 only (exclude 5,8)
  Row 5: 1s_right=[5,7,8], 2s=[1,2,3] - mirrors 7,5 + fills gap at 2
  Row 6: 1s_right=[5], 2s=[1] - vertical connectivity (no direct mirror)

Train 1 (spine=5, HAS bottom bar):
  Rows 3,4,6,7: 1s_right=[8,9], 2s=[1,2] - mirrors all
  Row 5: 1s_right=[6,7,8,9], 2s=[1,2,3,4] - mirrors all

Train 2 (spine=5, HAS bottom bar):
  Rows 3,7: 1s_right=[6,9], 2s=[1] - mirrors only 9 (6 is spine+1, skip)
  Rows 4,6: 1s_right=[7,9], 2s=[1,3] - mirrors both (no fill)
  Row 5: 1s_right=[6,7,8,9], 2s=[1,2,3,4] - mirrors all (continuous)
```

## Current failures

- Train 0: Rows 3,7,8 getting 2s but shouldn't (corridor detection too broad)
- Test 0: Various issues with corridor bounds and mirror positions

## Files

- Task: `tasks/1b60fb0c.json`
- Solution: `1b60fb0c/solution.py`
- Evaluator: `python3 evaluator.py 1b60fb0c 1b60fb0c/solution.py`

## Approach

First get a working solution (all 4 examples pass), then golf it down.

## Command to test solutions

```bash
python3 << 'EOF'
import json

def solve(g):
    # paste implementation here
    pass

with open('tasks/1b60fb0c.json') as f:
    task = json.load(f)
for t_idx, example in enumerate(task['train'] + task['test']):
    result = solve(example['input'])
    expected = example['output']
    status = "PASS" if result == expected else "FAIL"
    print(f"{'Train' if t_idx < 3 else 'Test'} {t_idx if t_idx < 3 else 0}: {status}")
    if status == "FAIL":
        for i in range(len(result)):
            exp_2s = sorted(j for j in range(len(result[0])) if expected[i][j] == 2)
            got_2s = sorted(j for j in range(len(result[0])) if result[i][j] == 2)
            if exp_2s != got_2s:
                print(f"  Row {i}: exp {exp_2s}, got {got_2s}")
EOF
```

## Resume prompt

```
Continue working on task 1b60fb0c. Read 1b60fb0c/CONTINUE_PROMPT.md for context, then fix the solution to pass all training and test examples. Focus solely on this task.
```
