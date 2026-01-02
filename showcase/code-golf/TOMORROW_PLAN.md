# Tomorrow's Plan: Code Golf Optimization

## Session Strategy (Combat Context Rot)

| Rule | Rationale |
|------|-----------|
| **2-3 tasks per session** | Keep context under ~50K tokens |
| **Clear between complex tasks** | Fresh context for novel patterns |
| **Hardest first** | Use clean context for tricky problems |
| **Single-task for V.Hard** | Maximum attention on algorithm rework |

---

## Phase 1: Re-Golf Scan (1-2 sessions)

### High Priority (600+ bytes - V.Hard tier)

These drag down our average significantly:

| Task | Current | Target | Potential Gain | Notes |
|------|---------|--------|----------------|-------|
| `0e206a2e` | 1,384 | ~600 | +784 pts | Rotated template - needs algorithm rework |
| `2dd70a9a` | 1,163 | ~600 | +563 pts | Periodic pattern - check for simpler approach |

**Strategy**: Fresh session, read task + current solution, look for completely different algorithm.

### Medium Priority (500-600 bytes)

| Task | Current | Target | Potential Gain | Notes |
|------|---------|--------|----------------|-------|
| `045e512c` | 591 | ~400 | +191 pts | Template matching |
| `0a938d79` | 539 | ~400 | +139 pts | Fill pattern grid |
| `a64e4611` | 523 | ~400 | +123 pts | Largest rectangle + cross |

**Strategy**: Quick scan with fresh context - apply new tricks learned in recent batches.

### Quick Wins Check (300-500 bytes)

Scan these for obvious improvements using new tricks:

```
178fcbfb (304), 05f2a901 (326), 11852cab (333), 06df4c85 (378),
1bfc4729 (406), 0b148d64 (454), 2bcee788 (465)
```

---

## Phase 2: New Tasks (2-3 sessions)

### From Analyzed Queue (Medium difficulty preferred)

Pick 6-8 tasks from the analyzed list, prioritizing Medium difficulty:

| Task | Pattern | Est. Difficulty |
|------|---------|-----------------|
| `0a938d79` | Fill pattern grid | Medium |
| `178fcbfb` | Fill cross pattern gaps | Medium |
| `1f876c06` | Diagonal line propagation | Medium |
| `22233c11` | Horizontal line interpolation | Medium |
| `2281f1f4` | Sparse to dense fill | Medium |
| `228f6490` | Diagonal shape continuation | Medium |
| `2bee17df` | Extract rectangular patch | Medium |
| `3428a4f5` | Replace one color | Medium |

---

## Execution Order

### Session 1 (Fresh Context)
1. **Re-golf `0e206a2e`** (1384 bytes) - biggest potential gain
   - Read task JSON fresh
   - Analyze current algorithm
   - Brainstorm completely different approaches
   - Target: <700 bytes

### Session 2 (Fresh Context)
2. **Re-golf `2dd70a9a`** (1163 bytes)
   - Same approach as above
   - Target: <700 bytes

### Session 3 (Fresh Context)
3. **Quick scan of 500-600 byte solutions**
   - `045e512c`, `0a938d79`, `a64e4611`
   - Apply recent tricks (walrus, tuple iteration, 1D array, etc.)
   - 10-15 min per task, move on if no quick win

### Session 4-5 (Fresh Context each)
4. **New tasks batch** (2-3 per session)
   - Pick from Medium difficulty analyzed tasks
   - Clear context between batches

---

## Success Metrics

| Metric | Current | Target |
|--------|---------|--------|
| Total Score | 89,131 | 92,000+ |
| Avg Score/Task | 2,174 | 2,200+ |
| V.Hard Avg | 1,540 | 1,700+ |

**Key goal**: Reduce `0e206a2e` and `2dd70a9a` by 400+ bytes each = +1,300 pts

---

## Commands Reference

```bash
cd /Users/aerickson/Documents/Claude\ Code\ Projects/agentic-evolve/showcase/code-golf

# Test a solution
python3 evaluator.py <task_id> <task_id>/solution.py

# View task
cat tasks/<task_id>.json | python3 -m json.tool

# Check current solution
cat <task_id>/solution.py && wc -c <task_id>/solution.py
```

---

## Resume Prompts

### For Re-Golf Session:
```
Re-golf ARC task <task_id>. Current solution is <bytes> bytes.
Read tasks/<task_id>.json and <task_id>/solution.py.
Look for a completely different algorithm approach.
Target: reduce by 200+ bytes.
```

### For New Task Session:
```
Solve ARC code golf tasks. Pick 2-3 from the Medium difficulty analyzed list.
Working directory: /Users/aerickson/Documents/Claude Code Projects/agentic-evolve/showcase/code-golf
Clear context between complex tasks.
```
