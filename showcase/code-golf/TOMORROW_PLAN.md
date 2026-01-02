# Tomorrow's Plan: Code Golf Optimization

## Session Strategy (Combat Context Rot)

| Rule | Rationale |
|------|-----------|
| **2-3 tasks per session** | Keep context under ~50K tokens |
| **Clear between complex tasks** | Fresh context for novel patterns |
| **Hardest first** | Use clean context for tricky problems |
| **Single-task for V.Hard** | Maximum attention on algorithm rework |

---

## Phase 0: Document Missing READMEs + Evolve (Priority!)

**15 solutions are missing README documentation.** For each:
1. Read the task JSON to understand the problem
2. Read the current solution to understand the approach
3. Write a README.md documenting the pattern and algorithm
4. Attempt to evolve/re-golf while context is fresh

### Missing READMEs by Priority

#### V.Hard (600+ bytes) - Document + Major Re-golf
| Task | Bytes | Status |
|------|-------|--------|
| `2dd70a9a` | ~~1,163~~ **673** | ✅ DONE! README + re-golf (-42%, +490 pts) |
| `1b60fb0c` | 933 | Missing README |
| `150deff5` | 684 | Missing README |
| `1a07d186` | 635 | Missing README |

#### Hard (300-600 bytes) - Document + Try Re-golf
| Task | Bytes | Status |
|------|-------|--------|
| `0a938d79` | 539 | Missing README, medium re-golf priority |
| `1bfc4729` | 406 | Missing README |
| `11852cab` | 333 | Missing README (batch 4) |
| `178fcbfb` | 304 | Missing README |

#### Medium (100-300 bytes) - Document + Quick Golf Check
| Task | Bytes | Status |
|------|-------|--------|
| `1caeab9d` | 280 | Missing README |
| `1e32b0e9` | 207 | Missing README (batch 4) |
| `363442ee` | 205 | Missing README |
| `1190e5a7` | 188 | Missing README (batch 4) |
| `10fcaaa3` | 176 | Missing README (batch 4) |
| `1b2d62fb` | 170 | Missing README |
| `1cf80156` | 138 | Missing README (batch 4) |

### README Template

```markdown
# Task [TASK_ID]

## Pattern
[One-line description of what the task does]

## Algorithm
[2-3 sentences explaining the approach]

## Key Tricks
- [Golf trick 1]
- [Golf trick 2]

## Evolution History
| Gen | Bytes | Change |
|-----|-------|--------|
| 1 | XXX | Initial solution |
```

---

## Phase 1: Re-Golf V.Hard (High Priority)

After documenting, these need major algorithm rework:

| Task | Current | Target | Potential Gain | Notes |
|------|---------|--------|----------------|-------|
| `0e206a2e` | 1,384 | ~600 | +784 pts | Rotated template (has README) |
| `2dd70a9a` | ~~1,163~~ **673** | ✅ | **+490 pts** | ✅ DONE! |
| `1b60fb0c` | 933 | ~600 | +333 pts | (needs README) |

**Strategy**: Fresh session per task. Document → Understand → Rethink algorithm.

---

## Phase 2: Re-Golf Hard/Medium (Medium Priority)

| Task | Current | Target | Potential Gain | Notes |
|------|---------|--------|----------------|-------|
| `045e512c` | 591 | ~400 | +191 pts | Template matching (has README) |
| `0a938d79` | 539 | ~400 | +139 pts | Fill pattern grid (needs README) |
| `a64e4611` | 523 | ~400 | +123 pts | Largest rectangle (has README) |

---

## Phase 3: New Tasks (Lower Priority)

Only after documentation is complete. Pick from Medium difficulty analyzed list.

---

## Execution Order

### Session 1: Document + Re-golf `2dd70a9a` ✅ COMPLETE
- ~~1163 bytes~~ → **673 bytes** (-42%)
- README.md written with 11 evolution steps documented
- Key tricks: walrus, tuple indexing, __setitem__, variable reuse
- **+490 points gained!**

### Session 2: Document + Re-golf `1b60fb0c` (933 bytes)
Same process as Session 1

### Session 3: Document batch (3-4 medium/hard tasks)
- `0a938d79`, `1bfc4729`, `11852cab`, `178fcbfb`
- Quick golf attempt on each
- Write READMEs

### Session 4: Document batch (remaining 7 medium tasks)
- All the 100-300 byte solutions
- Quick golf check
- Write READMEs

### Session 5+: New tasks or continued re-golf

---

## Success Metrics

| Metric | Current | Target |
|--------|---------|--------|
| READMEs Complete | 27/41 | 41/41 |
| Total Score | 89,621 | 92,000+ |
| Avg Score/Task | 2,186 | 2,200+ |
| V.Hard Avg | 1,638 | 1,700+ |

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

# Check for missing READMEs
for d in */; do [ -f "${d}solution.py" ] && [ ! -f "${d}README.md" ] && echo "$d"; done
```

---

## Resume Prompts

### For Document + Re-Golf Session:
```
Document and re-golf ARC task <task_id>. Current solution is <bytes> bytes.

1. Read tasks/<task_id>.json - understand the pattern
2. Read <task_id>/solution.py - understand current approach
3. Write <task_id>/README.md documenting the pattern and algorithm
4. Attempt to re-golf with fresh perspective

Target: reduce by 100+ bytes if possible.
Working directory: /Users/aerickson/Documents/Claude Code Projects/agentic-evolve/showcase/code-golf
```

### For Documentation Batch Session:
```
Document missing READMEs for these ARC tasks: <task1>, <task2>, <task3>

For each task:
1. Read tasks/<task_id>.json
2. Read <task_id>/solution.py
3. Write <task_id>/README.md with pattern description and algorithm explanation
4. Quick golf check - apply any obvious tricks

Working directory: /Users/aerickson/Documents/Claude Code Projects/agentic-evolve/showcase/code-golf
```
