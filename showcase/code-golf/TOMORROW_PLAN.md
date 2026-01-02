# Tomorrow's Plan: Code Golf Optimization

## Session Strategy (Combat Context Rot)

| Rule | Rationale |
|------|-----------|
| **ONE task per session** | Context availability = evolution effectiveness |
| **Clear after every task** | Fresh context for maximum gains |
| **Largest first** | More bytes = more room for improvement |
| **Full evolution always** | 5-10 gens, 3-4 mutations each (AlphaEvolve-inspired) |

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
| `150deff5` | ~~684~~ **494** | ✅ DONE! README + re-golf (-28%, +190 pts) |
| `1a07d186` | ~~635~~ **434** | ✅ DONE! README + re-golf (-32%, +201 pts) |
| `1b60fb0c` | ~~933~~ **1,026** | ✅ DONE! Fixed broken solution + re-golf (-30%, +430 pts) |

#### Hard (300-600 bytes) - Document + Try Re-golf
| Task | Bytes | Status |
|------|-------|--------|
| `0a938d79` | ~~539~~ **237** | ✅ DONE! README + re-golf (-56%, +302 pts) |
| `1bfc4729` | ~~406~~ **108** | ✅ DONE! README + re-golf (-73%, +298 pts) |
| `11852cab` | ~~333~~ **280** | ✅ DONE! README + evolution (-16%, +53 pts) |
| `178fcbfb` | ~~304~~ **217** | ✅ DONE! README + evolution (-29%, +87 pts) |

#### Medium (100-300 bytes) - Document + Evolution (REQUIRED)
| Task | Bytes | Status | Target |
|------|-------|--------|--------|
| `1caeab9d` | ~~280~~ **207** | ✅ DONE! README + evolution (-26%, +73 pts) | ~220 |
| `1e32b0e9` | ~~207~~ **201** | ✅ DONE! README + evolution (-3%, +6 pts) | ~170 |
| `363442ee` | ~~205~~ **144** | ✅ DONE! README + evolution (-30%, +61 pts) | ~170 |
| `1190e5a7` | ~~188~~ **124** | ✅ DONE! README + evolution (-34%, +64 pts) | ~150 |
| `10fcaaa3` | ~~176~~ **174** | ✅ DONE! README + evolution (-1%, +2 pts) | ~140 |
| `1b2d62fb` | ~~170~~ **58** | ✅ DONE! README + evolution (-66%, +112 pts) | ~140 |
| `1cf80156` | ~~138~~ **130** | ✅ DONE! README + evolution (-6%, +8 pts) | ~110 |

**PHASE 0 COMPLETE! All Medium tier tasks documented and evolved.**
**Total Phase 0 gain: +1,947 pts**

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
| `0e206a2e` | ~~1,384~~ **1,135** | ✅ | **+249 pts** | ✅ DONE! 12 gens, 48 mutations |
| `2dd70a9a` | ~~1,163~~ **673** | ✅ | **+490 pts** | ✅ DONE! |
| `1b60fb0c` | ~~933~~ **1,026** | ✅ | **+430 pts** | ✅ DONE! Fixed broken + 18 gens evolution |

**Strategy**: Fresh session per task. Document → Understand → Rethink algorithm.

---

## Phase 2: Re-Golf Hard/Medium (Medium Priority)

| Task | Current | Target | Potential Gain | Notes |
|------|---------|--------|----------------|-------|
| `045e512c` | ~~591~~ **486** | ✅ | **+105 pts** | ✅ DONE! 15 gens evolution |
| `0a938d79` | ~~539~~ **237** | ✅ | **+302 pts** | ✅ DONE! |
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

### Session 2: Document + Re-golf `150deff5` ✅ COMPLETE
- ~~684 bytes~~ → **494 bytes** (-28%)
- README.md written with 7 evolution steps documented
- Key breakthrough: bitmask enumeration + non-overlap via len check
- **+190 points gained!**

### Session 3: Document + Re-golf `1a07d186` ✅ COMPLETE
- ~~635 bytes~~ → **434 bytes** (-32%)
- README.md written
- **+201 points gained!**

### Session 4: Hard tier batch ✅ COMPLETE
- ~~`0a938d79`~~ ✅, ~~`1bfc4729`~~ ✅, ~~`11852cab`~~ ✅, ~~`178fcbfb`~~ ✅
- All 4 tasks re-golfed with evolution (AlphaEvolve-inspired)
- All READMEs written
- **+740 pts gained in Session 4**

### Sessions 5-11: Medium tier (ONE TASK PER SESSION)

**IMPORTANT: One task per session, fresh context each time.**
Context availability directly correlates with evolution effectiveness.

**Each session:**
1. Read task JSON + current solution
2. Write README.md
3. Run evolution (5-10 gens, 3-4 mutations each)
4. Update README with evolution summary
5. Commit, push, clear context

**Queue (largest first for maximum gains):**
| Session | Task | Bytes | Target | Est. Gain |
|---------|------|-------|--------|-----------|
| 5 | `1caeab9d` | ~~280~~ **207** | ✅ DONE | **+73 pts** |
| 6 | `1e32b0e9` | ~~207~~ **201** | ✅ DONE | **+6 pts** |
| 7 | `363442ee` | ~~205~~ **144** | ✅ DONE | **+61 pts** |
| 8 | `1190e5a7` | ~~188~~ **124** | ✅ DONE | **+64 pts** |
| 9 | `10fcaaa3` | ~~176~~ **174** | ✅ DONE | **+2 pts** |
| 10 | `1b2d62fb` | ~~170~~ **58** | ✅ DONE | **+112 pts** |
| 11 | `1cf80156` | ~~138~~ **130** | ✅ DONE | **+8 pts** |

### Session 5: Document + Evolve `1caeab9d` ✅ COMPLETE
- ~~280 bytes~~ → **207 bytes** (-26%)
- 10 generations, ~40 mutations tested
- Key breakthrough: P[0][0] trick (row-order collection = sorted)
- **+73 points gained!**

### Session 6: Document + Evolve `1e32b0e9` ✅ COMPLETE
- ~~207 bytes~~ → **201 bytes** (-3%)
- 10 generations, ~40 mutations tested
- Key breakthrough: Inline `gc=g[5][0]` variable
- Solution hit plateau - well-optimized template overlay algorithm
- **+6 points gained!**

### Session 7: Document + Evolve `363442ee` ✅ COMPLETE
- ~~205 bytes~~ → **144 bytes** (-30%)
- 10 generations, ~40 mutations tested
- Key breakthroughs: Inline p, slice assignment, `i-i%3`, exec in listcomp, one-liner
- Exceeded target by 26 bytes!
- **+61 points gained!**

### Session 8: Document + Evolve `1190e5a7` ✅ COMPLETE
- ~~188 bytes~~ → **124 bytes** (-34%)
- 11 generations, ~44 mutations tested
- Key breakthroughs: `min(map(max,g))` for separator, `{*g[0]}` for colors, `-~`, `g[:n]` loop
- Exceeded target by 26 bytes!
- **+64 points gained!**

### Session 9: Document + Evolve `10fcaaa3` ✅ COMPLETE
- ~~176 bytes~~ → **174 bytes** (-1%)
- 10 generations, ~40 mutations tested
- Key breakthrough: range(H*2) with modular lookup vs enumerate(g*2)
- Solution near plateau - already well-optimized modular tiling algorithm
- **+2 points gained!**

### Session 10: Document + Evolve `1b2d62fb` ✅ COMPLETE
- ~~170 bytes~~ → **58 bytes** (-66%)
- 8 generations, ~32 mutations tested
- Key breakthrough: Bit shift trick `8>>sum` - when both values are 0, 8>>0=8; otherwise 8>>9=0
- MASSIVELY exceeded target! Expected ~30 pts, got **+112 pts**
- **+112 points gained!**

### Session 11: Document + Evolve `1cf80156` ✅ COMPLETE
- ~~138 bytes~~ → **130 bytes** (-6%)
- 10 generations, ~40 mutations tested
- Key breakthrough: E=enumerate alias (saves 4 bytes when used twice)
- Solution near plateau - bounding box extraction is already compact
- **+8 points gained!**

### Session 12: Re-golf `0e206a2e` ✅ COMPLETE
- ~~1384 bytes~~ → **1135 bytes** (-18%)
- 12 generations, ~48 mutations tested
- Key breakthroughs: `all()` for matching, inline transforms, DFS pop(), `_ in g`
- **+249 points gained!**

### Session 14: Fix + Re-golf `1b60fb0c` ✅ COMPLETE
- Original: 933 bytes (BROKEN - 0/4 tests)
- Fixed: 1456 bytes (4/4 tests)
- Golfed: **1026 bytes** (-30%)
- 18 generations, ~72 mutations tested
- Key breakthrough: segment detection not needed, `sy*rg>sy` trick, combined while loops
- **+430 points gained!** (from fixed baseline)

### Session 15+: Continue with Hard tier or new tasks

---

## Success Metrics

| Metric | Current | Target |
|--------|---------|--------|
| READMEs Complete | **41/41** | 41/41 ✅ |
| Total Score | 91,339 | 92,000+ |
| Avg Score/Task | 2,228 | 2,250+ |
| Points Gained | **+2,731** | +2,000+ ✅ |

### Remaining Potential (Phase 2)
| Source | Tasks | Est. Gain |
|--------|-------|-----------|
| V.Hard re-golf | 0 | ✅ All done! |
| Hard re-golf | 1 | ~123 pts |
| **Total remaining** | **1** | **~123 pts** |

### Points Breakdown
- `2dd70a9a`: +490 pts (1163→673)
- `1b60fb0c`: +430 pts (1456→1026) **18 gens, 72 mutations** (was BROKEN, fixed & golfed!)
- `0a938d79`: +302 pts (539→237)
- `0e206a2e`: +249 pts (1384→1135) **12 gens, 48 mutations** (Phase 1)
- `1bfc4729`: +298 pts (406→108)
- `1a07d186`: +201 pts (635→434)
- `150deff5`: +190 pts (684→494)
- `1b2d62fb`: +112 pts (170→58) **8 gens, 32 mutations** (MASSIVELY exceeded target!)
- `045e512c`: +105 pts (591→486) **15 gens, 60 mutations** (Phase 2)
- `178fcbfb`: +87 pts (304→217) **10 gens, 40 mutations**
- `1caeab9d`: +73 pts (280→207) **10 gens, 40 mutations**
- `1190e5a7`: +64 pts (188→124) **11 gens, 44 mutations** (exceeded target!)
- `363442ee`: +61 pts (205→144) **10 gens, 40 mutations** (exceeded target!)
- `11852cab`: +53 pts (333→280) **10 gens, 40 mutations**
- `1cf80156`: +8 pts (138→130) **10 gens, 40 mutations** (near plateau)
- `1e32b0e9`: +6 pts (207→201) **10 gens, 40 mutations** (hit plateau)
- `10fcaaa3`: +2 pts (176→174) **10 gens, 40 mutations** (near plateau)
- **Phase 0 total: +1,947 pts** ✅ COMPLETE
- **Phase 1 (0e206a2e): +249 pts** ✅ COMPLETE
- **Phase 1 (1b60fb0c): +430 pts** ✅ COMPLETE (fixed broken + golfed)
- **Phase 2 (045e512c): +105 pts** ✅ COMPLETE
- **GRAND TOTAL: +2,731 pts** ✅ TARGET EXCEEDED!

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

### For Document + Re-Golf Session (WITH EVOLUTION):
```
Document and re-golf ARC task <task_id>. Current solution is <bytes> bytes (<tier> tier, missing README).

Steps:
1. Read tasks/<task_id>.json - understand the pattern
2. Read <task_id>/solution.py - understand current approach
3. Write <task_id>/README.md following CONTRIBUTING.md template
4. Run evolution (AlphaEvolve-inspired):
   - Create .evolve/<task_id>/mutations/ directory
   - Run 5-10 generations with 3-4 mutations each
   - Track all results, document learnings
   - Update README with evolution summary

Target: reduce by 10-20% if possible.
Working directory: /Users/aerickson/Documents/Claude Code Projects/agentic-evolve/showcase/code-golf

Reference: See CONTRIBUTING.md for evolution requirements, CLAUDE.md for workflow.
```

### For Documentation Batch Session:
```
Document missing READMEs for these ARC tasks: <task1>, <task2>, <task3>

For each task:
1. Read tasks/<task_id>.json
2. Read <task_id>/solution.py
3. Write <task_id>/README.md with pattern description and algorithm explanation
4. For 200+ byte solutions: Run evolution (min 5 generations)
5. Quick golf check for smaller solutions

Working directory: /Users/aerickson/Documents/Claude Code Projects/agentic-evolve/showcase/code-golf
Reference: See CONTRIBUTING.md and CLAUDE.md for workflow requirements.
```

### Key Evolution Insights (from 11852cab)
- Variable elimination often beats variable reuse
- `x>>1` saves 2 bytes over `(x)//2`
- Failed approaches are valuable data - document them
- Crossover compounds: combine tricks from different generations
