# RALPLAN: Add Communication Domain (Rank) Information to AICB LogItem CSV Output

**Date:** 2026-06-02 | **Revision:** v4 (Critic R2 fixes)
**Mode:** SHORT
**Review Iteration:** 3/5 — Critic R2(REVISE: test+example inconsistencies) → Plan v4

---

## 1. RALPLAN-DR Summary

### Principles (top 5)

1. **Add-only** — no existing LogItem fields, columns, or CSV format elements are removed or reordered. New fields appended at end.
2. **Generator-owned correctness** — ranks populated at LogItem creation time where possible; fallback post-processing fills model-generated LogItems with logging warnings.
3. **Explicit rank identity** — each communication row carries the full list of participating rank IDs, not just a group label.
4. **Verifiable** — independent verification script cross-checks CSV output against RankGenerator output.
5. **Backward compatible** — rank mapping info in a sidecar file (`_rank_mapping.csv`), NEVER inline in the workload CSV. Existing CSV consumers (visualize/generate.py) are NOT broken.

### Decision Drivers (top 3)

1. **Accuracy** — ranks must exactly match the mathematical rank decomposition produced by `RankGenerator.get_ranks()`.
2. **Coverage** — all four workload generators (Megatron, DeepSpeed S1/2, DeepSpeed S3, collective_test) and all mocked models must produce correct ranks.
3. **Testability** — the verification script must detect mismatches without manual inspection.

### Viable Options (3)

| # | Option | Pros | Cons |
|---|--------|------|------|
| A | **Pass ranks at each LogItem constructor call** | Most explicit; easiest to debug; no post-processing magic | Requires touching ~80+ LogItem construction sites across generators and mocked models |
| B | **Auto-populate ranks in Workload via CommGroup lookup at dump time** | Minimal code changes (~3 files); mocked models need zero changes | LogItem.ranks is None during the object lifetime; non-obvious behavior |
| C | **Hybrid: generators set ranks explicitly; Workload fills missing ranks for mocked-model LogItems** | Best of both; generators are explicit; model code is untouched; clear separation of concerns | Slightly more complex; two different population paths; requires deprecation warnings on fallback path |

### Selected: Option C (Hybrid)

**Why Option C:**
- The mocked models (~15 classes across `aicb/workload_generator/mocked_model/`) create LogItems inside `.forward()` and `.backward()` methods that lack access to the global RankGenerator.
- Workload generators create LogItems directly in their init/forward/backward/step methods and can carry a RankGenerator reference.
- The dump-time fallback path emits a `logging.warning()` each time it fires, making maintenance debt visible.

**Why not A:** Invasive changes across the model hierarchy. Mocked models have ~25+ LogItem construction sites that omit `comm_group_size` entirely (systematic, not incidental). Threading RankGenerator through the entire model hierarchy introduces coupling.

**Why not B:** LogItems would carry `ranks=None` for most of their lifetime. Violates principle #3 "Explicit rank identity."

---

## 2. Architecture Design

### 2.1 RankGenerator Integration

The `RankGenerator` class (`aicb/utils/utils.py:141-220`) already computes orthogonal rank groups.

```
get_ranks(token: str, independent_ep: bool = False) -> List[List[int]]
```

Returns a list of rank groups, each being a list of GPU rank IDs. For example, `get_ranks('tp')` with tp=2, dp=4 returns 8 groups of 2 ranks each: `[[0,1], [2,3], [4,5], [6,7], ...]`.

**RankGenerator construction** uses the `order` string. The only existing construction in the codebase is at `workload_applyer.py:105`, hardcoded as `'tp-cp-ep-dp-pp'`. We use this as the default fallback.

### 2.2 CommGroup to Rank Token Mapping

| CommGroup | get_ranks token | independent_ep | Notes |
|-----------|----------------|----------------|-------|
| `tp_group` | `"tp"` | `False` | Standard TP group |
| `pp_group` | `"pp"` | `False` | Standard PP group |
| `dp_group` | `"dp"` | `False` | Full DP groups |
| `ep_group` | `"ep"` | `True` | EP only exists in `order_w_ep` |
| `ep_dp_group` | `"ep-dp"` | `True` | Cross-EP-DP group |
| `ep_tp_group` | `"ep-tp"` | `True` | Cross-EP-TP group |
| `embedding_group` | `"tp"` | `False` | Fallback to TP group (see Open Question 1) |
| `all` | N/A | N/A | `list(range(world_size))` — ALWAYS world_size regardless of LogItem's `comm_group_size` |
| `cp_group` | `"cp"` | `False` | Context parallelism (TODO: no generator currently uses this) |

**Verified correct** for tp=2, dp=4, pp=1:
- `get_ranks('tp')` → rank 0's tp_group = `[0,1]` ✓
- `get_ranks('dp')` → rank 0's dp_group = `[0,2,4,6]` ✓

### 2.3 Data Flow

```
RankGenerator(config)
       │
       ├─► rank_mapper.get_rank_list_for_comm_group(generator, comm_group, comm_group_size, ref_rank=0)
       │         │
       │         ▼
       │    List[int]  (e.g., [0, 2, 4, 6] for dp_group containing rank 0)
       │
       ├─► WorkloadGenerator subclasses
       │    └─► LogItem(..., ranks=rank_mapper.get_rank_list_for_comm_group(...))
       │
       └─► Workload.dump()
            │
            ├─► Post-processing pass: for LogItems with ranks=None:
            │     - Fill ranks via rank_mapper.get_rank_list_for_comm_group()
            │     - Fill comm_group_size from len(ranks) if comm_group_size is None
            │     - Emit logging.warning("Rank auto-populated for LogItem at dump time...")
            │     - Skip LogItems with comm_group=None (computation LogItems)
            │
            ├─► Write main CSV (unchanged format + trailing ranks column)
            │
            └─► Write sidecar _rank_mapping.csv (rank group decomposition table)
```

### 2.4 CSV Format

**Main CSV** (`workload.csv`): UNCHANGED format except one new `ranks` column appended at end:

```csv
comm_type,comm_group,...,count,ranks
all_reduce,dp_group,...,1,"0,2,4,6"
broadcast,tp_group,...,1,"0,1"
computation,,...,1,
```

The `ranks` column is ALWAYS double-quoted per RFC 4180 (consistency even for single-element groups). Empty for computation LogItems (`comm_group=None`).

**Sidecar rank mapping** (`workload_rank_mapping.csv`): Separate file in same directory:

```csv
group,size,rank_groups
tp_group,2,"[0,1] [2,3] [4,5] [6,7]"
dp_group,4,"[0,2,4,6] [1,3,5,7]"
pp_group,1,"[0] [1] [2] [3] [4] [5] [6] [7]"
ep_group,1,"[0] [1] [2] [3] [4] [5] [6] [7]"
```

When pp=1/ep=1, `get_ranks()` returns single-element groups (one per rank). Each rank is its own pipeline/expert stage. The pp=1 "all ranks" special case was removed in v3.

This is a clean CSV that any tool can parse with `pd.read_csv()`. No `#`-prefixed comments needed.

### 2.5 Which Rank's Group?

When a LogItem has `comm_group=dp_group` and there are 2 DP groups `[[0,2,4,6], [1,3,5,7]]`, which group's ranks go into the CSV row? The workload is generated from the perspective of a specific rank (typically 0 in `workload_only` mode). The utility function takes a `ref_rank` parameter (default 0) and returns the group that contains that rank.

---

## 3. Implementation Steps

### Step 0: Create test infrastructure

**Files:** New directory `aicb/tests/` with `__init__.py`

**Changes:**
1. Create `aicb/tests/__init__.py` (empty)
2. Project uses pytest (consistent with CLAUDE.md: `python3 -m pytest tests/ -v`)

### Step 1: Add `ranks` field to LogItem and update CSV serialization

**Files:** `aicb/log_analyzer/log.py`

**Changes:**
1. Add `ranks: list = dataclasses.field(default=None)` to the `LogItem` dataclass (after `count`).
2. Modify `view_as_csv_line()` to conditionally format the `ranks` field:
   ```python
   def view_as_csv_line(self):
       parts = []
       for k in self.__dict__.keys():
           v = getattr(self, k)
           if k == 'ranks':
               if v is not None:
                   parts.append('"' + ','.join(str(r) for r in v) + '"')
               else:
                   parts.append('')
           else:
               parts.append(str(v))
       return ','.join(parts)
   ```
   This replaces the current one-liner `",".join([str(getattr(self, k)) for k in self.__dict__.keys()])` at `log.py:74-75`. The key behavior: `ranks=None` → empty string `""` (NOT `"None"`); `ranks=[0,1]` → `"0,1"` (always double-quoted).
3. `csv_header()` automatically includes `ranks` via `__dict__` iteration (no change needed).

**Acceptance criteria:**
- `LogItem(comm_type=..., ranks=[0,1,2,3]).view_as_csv_line()` produces `...,\"0,1,2,3\"`
- `LogItem(comm_type=..., ranks=None).view_as_csv_line()` produces `...,`
- `csv_header()` ends with `,ranks`

### Step 2: Create CommGroup-to-ranks mapping utility

**Files:** New file `aicb/utils/rank_mapper.py`

**Changes:**
1. Create function `get_rank_list_for_comm_group(rank_generator: RankGenerator, comm_group: CommGroup, comm_group_size: int | None, ref_rank: int = 0) -> List[int]`.
2. Mapping logic (per table in Section 2.2):
   - Use `rank_generator.get_ranks(token, independent_ep)` for each token mapping
   - From the returned `List[List[int]]`, find the group that contains `ref_rank`
   - For `CommGroup.all` → `list(range(rank_generator.world_size))` (IGNORES comm_group_size — document that existing DeepSpeed generators use `dp_world_size` for barriers, but all world ranks participate)
   - For `CommGroup.embedding_group` → fall back to `tp_group` behavior
   - For `CommGroup.pp_group` → use `get_ranks('pp')`; returns the group containing `ref_rank`. When pp=1, each rank is its own group, so this returns `[ref_rank]` (a single-element list, matching pp point-to-point semantics). Do NOT special-case pp=1 to return all ranks.
   - For unknown CommGroup values → raise `ValueError(f"Unknown CommGroup: {comm_group}")`
3. Create function `build_rank_mapping_table(rank_generator: RankGenerator) -> List[Dict]` that returns all group decompositions for the sidecar file.
4. Handle edge case: if `ref_rank` not found in any group → raise `ValueError`.

**Acceptance criteria:**
- Unit test: tp=2, dp=4, pp=1: dp_group for rank 0 = `[0,2,4,6]`, tp_group = `[0,1]`
- Unit test: `CommGroup.all` returns `list(range(world_size))` regardless of comm_group_size
- Unit test: unknown CommGroup → ValueError raised
- Unit test: rank not in any group → ValueError raised

### Step 3: Integrate RankGenerator into WorkloadGenerator base class

**Files:** `aicb/workload_generator/workload_generator.py`

**Changes:**
1. Add class-level constant: `DEFAULT_ORDER = 'tp-cp-ep-dp-pp'` (matches `workload_applyer.py:105`).
2. In `WorkloadGenerator.__init__()`:
   - Derive `order = getattr(args, 'order', None) or WorkloadGenerator.DEFAULT_ORDER`
   - Construct `self.rank_generator = RankGenerator(tp=args.tensor_model_parallel_size, ep=getattr(args, 'expert_model_parallel_size', 1), dp=args.dp_num, pp=args.pipeline_model_parallel, cp=getattr(args, 'context_parallel_size', 1), order=order)`
3. Add convenience method `self.get_ranks(comm_group, comm_group_size)` that delegates to `rank_mapper.get_rank_list_for_comm_group()`.
4. Set `self.workload.rank_generator = self.rank_generator` for dump-time post-processing.

**Parameters note:** `get_params()` (`utils.py:598-700`) does NOT set `args.order`. The `DEFAULT_ORDER` fallback is used. If `--order` is added to `get_params()` later, `getattr(args, 'order', None)` picks it up automatically.

**Parameters bridge:**
- `args.tensor_model_parallel_size` → `RankGenerator.tp`
- `args.pipeline_model_parallel_size` → `RankGenerator.pp`
- `args.dp_num` → `RankGenerator.dp`
- `getattr(args, 'expert_model_parallel_size', 1)` → `RankGenerator.ep` (get_params calls it `expert_model_parallel_size`, RankGenerator calls it `ep`)

**Acceptance criteria:**
- All existing generators continue to work without modification
- `self.rank_generator` is populated and accessible in subclasses

### Step 4: Populate ranks in workload generators (EXPLICIT boundary rules)

**Generator-created LogItems** (modify these — generators have RankGenerator access):

| File | Method(s) | Approx sites |
|------|-----------|-------------|
| `generate_megatron_workload.py` | `init()`, `forward()`, `backward()`, `step()`, `with_pipeline_forward_backward()` | ~25 |
| `generate_deepspeed_stage1_2_workload.py` | `init()`, `_reduce_ipg_grads()`, `backward()`, `step()` | ~8 |
| `generate_deepspeed_stage3_workload.py` | `init()`, `_gather_param_*()`, `_reduce_param_*()`, `step()` | ~10 |
| `generate_collective_test.py` | `init()`, `step()` | ~4 |

At every `LogItem(...)` construction in these files, add: `ranks=self.get_ranks(comm_group, comm_group_size)`

**Mocked-model LogItems** (do NOT modify — use dump-time fallback):

| File | Why excluded |
|------|-------------|
| `mocked_model/training/MockedMegatron.py` | ~5-6 LogItem sites (MoE permutation/unpermutation), no RankGenerator access, many omit comm_group_size |
| `mocked_model/training/MockedDeepSeek.py` | ~4-5 LogItem sites (MoE dispatch/combine), same pattern |
| `mocked_model/training/MockedDeepspeed.py` | 0 LogItem sites (class definitions only; all DeepSpeed LogItems are generator-created) |
| All files under `mocked_model/inference/` | Inference workload generators use different format (SimAI TXT) |

**Acceptance criteria:**
- All generator-created LogItems have `ranks` populated (not None)
- collective_test with tp=2, dp=4 produces correct dp_group ranks
- Mocked model LogItems still have `ranks=None` (handled by Step 5)

### Step 5: Add dump-time post-processing + sidecar rank mapping

**Files:** `aicb/log_analyzer/log.py`

**Changes to `Workload.dump()`:**
1. **Guard for missing rank_generator**: `rank_gen = getattr(self, 'rank_generator', None)`. If `rank_gen is None` (e.g., Workload created outside a generator), emit `logging.warning("No rank_generator available; skipping rank population and sidecar.")`, write the CSV without the `ranks` column or sidecar file, and return. This ensures graceful degradation for non-generator code paths.
2. **Before writing data rows**: No change to CSV format. Rank mapping goes to sidecar file.
2. **Per-row post-processing**: For each LogItem in `self.workload`:
   a. If `comm_group is None` → computation LogItem, leave `ranks` as None, emit `""` in CSV.
   b. If `ranks` is not None → use as-is (generator-populated).
   c. If `ranks` is None AND `comm_group` is not None → fill via `rank_mapper.get_rank_list_for_comm_group()`, emit `logging.warning("Rank auto-populated for LogItem stage={stage}, comm_group={comm_group}. Consider passing ranks explicitly.")`.
   d. If `comm_group_size` is None → fill from `len(ranks)` after rank lookup.
3. **After writing data rows**: Write sidecar file `{result_path}_rank_mapping.csv` with the rank decomposition table (format per Section 2.5).

**Note:** If two workload generations produce identically named output files, the second run overwrites the first's sidecar. For SimAI's typical workflow (single workload per run, output to timestamped directories) this is low risk.

**Changes to `Log.dump()`:**
- `Log.dump()` (`log.py:200-217`) also uses `csv_header()` / `view_as_csv_line()`. The `ranks` column automatically appears via the shared methods. Runtime comm logs (which don't populate ranks) will also gain the column — it will be empty, which is expected behavior. No code changes to `Log.dump()` are needed.

**Sidecar file format** (`_rank_mapping.csv`):
```csv
group,size,rank_groups
tp_group,2,"[0,1] [2,3] [4,5] [6,7]"
dp_group,4,"[0,2,4,6] [1,3,5,7]"
pp_group,1,"[0] [1] [2] [3] [4] [5] [6] [7]"
```

**Acceptance criteria:**
- Main CSV unchanged except `ranks` column at end
- Sidecar `_rank_mapping.csv` written alongside workload CSV with correct group decompositions
- `pd.read_csv('workload.csv')` works without errors (new `ranks` column appears, may need `quotechar` aware reader)
- All LogItem rows have non-empty `ranks` column (except computation-only rows)
- Deprecation warnings emitted when dump-time fallback fires
- Rank mapping sidecar is human-readable and machine-parseable

### Step 6: Create independent verification script

**Files:** New file `aicb/tests/verify_ranks.py`

**Changes:**
1. Create verification script that:
   a. Takes CSV file path and RankGenerator parameters as CLI args
   b. Parses the sidecar `_rank_mapping.csv` to load reference group decomposition
   c. Parses each data row from main CSV, extracts `comm_group` and `ranks`
   d. For rows with `comm_group` set: recomputes expected ranks using `RankGenerator` with same parameters
   e. Reports mismatches: file, line number, comm_group, expected vs actual ranks
   f. Skips rows with `comm_group=None` (computation) or `ranks=""` (empty)
   g. For `CommGroup.all` rows: verifies `ranks == list(range(world_size))`, does NOT compare against `comm_group_size`
2. Exit code 0 on success, non-zero on any mismatch.

**Acceptance criteria:**
- Run: `python -m pytest aicb/tests/verify_ranks.py --csv results/mocked_workload/megatron_*_workload.csv --tp 2 --dp 4 --pp 1`
- Returns exit code 0 for correctly generated workloads
- Detects: wrong rank IDs, missing ranks, rank count mismatch with group decomposition

### Step 7: Update visualize/generate.py for ranks column (if needed)

**Files:** `aicb/visualize/generate.py`

**Changes:**
1. Verify `custom_csv_reader()` works with the new `ranks` column. Since the new column is appended at end, existing column indexing (by header name via `dict(zip(header, row))`) should be unaffected.
2. Add a smoke test: read a workload CSV with ranks column, verify the dict keys include `ranks`.

**Note:** Since we use a sidecar file (NOT `#`-prefixed comments in the CSV), `custom_csv_reader`'s `header = next(reader)` logic is NOT broken. The CSV format is unchanged except for the trailing column.

---

## 4. Risk Assessment (Revised after Architect/Critic review)

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| `CommGroup` enum values don't map cleanly to `RankGenerator` tokens | Medium | High | `embedding_group` falls back to TP; unknown groups raise ValueError (not silent); unit tests cover all mapped groups |
| Sidecar file adds deployment complexity (two files to manage) | Low | Low | Sidecar written alongside main CSV in same directory; filename derived from main CSV name; verification script reads both |
| Mocked model LogItems systematically omit `comm_group_size` (~9 sites across MockedMegatron + MockedDeepSeek) | **High** (revised from Low) | Medium | **Revised**: dump-time post-processing fills BOTH `ranks` AND `comm_group_size` when either is None; deprecation warnings guide future fixes |
| `RankGenerator` token for `dp_group` with EP might be incorrect | Medium | Medium | Inherited from existing code's `dp_num = world_size // (tp*pp)`; verification script flags mismatches |
| CSV quoting of rank lists breaks downstream parsers | Low | Low | Always double-quote ranks column; standard RFC 4180; pandas handles this |
| Two-population-path maintenance debt | Medium | Low | Dump-time fallback emits `logging.warning`; verification script gates correctness; documented as transitional pattern |
| `CommGroup.all` with `dp_world_size` vs `world_size` mismatch | High (resolved) | Low | `CommGroup.all` ALWAYS returns `list(range(world_size))`; verification script skips `len(ranks) == comm_group_size` check for `all` groups; existing `comm_group_size=dp_world_size` values preserved (don't break existing consumers) |

---

## 5. Test Plan

### Unit Tests (`aicb/tests/test_rank_mapper.py`)

1. `test_tp_dp_pp_basic`: tp=2, dp=4, pp=1 → dp_group for rank 0 = [0,2,4,6]; tp_group = [0,1]; pp_group = [0] (single-element group, matching pp=1 point-to-point semantics)
2. `test_world_size_1`: world_size=1 → all groups = [0]
3. `test_ep_enabled`: tp=2, dp=8, ep=2 → verify ep_group, ep_dp_group, ep_tp_group decomposition
4. `test_embedding_group`: falls back to tp_group behavior
5. `test_comm_group_all`: ALWAYS returns `list(range(world_size))` regardless of comm_group_size argument
6. `test_unknown_comm_group`: raises ValueError
7. `test_ref_rank_not_found`: raises ValueError
8. `test_rank_mapping_table_build`: sidecar data format matches expected structure

### Unit Tests (`aicb/tests/test_logitem_ranks.py`)

9. `test_ranks_list_to_csv`: ranks=[0,1,2,3] → `"0,1,2,3"` (quoted)
10. `test_ranks_none_to_csv`: ranks=None → empty field
11. `test_ranks_empty_to_csv`: ranks=[] → `""`
12. `test_ranks_single_to_csv`: ranks=[0] → `"0"` (always quoted)

### Integration Tests

13. **End-to-end generator test**: Run each workload generator with small configs from the `aicb/` directory (required: `aicb/` uses relative imports, set `PYTHONPATH=.`):
    ```bash
    cd aicb && PYTHONPATH=. python -m workload_generator.generate_megatron_workload --world_size=8 --tensor_model_parallel_size=2 --epoch_num=1 --model_name=test
    cd aicb && PYTHONPATH=. python -m workload_generator.generate_deepspeed_stage1_2_workload --stage=2 --world_size=8 --epoch_num=1 --model_name=test
    cd aicb && PYTHONPATH=. python -m workload_generator.generate_deepspeed_stage3_workload --world_size=8 --epoch_num=1 --model_name=test
    cd aicb && PYTHONPATH=. python -m workload_generator.generate_collective_test --world_size=8 --epoch_num=1
    ```
    Test invocation: `cd aicb && python -m pytest tests/ -v` (matching existing CLAUDE.md pattern).
14. Run verification script against each output CSV

### Verification Script (`aicb/tests/verify_ranks.py`)

15. Cross-check every row's `ranks` column against `RankGenerator.get_ranks()` output for the corresponding `comm_group`
16. `len(ranks) == comm_group_size` for every row (SKIP for: `CommGroup.all`, `CommGroup.pp_group`, and `comm_group=None/comm_group_size=None`). Note: pp_group uses `comm_group_size=1` for point-to-point ops (irecv/isend between adjacent PP stages) but the ranks list contains the full pp group. This is a pre-existing inconsistency — verification should verify `ranks` is a valid subset of the pp group for the calling rank, not compare against `comm_group_size`.
17. Rank mapping sidecar matches RankGenerator decomposition

---

## 6. Rollback Plan

Add a feature flag `--no-ranks` to `get_params()` that suppresses rank population. When set:
- `WorkloadGenerator.__init__()` skips RankGenerator construction
- `Workload.dump()` skips rank column and sidecar file
- Existing behavior is fully preserved

This flag is documented but NOT implemented as part of this plan (the plan targets minimal viable delivery).

---

## 7. Open Questions

1. **`embedding_group` semantics** (from v1): How does `CommGroup.embedding_group` differ from `tp_group`? In Megatron, embedding layers use TP group for communication. We fall back to `tp_group` behavior until clarified. Test covers this case.

2. **EP + DP interaction for `dp_group`** (from v1): Existing code uses `dp_num = world_size // (tp * pp)` ignoring EP. This is an existing design choice preserved as-is.

3. **`cp_group` (context parallelism) support** (from v1): RankGenerator supports `cp` but no generator currently uses it. Mapping table entry added as TODO placeholder.

---

## 8. ADR (Architecture Decision Record)

**Decision:** Use hybrid rank population (Option C): generators set ranks explicitly; dump-time fills mocked-model LogItems with logging warnings. Rank mapping info in a sidecar file (`_rank_mapping.csv`), never inline in the CSV.

**Drivers:**
- Mocked models (~15 classes, ~25+ LogItem sites) systematically lack rank context AND omit `comm_group_size`
- Modifying all mocked models is high-effort and introduces coupling
- Sidecar file prevents breaking `visualize/generate.py` (`custom_csv_reader` does `header = next(reader)` without comment filtering)
- `get_params()` has no `--order` argument; we use `DEFAULT_ORDER = 'tp-cp-ep-dp-pp'` as fallback

**Alternatives considered:**
- Option A (explicit-only): rejected due to invasive changes across the model hierarchy
- Option B (post-process-only): rejected — LogItem should be self-describing during its lifetime
- Inline `#`-prefixed comments in CSV: rejected — breaks `visualize/generate.py:41-43`

**Consequences:**
- Two rank-population paths exist: explicit in generators, automatic in dump(). Dump-time path emits `logging.warning`.
- Verification script catches divergence between the two paths.
- If mocked models are refactored to carry RankGenerator, the dump-time fallback can be removed.

**Follow-ups:**
- Investigate `embedding_group` semantics (Open Question 1)
- Add `--order` CLI parameter to `get_params()` to eliminate the fallback
- Add `--no-ranks` rollback flag
- Add `cp_group` mapping when context parallelism support is added

---

## Review Iteration Log

### V3 → V4 Changes (Critic R2 feedback)

| # | Finding | Source | Fix Applied |
|---|---------|--------|-------------|
| C-1 | Test `test_tp_dp_pp_basic` expected pp_group=[0,1,2,3,4,5,6,7] contradicts Step 2 pp=1→[0] design | Critic R2 | Fixed: pp_group = [0] |
| C-2 | Sidecar example shows pp_group/ep_group as single big groups (stale from pre-V3) | Critic R2 | Fixed: pp=1 shows 8 single-element groups `[0]...[7]` |
| M-3 | `Workload.dump()` rank_generator access underspecified (no fallback for non-generator Workloads) | Critic R2 | Added `getattr(self, 'rank_generator', None)` guard with graceful skip |
| M-4 | Integration test commands missing working directory / PYTHONPATH | Critic R2 | Added `cd aicb && PYTHONPATH=.` prefix to all test commands |
| M-5 | MockedDeepspeed.py has 0 LogItem constructions (not ~5+) | Critic R2 | Corrected to "0 sites"; updated MockedMegatron+MockedDeepSeek counts to ~9 total |

### V2 → V3 Changes (Architect R2 feedback)

| # | Finding | Source | Fix Applied |
|---|---------|--------|-------------|
| P0-1 | `args.pipeline_model_parallel_size` typo (get_params uses `pipeline_model_parallel` without `_size`) | Architect R2 | Fixed to `args.pipeline_model_parallel` |
| P0-2 | pp=1 "return all ranks" special case semantically incorrect (creates 8 != 1 mismatch) | Architect R2 | Removed special case; pp=1 returns `[ref_rank]` matching point-to-point semantics |
| P1-3 | pp_group verification false positives (comm_group_size=1 for P2P, but ranks contains full group) | Architect R2 | Added `CommGroup.pp_group` to verification skip list with explanation of pre-existing inconsistency |
| P1-4 | `view_as_csv_line()` implementation underspecified (str(None) → "None", not "") | Architect R2 | Provided explicit code for conditional `ranks` key formatting |
| P2-5 | Sidecar filename collision risk | Architect R2 | Added note about uniqueness assumption in typical SimAI workflow |
| P2-6 | `Log.dump()` unexpected empty `ranks` column in runtime comm logs | Architect R2 | Added explicit documentation that empty column is expected |

### V1 → V2 Changes (Architect + Critic feedback)

| # | Finding | Source | Fix Applied |
|---|---------|--------|-------------|
| 1 | `#`-prefixed comments break `visualize/generate.py:41-43` | Architect (HIGH), Critic (CRITICAL) | Changed to sidecar `_rank_mapping.csv` file; zero CSV format change except trailing `ranks` column |
| 2 | No `order` string source for RankGenerator | Architect (HIGH), Critic (CRITICAL) | Added `DEFAULT_ORDER = 'tp-cp-ep-dp-pp'` fallback in Step 3; documented get_params() gap |
| 3 | `CommGroup.all` maps to `list(range(world_size))` but DeepSpeed uses `comm_group_size=dp_world_size` | Architect (MEDIUM), Critic (MAJOR) | Documented: `CommGroup.all` ALWAYS returns world_size ranks; verification skips `len(ranks)==comm_group_size` for `all` groups |
| 4 | EP group + broader `comm_group_size=None` in ~25+ mocked model sites | Architect (MEDIUM), Critic (MAJOR) | Dump-time post-processing fills BOTH `ranks` AND `comm_group_size` when None; risk likelihood revised to High |
| 5 | Two inconsistent rank mapping header formats (Section 2.4 vs Step 5) | Critic (Minor) | Resolved: single sidecar format defined in Section 2.4 and Step 5 |
| 6 | No `cp_group` mapping | Critic (Minor) | Added to mapping table as TODO |
| 7 | `Log.dump()` modifications underspecified | Critic (Minor) | Clarified: no explicit changes needed; `ranks` column auto-appears via shared `csv_header()/view_as_csv_line()` |
| 8 | No `embedding_group` test | Critic (Minor) | Added test_embedding_group to unit test plan |
| 9 | No test infrastructure (`aicb/tests/` doesn't exist) | Critic (Minor) | Added Step 0 to create test directory |
| 10 | No rollback plan | Critic (Minor) | Added Section 6: `--no-ranks` feature flag (documented, not implemented) |
| 11 | Two-population-path maintenance risk | Architect + Critic (Medium) | Added `logging.warning()` on fallback path; documented as transitional pattern |
| 12 | Unknown CommGroup silent failure risk | Architect + Critic | Added `ValueError` for unmapped CommGroup values |
