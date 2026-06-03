# Open Questions

## ralplan-aicb-comm-domain - 2026-06-02

- [ ] **`embedding_group` semantics** -- How does `CommGroup.embedding_group` differ from `tp_group`? In Megatron, embedding layers use TP group for communication. The mapping utility needs the correct behavior. Affects Step 2 (RankMapper).
- [ ] **EP + DP interaction for `dp_group`** -- Existing code uses `dp_num = world_size // (tp * pp)` which does not subtract EP ranks. Should `dp_group` ranks reflect the full DP group or DP modulo EP? This is inherited behavior; need to decide whether to preserve or fix.
- [ ] **`cp` (context parallelism) support** -- `RankGenerator` supports `cp` in the order string but no generator uses it. Should the mapping utility add `cp_group` preemptively?
