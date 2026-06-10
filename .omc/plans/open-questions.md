# Open Questions

## ralplan-aicb-comm-domain - 2026-06-02

- [ ] **`embedding_group` semantics** -- How does `CommGroup.embedding_group` differ from `tp_group`? In Megatron, embedding layers use TP group for communication. The mapping utility needs the correct behavior. Affects Step 2 (RankMapper).
- [ ] **EP + DP interaction for `dp_group`** -- Existing code uses `dp_num = world_size // (tp * pp)` which does not subtract EP ranks. Should `dp_group` ranks reflect the full DP group or DP modulo EP? This is inherited behavior; need to decide whether to preserve or fix.
- [ ] **`cp` (context parallelism) support** -- `RankGenerator` supports `cp` in the order string but no generator uses it. Should the mapping utility add `cp_group` preemptively?

## lld-v2-to-v3-migration - 2026-06-04

- [ ] **Real v3.0.260530 lld.json sample** -- Can we get a real v3 lld.json to validate the fixture format assumptions? The plan assumes `port_id` format `(\d+)GE(\d+)/(\d+)/(\d+):(\d+)` and `chassis_topo` format `{NPU}_{DEPLOYMENT}`. A real sample would eliminate format-mismatch risk.
- [ ] **`node_id` semantic equivalence** -- Is `node_id` in v3 semantically identical to `node_ip` in v2 (same IP address values), or is it a different identifier format? If different, the EDG protocol cross-matching (which uses IP-based `node_ip` in responses) may break.
- [ ] **`chassis_topo` edge cases** -- Does `chassis_topo` always follow the `{NPU}_{DEPLOYMENT}` pattern? Any multi-NPU chassis formats that would break `split("_")[0]` extraction?
- [ ] **Existing workspace lld.json files** -- Workspace directories under `server/workspaces/` contain cached v2 lld.json files. After migration, `/api/edg/init` must be re-called to regenerate cached files. Should the migration include an auto-cleanup of stale workspace lld files, or is manual re-init sufficient?
