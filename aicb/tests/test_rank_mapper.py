"""Unit tests for rank_mapper.py — CommGroup-to-RankGenerator bridge."""
import pytest
from utils.utils import CommGroup, RankGenerator

def _make_gen(tp=1, dp=1, pp=1, ep=1, cp=1, order='tp-cp-ep-dp-pp'):
    return RankGenerator(tp=tp, ep=ep, dp=dp, pp=pp, cp=cp, order=order)

class TestGetRankListForCommGroup:
    def test_tp_dp_pp_basic(self):
        from utils.rank_mapper import get_rank_list_for_comm_group
        rg = _make_gen(tp=2, dp=4, pp=1)
        assert get_rank_list_for_comm_group(rg, CommGroup.tp_group, ref_rank=0) == [0, 1]
        assert get_rank_list_for_comm_group(rg, CommGroup.dp_group, ref_rank=0) == [0, 2, 4, 6]
        assert get_rank_list_for_comm_group(rg, CommGroup.pp_group, ref_rank=0) == [0]

    def test_world_size_1(self):
        from utils.rank_mapper import get_rank_list_for_comm_group
        rg = _make_gen(tp=1, dp=1, pp=1)
        for cg in [CommGroup.tp_group, CommGroup.dp_group, CommGroup.pp_group]:
            assert get_rank_list_for_comm_group(rg, cg) == [0]

    def test_comm_group_all(self):
        from utils.rank_mapper import get_rank_list_for_comm_group
        rg = _make_gen(tp=2, dp=4, pp=1)
        assert get_rank_list_for_comm_group(rg, CommGroup.all) == list(range(8))

    def test_embedding_group(self):
        from utils.rank_mapper import get_rank_list_for_comm_group
        rg = _make_gen(tp=2, dp=4, pp=1)
        assert get_rank_list_for_comm_group(rg, CommGroup.embedding_group, ref_rank=0) == [0, 1]

    def test_unknown_comm_group_raises(self):
        from utils.rank_mapper import get_rank_list_for_comm_group
        rg = _make_gen()
        with pytest.raises(ValueError, match="Unknown CommGroup"):
            get_rank_list_for_comm_group(rg, "nonexistent_group")

    def test_ref_rank_not_found_raises(self):
        from utils.rank_mapper import get_rank_list_for_comm_group
        rg = _make_gen(tp=2, dp=2, pp=1)
        with pytest.raises(ValueError, match="not found in any"):
            get_rank_list_for_comm_group(rg, CommGroup.tp_group, ref_rank=999)

    def test_comm_group_none(self):
        from utils.rank_mapper import get_rank_list_for_comm_group
        assert get_rank_list_for_comm_group(_make_gen(), None) == []

    def test_ep_enabled(self):
        from utils.rank_mapper import get_rank_list_for_comm_group
        rg = _make_gen(tp=1, dp=4, ep=2, order='tp-cp-ep-dp-pp')
        ranks = get_rank_list_for_comm_group(rg, CommGroup.ep_group, ref_rank=0)
        assert ranks == [0, 1]
        ranks = get_rank_list_for_comm_group(rg, CommGroup.ep_dp_group, ref_rank=0)
        assert len(ranks) == 4
        ranks = get_rank_list_for_comm_group(rg, CommGroup.ep_tp_group, ref_rank=0)
        assert len(ranks) == 2

class TestBuildRankMappingTable:
    def test_basic_table(self):
        from utils.rank_mapper import build_rank_mapping_table
        rg = _make_gen(tp=2, dp=4, pp=1)
        rows = build_rank_mapping_table(rg)
        assert len(rows) >= 5
        groups_by_name = {r['group']: r for r in rows}
        assert groups_by_name['all_nodes']['size'] == 8
        assert groups_by_name['tp_group']['size'] == 2
        assert groups_by_name['dp_group']['size'] == 4
