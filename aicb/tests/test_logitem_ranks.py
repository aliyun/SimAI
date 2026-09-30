"""Unit tests for LogItem ranks field serialization."""
from log_analyzer.log import LogItem
from utils.utils import CommType, CommGroup

class TestLogItemRanks:
    def test_ranks_list_to_csv(self):
        li = LogItem(comm_type=CommType.all_reduce, ranks=[0, 1, 2, 3])
        assert '"0,1,2,3"' in li.view_as_csv_line()

    def test_ranks_none_to_csv(self):
        li = LogItem(comm_type=CommType.all_reduce, ranks=None)
        assert li.view_as_csv_line().endswith(',')

    def test_ranks_empty_to_csv(self):
        li = LogItem(comm_type=CommType.all_reduce, ranks=[])
        assert '""' in li.view_as_csv_line()

    def test_ranks_single_to_csv(self):
        li = LogItem(comm_type=CommType.all_reduce, ranks=[0])
        assert '"0"' in li.view_as_csv_line()

    def test_ranks_in_header(self):
        assert 'ranks' in LogItem(comm_type=CommType.all_reduce).csv_header()

    def test_default_ranks_is_none(self):
        assert LogItem(comm_type=CommType.all_reduce).ranks is None
