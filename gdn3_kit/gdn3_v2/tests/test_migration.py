import pytest
from gdn3.migration.statex import build_statex_plan

def test_statex_partition_contract():
    p=build_statex_plan([0],[[[0,1],[2,3]]],4,8,8);assert p.new_heads==2
    with pytest.raises(ValueError):build_statex_plan([0],[[[0,1],[1,2]]],4,8,8)
