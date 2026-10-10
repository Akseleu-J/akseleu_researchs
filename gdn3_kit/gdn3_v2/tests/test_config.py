import pytest
from gdn3.config import GDN3Config

def test_baseline_contract():
    c=GDN3Config.baseline();assert c.ha==0 and c.memory=="none"

def test_invalid_sse_rejected():
    with pytest.raises(ValueError):GDN3Config(sse_top_k=2,sse_partitions=1)
