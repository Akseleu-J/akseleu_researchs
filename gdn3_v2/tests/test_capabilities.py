from gdn3.capabilities import report
from gdn3.config import GDN3Config

def test_capability_report_never_claims_validation():
    r=report(GDN3Config.baseline());assert r["validated"] is False and r["production_ready"] is False
    assert "universal lossless recall is not promised" in r["context_claim"]
