import numpy as np
from gdn3.persistence.archive import Archive

def test_archive_visibility_and_checksum(tmp_path):
    a=Archive(tmp_path/"a.db",2);a.append("s",1,2,{"x":1},np.array([1.,0.]),"x")
    assert a.latest("s","x",3)["payload"]=={"x":1}
    assert a.latest("s","x",2) is None;a.close()


def test_overwrite_hides_old_semantic_candidate(tmp_path):
    a=Archive(tmp_path/"b.db",2);old=a.append("s",1,2,{"old":1},np.array([1.,0.]),"k");a.append("s",3,4,{"new":1},np.array([0.,1.]),"k")
    rows,_=a.search("s",np.array([1.,0.]),before=5,limit=8);assert all(r["id"]!=old for r in rows);a.close()
