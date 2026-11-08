"""The set-up heap freeze and the data worker's per-frame wake-up."""
import gc

from mstar.utils.gc_freeze import freeze_after_setup


def test_freeze_after_setup_moves_the_heap():
    gc.unfreeze()
    keep = [[i] for i in range(100)]  # live objects the collector tracks
    keep_ids = {id(obj) for obj in keep}
    try:
        frozen = freeze_after_setup("test")
        assert frozen > 0
        # Refcounting can free other frozen objects, so the total can shrink.
        # Check our retained objects moved out of the collected generations.
        assert keep_ids.isdisjoint(map(id, gc.get_objects()))
        assert keep[0] == [0]  # frozen objects are still alive and usable
    finally:
        gc.unfreeze()
        assert gc.get_freeze_count() == 0
    assert keep_ids <= {id(obj) for obj in gc.get_objects()}
