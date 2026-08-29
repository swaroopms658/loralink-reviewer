import pytest
from device_manager import DeviceHandle, DeviceStats, DeviceStatus, compute_assignments, PartitionInfeasible

def _dev(ip, flops, mem, dtype="cuda"):
    return DeviceHandle(ip=ip, status=DeviceStatus.HEALTHY,
                        stats=DeviceStats(flops=flops, memory_gb=mem, device_type=dtype))

def _cluster():
    return [
        _dev("127.0.0.1", 0.2, 12.0, "cpu"),   # coordinator
        _dev("127.0.0.2", 8.0, 15.0),
        _dev("127.0.0.3", 8.0, 15.0),
        _dev("127.0.0.4", 8.0, 15.0),
    ]

@pytest.mark.parametrize("strategy", ["smart", "round_robin", "proportional", "random"])
def test_all_layers_assigned_and_coordinator_capped(strategy):
    devs = _cluster()
    a = compute_assignments(strategy, devs, num_layers=12, layer_size_gb=0.02,
                            embedding_size_gb=0.1, master_ip="127.0.0.1",
                            utilization_limit=0.70, seed=0)
    assert sum(a.values()) == 12
    assert a["127.0.0.1"] <= 1
    assert all(c >= 0 for c in a.values())
    assert all(a[d.ip] >= 1 for d in devs[1:])          # every worker participates

def test_proportional_favours_faster_worker():
    devs = _cluster()
    devs[1].stats.flops = 40.0                           # 127.0.0.2 much faster
    a = compute_assignments("proportional", devs, 12, 0.02, 0.1,
                            "127.0.0.1", 0.70, seed=0)
    assert a["127.0.0.2"] > a["127.0.0.3"]

def test_infeasible_raises():
    devs = _cluster()
    for d in devs:
        d.stats.memory_gb = 0.05                         # nothing fits
    with pytest.raises(PartitionInfeasible):
        compute_assignments("round_robin", devs, 12, 0.5, 0.1,
                            "127.0.0.1", 0.70, seed=0)

def test_random_is_seed_deterministic():
    devs = _cluster()
    a1 = compute_assignments("random", devs, 12, 0.02, 0.1, "127.0.0.1", 0.70, seed=7)
    a2 = compute_assignments("random", devs, 12, 0.02, 0.1, "127.0.0.1", 0.70, seed=7)
    assert a1 == a2


def _three_workers():
    return [
        _dev("127.0.0.1", 0.2, 12.0, "cpu"),   # coordinator
        _dev("127.0.0.2", 8.0, 15.0),
        _dev("127.0.0.3", 8.0, 15.0),
        _dev("127.0.0.4", 8.0, 15.0),
    ]

@pytest.mark.parametrize("strategy", ["random", "proportional"])
def test_too_few_layers_raises_partition_infeasible(strategy):
    # F4: 3 workers, 2 layers -> cannot give every worker >= 1.
    devs = _three_workers()
    with pytest.raises(PartitionInfeasible):
        compute_assignments(strategy, devs, num_layers=2, layer_size_gb=0.02,
                            embedding_size_gb=0.1, master_ip="127.0.0.1",
                            utilization_limit=0.70, seed=0)

def test_proportional_repair_gives_every_worker_at_least_one():
    # F5: lopsided flops that pushes all remainder onto one worker; repair must
    # still leave every worker with >= 1 layer (and never re-rob a repaired one).
    devs = _three_workers()
    devs[1].stats.flops = 1000.0
    devs[2].stats.flops = 1.0
    devs[3].stats.flops = 1.0
    a = compute_assignments("proportional", devs, num_layers=4, layer_size_gb=0.02,
                            embedding_size_gb=0.1, master_ip="127.0.0.1",
                            utilization_limit=0.70, seed=0)
    assert sum(a.values()) == 4
    assert all(a[d.ip] >= 1 for d in devs[1:])

def test_smart_dropped_layers_raises_partition_infeasible():
    # F6: tail worker tight on memory -> section-6 LM-head shed drops layers with
    # nowhere to re-place them. Must raise instead of returning a partial model.
    devs = [
        _dev("127.0.0.1", 100.0, 10.0),   # coordinator (cuda)
        _dev("127.0.0.2", 30.0, 10.0),
        _dev("127.0.0.3", 20.0, 10.0),
        _dev("127.0.0.4", 10.0, 6.0),     # tail, tight
    ]
    with pytest.raises(PartitionInfeasible):
        compute_assignments("smart", devs, num_layers=25, layer_size_gb=1.0,
                            embedding_size_gb=2.0, master_ip="127.0.0.1",
                            utilization_limit=0.70, seed=0)
