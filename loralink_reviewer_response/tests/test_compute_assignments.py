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
