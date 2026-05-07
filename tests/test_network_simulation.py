import unittest

from remote_target import NetworkSimulation


class NetworkSimulationTests(unittest.TestCase):
    def test_disabled_simulation_has_zero_delay(self):
        simulation = NetworkSimulation()

        self.assertEqual(simulation.uplink_delay_ns(1024), 0)
        self.assertEqual(simulation.downlink_delay_ns(1024), 0)

    def test_delay_includes_half_rtt_and_transfer_time(self):
        simulation = NetworkSimulation(
            enabled=True,
            rtt_ms=40,
            uplink_mbps=100,
            downlink_mbps=200,
        )

        self.assertEqual(simulation.one_way_latency_ns, 20_000_000)
        self.assertEqual(simulation.uplink_delay_ns(1_000_000), 100_000_000)
        self.assertEqual(simulation.downlink_delay_ns(1_000_000), 60_000_000)


if __name__ == "__main__":
    unittest.main()
