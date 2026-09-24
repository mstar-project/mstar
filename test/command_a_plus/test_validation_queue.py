"""The full validation queue must never start over an occupied GPU."""

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from test.command_a_plus.run_validation import Pipeline


class QueueTests(unittest.TestCase):
    def make_pipeline(self):
        directory = self.enterContext(tempfile.TemporaryDirectory())
        pipeline = Pipeline(SimpleNamespace(run_dir=Path(directory), wait_hours=1))
        pipeline.deadline = 200
        return pipeline

    def test_busy_gpu_resets_idle_interval(self):
        pipeline = self.make_pipeline()
        clock = [1.0]

        def query(command, **kwargs):
            if "--query-compute-apps=pid" in command:
                # A new process arrives after the first idle sample. The idle
                # interval must start again after this job disappears.
                return "12345\n" if 15 <= clock[0] < 45 else ""
            return "\n".join(f"{i}, 4, 0" for i in range(8))

        def sleep(seconds):
            clock[0] += seconds

        with patch("test.command_a_plus.run_validation.subprocess.check_output", side_effect=query), \
             patch("test.command_a_plus.run_validation.time.monotonic", side_effect=lambda: clock[0]), \
             patch("test.command_a_plus.run_validation.time.sleep", side_effect=sleep), \
             patch.object(pipeline, "status"):
            pipeline.wait_idle("reference")
        self.assertGreaterEqual(clock[0], 105)

    def test_seven_gpus_or_busy_gpu_cannot_pass(self):
        for count, memory, processes in ((7, 4, ""), (8, 1024, ""), (8, 4, "12345")):
            with self.subTest(count=count, memory=memory, processes=processes):
                pipeline = self.make_pipeline()
                clock = [1.0]
                pipeline.deadline = 90

                def query(command, count=count, memory=memory, processes=processes, **kwargs):
                    if "--query-compute-apps=pid" in command:
                        return processes
                    return "\n".join(f"{i}, {memory}, 0" for i in range(count))

                def sleep(seconds, clock=clock):
                    clock[0] += seconds

                with patch("test.command_a_plus.run_validation.subprocess.check_output", side_effect=query), \
                     patch("test.command_a_plus.run_validation.time.monotonic",
                           side_effect=lambda clock=clock: clock[0]), \
                     patch("test.command_a_plus.run_validation.time.sleep", side_effect=sleep), \
                     patch.object(pipeline, "status"), self.assertRaises(TimeoutError):
                    pipeline.wait_idle("reference")


if __name__ == "__main__":
    unittest.main()
