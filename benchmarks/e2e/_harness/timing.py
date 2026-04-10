"""Timing primitives for benchmark runners.

Neither class performs CUDA synchronization — callers are expected to call
``torch.cuda.synchronize()`` *before* calling ``record()`` so the elapsed
measurement captures the real kernel completion time.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Mapping

from benchmarks.e2e._harness.result_schema import NodeRecord, StageRecord

_OTHER_STAGE = "other"


class NodeRecorder:
    """Collects per-node timing measurements with per-class_type call indices."""

    def __init__(self) -> None:
        self._records: list[NodeRecord] = []
        self._counts: dict[str, int] = defaultdict(int)

    def record(self, class_type: str, elapsed_ns: int) -> None:
        """Append a new node record.

        Each call to the same ``class_type`` gets an incrementing
        ``call_index`` (0, 1, 2, ...) so repeated nodes can be distinguished
        in the final report.

        Args:
            class_type: The ComfyUI node class name (e.g. ``"KSampler"``).
            elapsed_ns: Wall-clock elapsed time in nanoseconds. Callers must
                call ``torch.cuda.synchronize()`` before computing this value.
        """
        call_index = self._counts[class_type]
        self._counts[class_type] += 1
        self._records.append(
            NodeRecord(
                class_type=class_type,
                call_index=call_index,
                elapsed_ns=elapsed_ns,
            )
        )

    def as_list(self) -> list[NodeRecord]:
        """Return the recorded nodes in invocation order."""
        return list(self._records)


class StageRecorder:
    """Aggregates per-node timings into named stages.

    ``stage_mapping`` is an ordered mapping from stage name to the list of
    ``class_type`` strings that belong to that stage. Any ``class_type`` not
    in the mapping is accumulated under the ``"other"`` stage.
    """

    def __init__(self, stage_mapping: Mapping[str, list[str]]) -> None:
        # Preserve the caller-provided ordering so the report table has a
        # predictable column order.
        self._stage_mapping = dict(stage_mapping)
        self._class_to_stage: dict[str, str] = {}
        for stage_name, class_types in self._stage_mapping.items():
            for class_type in class_types:
                self._class_to_stage[class_type] = stage_name
        # Initialize each known stage to zero so they always appear in the
        # output even if never recorded.
        self._elapsed: dict[str, int] = {name: 0 for name in self._stage_mapping}
        self._counts: dict[str, int] = {name: 0 for name in self._stage_mapping}

    def record(self, class_type: str, elapsed_ns: int) -> None:
        """Accumulate ``elapsed_ns`` into the stage that owns ``class_type``.

        Args:
            class_type: The ComfyUI node class name. If not present in the
                stage mapping, the timing is routed to the ``"other"`` stage.
            elapsed_ns: Wall-clock elapsed time in nanoseconds. Callers must
                call ``torch.cuda.synchronize()`` before computing this value.
        """
        stage_name = self._class_to_stage.get(class_type, _OTHER_STAGE)
        self._elapsed[stage_name] = self._elapsed.get(stage_name, 0) + elapsed_ns
        self._counts[stage_name] = self._counts.get(stage_name, 0) + 1

    def as_list(self) -> list[StageRecord]:
        """Return stage records in stage-mapping order, with ``"other"`` last.

        Returns:
            List of :class:`StageRecord` instances. Known stages always appear
            (even if their count is zero). The ``"other"`` stage only appears
            if at least one unknown ``class_type`` was recorded.
        """
        ordered: list[StageRecord] = []
        for name in self._stage_mapping:
            ordered.append(
                StageRecord(
                    name=name,
                    elapsed_ns=self._elapsed.get(name, 0),
                    node_count=self._counts.get(name, 0),
                )
            )
        if self._elapsed.get(_OTHER_STAGE, 0) > 0:
            ordered.append(
                StageRecord(
                    name=_OTHER_STAGE,
                    elapsed_ns=self._elapsed[_OTHER_STAGE],
                    node_count=self._counts[_OTHER_STAGE],
                )
            )
        return ordered
