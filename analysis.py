import numpy as np
from dataclasses import dataclass, field
from numpy.typing import NDArray

from system import Snapshots

SERVING = 0
TUNING = 1

def calculate_mean_wait_time(snapshots: Snapshots) -> NDArray[np.float64]:
    size = snapshots.size

    totals = snapshots.total_wait_times[:size]
    counts = snapshots.cumulative_served_clients[:size]

    return np.divide(
        totals,
        counts,
        out=np.zeros_like(totals),
        where=counts != 0,
    )

def calculate_mean_queue_length(snapshots: Snapshots) -> NDArray[np.float64]:
    size = snapshots.size

    times = snapshots.times[:size]
    queues = snapshots.queue_lengths[:size]

    if size == 0:
        return np.empty((0, snapshots.number_of_threads))

    elapsed = times - times[0]

    areas = np.zeros_like(queues, dtype=np.float64)

    if size > 1:
        dt = np.diff(times)
        areas[1:] = np.cumsum(
            queues[:-1] * dt[:, None],
            axis=0,
        )

    return np.divide(
        areas,
        elapsed[:, None],
        out=np.zeros_like(areas),
        where=elapsed[:, None] != 0,
    )

def calculate_mean_state_duration(snapshots: Snapshots) -> NDArray[np.float64]:

    size = snapshots.size
    n_threads = snapshots.number_of_threads

    times = snapshots.times[:size]
    states = snapshots.states[:size]

    result = np.zeros((size, n_threads), dtype=np.float64)

    total_duration = np.zeros(n_threads)
    completed_periods = np.zeros(n_threads, dtype=np.int64)

    serving_start = times[0] if states[0, 1] == SERVING else None

    for k in range(1, size):
        previous_state = states[k - 1]
        current_state = states[k]

        # Detect the end of a serving period.
        if previous_state[1] == SERVING and current_state[1] == TUNING:
            thread = previous_state[0]

            total_duration[thread] += times[k] - serving_start
            completed_periods[thread] += 1

            serving_start = None

        # Detect the beginning of a serving period.
        elif previous_state[1] == TUNING and current_state[1] == SERVING:
            serving_start = times[k]

        # Calculate current means.
        np.divide(
            total_duration,
            completed_periods,
            out=result[k],
            where=completed_periods != 0,
        )

    return result