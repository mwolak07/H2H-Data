from typing import List, Tuple, Any, Optional, Dict
from enum import Enum


class SamplingMethod(Enum):
    TO_TIMESTAMPS = 'to_timestamps'
    SIMPLE = 'simple'


def down_sample(data: List[Tuple[float, Any]], method: Optional[SamplingMethod] = SamplingMethod.TO_TIMESTAMPS,
                args: Optional[Dict[str, Any]] = None) -> List[Tuple[float, Any]]:
    """
    Down samples the given data using the given method.

    Args:
        data: The data to down sample.
        method: The method to use for the down sampling.
        args: The dict of arguments for the chosen method.

    Returns:
        The down-sampled data.

    Raises:
        ValueError: If the method is not supported.
    """
    if method == SamplingMethod.TO_TIMESTAMPS:
        return down_sample_to_timestamps(data, args['timestamps'])
    if method == SamplingMethod.SIMPLE:
        return simple_down_sample(data, args['source_fps'], args['target_fps'])
    else:
        raise ValueError(f'Method {method} is not supported.')


def down_sample_to_timestamps(data: List[Tuple[float, Any]], timestamps: List[float]) -> List[Tuple[float, Any]]:
    """
    Down samples the given data to the given timestamps. Selects the closest frame from data for each timestamp.
    Output timestamps are guaranteed to match the given timestamps.

    Args:
        data: The data to down sample, formatted (timestamp, ...).
        timestamps: The list of timestamps to target.

    Returns:
        The down-sampled data.
    """
    output = []
    for timestamp in timestamps:
        output.append(_get_closest_frame(data, timestamp))
    return output


def _get_closest_frame(data: List[Tuple[float, Any]], timestamp: float) -> Tuple[float, Any]:
    """
    Gets the frame from data closest to the given timestamp.

    Args:
        data: The data, formatted (timestamp, ...).
        timestamp: The timestamp to get the closest frame to.

    Returns:
        The frame from data closest to timestamp.
    """
    min_error = float('inf')
    output = data[0]
    for element in data:
        t = element[0]
        error = abs(t - timestamp)
        if error < min_error:
            output = element
            min_error = error
    return output


def simple_down_sample(data: List[Tuple[float, Any]], source_fps: int, target_fps: int) -> List[Tuple[float, Any]]:
    """
    Down samples the given data from the source fps to the target fps using simple closest frame selection.
    Generates the list of timestamps at the target fps, and for each target timestamp, chooses the closest frame
    from data.

    Args:
        data: The data to down sample, formatted (timestamp, ...).
        source_fps: The source fps.
        target_fps: The target fps.

    Returns:
        The down-sampled data.
    """
    # Store the offset as the first timestamp.
    offset = data[0][0]
    # Subtract the offset from each frame in the data.
    data = [(element[0] - offset,) + element[1:] for element in data]
    # Generate the target timestamps.
    start = 0
    source_t = 1 / source_fps
    target_t = 1 / target_fps
    duration = len(data) * source_t
    stop = int(round(duration / target_t))
    target_timestamps = [i * target_t for i in range(start, stop + 1)]
    # Sample from the target timestamps.
    output = []
    for timestamp in target_timestamps:
        output.append(_get_closest_frame(data, timestamp))
    # Add the offset back in for each frame in the data.
    data = [(element[0] + offset,) + element[1:] for element in data]
    return data
