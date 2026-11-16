"""Discover event streams without changing the upstream P300 protocol."""
from pylsl import resolve_streams


def discover_marker_streams(timeout=1.0):
    # A broad lookup also finds Events/Triggers and case variants of Markers.
    candidates = []
    seen = set()
    for info in resolve_streams(wait_time=timeout):
        kind = info.type().strip().lower()
        name = info.name().strip().lower()
        if info.channel_count() != 1 or kind in ('eeg', 'signal'):
            continue
        if 'migalka' in name:  # SSVEP events are not P300 tile events.
            continue
        if not (any(word in kind for word in ('marker', 'event', 'trigger'))
                or any(word in name for word in ('marker', 'event', 'trigger', 'stim'))):
            continue
        identity = info.uid() or (info.name(), info.source_id(), info.hostname())
        if identity not in seen:
            seen.add(identity)
            candidates.append(info)
    return sorted(candidates, key=lambda info: (info.name(), info.hostname(), info.source_id()))


def marker_stream_label(info):
    return (f'{info.name()} | {info.type()} | {info.channel_count()} кан. | '
            f'{info.hostname() or "?"} | {info.source_id() or "без source_id"} '
            f'| {info.uid()[:8]}')
