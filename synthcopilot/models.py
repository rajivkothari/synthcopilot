"""Data models for Synth Riders track data.

These are SynthCoPilot's lightweight in-memory representation. Reading and
writing real ``.synth`` files is handled by :mod:`synthcopilot.smh_io` (which
delegates to ``synth_mapping_helper``); these dataclasses are the neutral
form the generator and style engine operate on.
"""

from dataclasses import dataclass, field

DIFFICULTIES = ("Easy", "Normal", "Hard", "Expert", "Master")

HAND_RIGHT = 0
HAND_LEFT = 1


@dataclass
class Note:
    """A single note. ``x``/``y`` are grid units (floor-relative y, center 1.5);
    ``time`` is in beats."""

    time: float
    x: float
    y: float
    hand_type: int = HAND_RIGHT


@dataclass
class RailNode:
    """One node along a rail path (same coordinate convention as Note)."""

    time: float
    x: float
    y: float


@dataclass
class Rail:
    """A rail: an ordered list of RailNodes for one hand."""

    hand_type: int = HAND_RIGHT
    nodes: list = field(default_factory=list)


@dataclass
class Difficulty:
    name: str
    notes: list = field(default_factory=list)
    rails: list = field(default_factory=list)
    walls: list = field(default_factory=list)


@dataclass
class TrackData:
    bpm: float = 120.0
    offset: float = 0.0
    name: str = ""
    author: str = ""
    difficulties: dict = field(default_factory=dict)
    audio_filename: str | None = None

    def seconds_to_beats(self, seconds: float) -> float:
        """Convert a timestamp in seconds to beat time."""
        return (seconds - self.offset) * (self.bpm / 60.0)

    def beats_to_seconds(self, beats: float) -> float:
        """Convert beat time back to seconds."""
        return (beats * 60.0 / self.bpm) + self.offset
