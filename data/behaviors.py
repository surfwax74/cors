"""Nominal behavioral noise injectors for the synthetic telemetry generator.

These represent *nominal* daily activity (not anomalies): the everyday shifting
you would see on a healthy spacecraft bus. They are applied across the whole
timeline (train + test) so a good detector learns them as normal.

Two families are provided:

**Static-noise distributions** (stationary, additive noise floors):
  * ``gaussian``  - Normal / Gaussian noise (the two names are the same thing).
  * ``uniform``   - flat "random" noise within +/- amplitude.
  * ``laplace``   - sharper peak, heavier tails than Gaussian (spiky sensors).
  * ``student_t`` - heavy-tailed; occasional large excursions (df controls it).
  * ``pink``      - 1/f noise, typical of analog electronics.

**Behavioral activity** (non-stationary but nominal):
  * ``voltage_shift``  - random-walk between operating points (bus voltage).
  * ``step_wave``      - random-length square waves between discrete levels
                         (mode / state switching).
  * ``thruster``       - sparse impulsive firings with exponential decay.
  * ``reaction_wheel`` - trapezoidal spin up / hold / spin down ramps.

Each generator has the signature
``fn(n, fs, rng, amplitude, **params) -> np.ndarray`` where ``n`` is the number
of raw bins and ``fs`` is the raw sample period in seconds (so event timings can
be expressed in real seconds, independent of resolution).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class BehaviorSpec:
    """One behavioral noise source applied to a fraction of channels.

    Attributes:
        kind: a key in :data:`BEHAVIOR_GENERATORS`.
        fraction: fraction of eligible channels that receive this behavior.
        amplitude: overall scale of the contribution.
        params: kind-specific parameters (see each generator).
        name: optional label used in the assignment report.
    """

    kind: str
    fraction: float = 0.2
    amplitude: float = 1.0
    params: dict = field(default_factory=dict)
    name: str = ""

    def label(self) -> str:
        return self.name or self.kind


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _seconds_to_bins(seconds: float, fs: float, minimum: int = 1) -> int:
    return max(int(round(seconds / fs)), minimum)


# --------------------------------------------------------------------------- #
# Static-noise distributions
# --------------------------------------------------------------------------- #
def _noise_gaussian(n, fs, rng, amplitude, **p):
    """Normal/Gaussian noise with standard deviation ``amplitude``."""
    return rng.normal(0.0, amplitude, n)


def _noise_uniform(n, fs, rng, amplitude, **p):
    """Flat 'random' noise in [-amplitude, +amplitude]."""
    return rng.uniform(-amplitude, amplitude, n)


def _noise_laplace(n, fs, rng, amplitude, **p):
    """Laplace noise: sharper peak and heavier tails than Gaussian."""
    return rng.laplace(0.0, amplitude, n)


def _noise_student_t(n, fs, rng, amplitude, df=3.0, **p):
    """Heavy-tailed Student-t noise; lower ``df`` => more extreme spikes."""
    return amplitude * rng.standard_t(df, n)


def _noise_pink(n, fs, rng, amplitude, **p):
    """Pink (1/f) noise via spectral shaping of white noise."""
    white = rng.normal(0.0, 1.0, n)
    spectrum = np.fft.rfft(white)
    freqs = np.fft.rfftfreq(n, d=1.0)
    scale = np.ones_like(freqs)
    nonzero = freqs > 0
    scale[nonzero] = 1.0 / np.sqrt(freqs[nonzero])
    shaped = np.fft.irfft(spectrum * scale, n=n)
    std = shaped.std()
    if std > 0:
        shaped = shaped / std
    return amplitude * shaped


# --------------------------------------------------------------------------- #
# Behavioral activity
# --------------------------------------------------------------------------- #
def _behavior_voltage_shift(n, fs, rng, amplitude, min_seconds=1800, max_seconds=7200, **p):
    """Random-walk between operating points: piecewise-constant level shifts.

    Models e.g. bus voltage settling at slightly different points as loads
    change through the day.
    """
    out = np.zeros(n)
    min_len = _seconds_to_bins(min_seconds, fs)
    max_len = max(min_len + 1, _seconds_to_bins(max_seconds, fs))
    level = 0.0
    i = 0
    while i < n:
        seg = int(rng.integers(min_len, max_len + 1))
        level = float(np.clip(level + rng.normal(0, amplitude * 0.3), -amplitude, amplitude))
        out[i:i + seg] = level
        i += seg
    return out


def _behavior_step_wave(n, fs, rng, amplitude, min_seconds=300, max_seconds=3600, n_levels=4, **p):
    """Random-length square wave hopping between discrete levels (mode switching)."""
    out = np.zeros(n)
    levels = np.linspace(-amplitude, amplitude, max(int(n_levels), 2))
    min_len = _seconds_to_bins(min_seconds, fs)
    max_len = max(min_len + 1, _seconds_to_bins(max_seconds, fs))
    i = 0
    while i < n:
        seg = int(rng.integers(min_len, max_len + 1))
        out[i:i + seg] = float(rng.choice(levels))
        i += seg
    return out


def _behavior_thruster(n, fs, rng, amplitude, rate_per_hour=1.5, pulse_seconds=40, decay=True, **p):
    """Sparse impulsive firings: short pulses (optionally decaying) at random times."""
    out = np.zeros(n)
    if n <= 0:
        return out
    hours = n * fs / 3600.0
    n_events = int(rng.poisson(max(rate_per_hour * hours, 0.0)))
    plen = _seconds_to_bins(pulse_seconds, fs)
    for _ in range(n_events):
        start = int(rng.integers(0, n))
        end = min(start + plen, n)
        seg = end - start
        if seg <= 0:
            continue
        amp = amplitude * (0.5 + rng.random())
        shape = np.exp(-np.linspace(0, 3, seg)) if decay else np.ones(seg)
        out[start:end] += amp * shape
    return out


def _behavior_reaction_wheel(n, fs, rng, amplitude, rate_per_hour=0.4, ramp_seconds=120, hold_seconds=600, **p):
    """Trapezoidal spin up / hold / spin down events at random times."""
    out = np.zeros(n)
    if n <= 0:
        return out
    hours = n * fs / 3600.0
    n_events = int(rng.poisson(max(rate_per_hour * hours, 0.0)))
    ramp = _seconds_to_bins(ramp_seconds, fs)
    base_hold = _seconds_to_bins(hold_seconds, fs)
    for _ in range(n_events):
        amp = amplitude * (0.5 + rng.random())
        hold = max(1, int(base_hold * (0.5 + rng.random())))
        dur = 2 * ramp + hold
        start = int(rng.integers(0, n))
        end = min(start + dur, n)
        seg = end - start
        if seg <= 0:
            continue
        trap = np.full(seg, amp)
        up = min(ramp, seg)
        trap[:up] = np.linspace(0, amp, up)
        if seg > ramp:
            down = min(ramp, seg - ramp)
            if down > 0:
                trap[seg - down:] = np.linspace(amp, 0, down)
        out[start:end] += trap
    return out


# --------------------------------------------------------------------------- #
# Registry
# --------------------------------------------------------------------------- #
BEHAVIOR_GENERATORS = {
    "gaussian": _noise_gaussian,
    "uniform": _noise_uniform,
    "laplace": _noise_laplace,
    "student_t": _noise_student_t,
    "pink": _noise_pink,
    "voltage_shift": _behavior_voltage_shift,
    "step_wave": _behavior_step_wave,
    "thruster": _behavior_thruster,
    "reaction_wheel": _behavior_reaction_wheel,
}


def list_behaviors() -> list[str]:
    """Return all registered behavior kinds."""
    return list(BEHAVIOR_GENERATORS.keys())


def generate_behavior(kind, n, fs, rng, amplitude, **params) -> np.ndarray:
    """Generate a single behavior contribution of length ``n``."""
    if kind not in BEHAVIOR_GENERATORS:
        raise KeyError(f"Unknown behavior {kind!r}. Available: {', '.join(BEHAVIOR_GENERATORS)}")
    return BEHAVIOR_GENERATORS[kind](n, fs, rng, amplitude, **params)


def default_behaviors() -> list[BehaviorSpec]:
    """A satellite-flavoured default mix of nominal behaviors."""
    return [
        BehaviorSpec("gaussian", fraction=0.50, amplitude=0.30, name="thermal_noise"),
        BehaviorSpec("uniform", fraction=0.15, amplitude=0.40, name="quantization"),
        BehaviorSpec("laplace", fraction=0.10, amplitude=0.30, name="spiky_sensor"),
        BehaviorSpec("student_t", fraction=0.10, amplitude=0.25, params={"df": 3}, name="heavy_tail"),
        BehaviorSpec("pink", fraction=0.20, amplitude=0.50, name="electronic_1f"),
        BehaviorSpec("voltage_shift", fraction=0.15, amplitude=1.5, name="bus_voltage"),
        BehaviorSpec("step_wave", fraction=0.10, amplitude=2.0, params={"n_levels": 4}, name="mode_switch"),
        BehaviorSpec("thruster", fraction=0.06, amplitude=4.0,
                     params={"rate_per_hour": 1.5, "pulse_seconds": 40}, name="thruster_fire"),
        BehaviorSpec("reaction_wheel", fraction=0.06, amplitude=3.0,
                     params={"rate_per_hour": 0.4}, name="reaction_wheel"),
    ]


def _select_channels(eligible, fraction, rng):
    if not eligible or fraction <= 0:
        return np.array([], dtype=int)
    k = int(round(fraction * len(eligible)))
    k = min(len(eligible), max(1, k))  # at least one channel when fraction > 0
    return np.sort(rng.choice(eligible, size=k, replace=False))


def apply_behaviors(
    signals: np.ndarray,
    behaviors: list[BehaviorSpec],
    fs: float,
    rng: np.random.Generator,
    exclude_channels=(),
) -> dict[str, list[int]]:
    """Add nominal behavioral noise to a subset of channels, in place.

    Args:
        signals: array shaped ``[n_channels, n_bins]`` (modified in place).
        behaviors: the behaviors to apply.
        fs: raw sample period in seconds.
        rng: random generator (kept separate from the base-signal stream so
            adding behaviors does not perturb base/anomaly reproducibility).
        exclude_channels: channel indices that should not receive behaviors.

    Returns:
        Mapping of behavior label -> list of channel indices it was applied to.
    """
    n_channels, n_bins = signals.shape
    excluded = set(int(c) for c in exclude_channels)
    eligible = [c for c in range(n_channels) if c not in excluded]

    assignments: dict[str, list[int]] = {}
    for spec in behaviors:
        channels = _select_channels(eligible, spec.fraction, rng)
        for ch in channels:
            signals[ch] += generate_behavior(
                spec.kind, n_bins, fs, rng, spec.amplitude, **spec.params
            )
        assignments[spec.label()] = channels.tolist()
    return assignments
