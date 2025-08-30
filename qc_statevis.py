"""
Statevector visualizer for Qiskit circuits
=========================================

Purpose
-------
Simulate any `QuantumCircuit` step by step (with optional interpolation), render the
statevector at each step as an image, and export the frames as an MP4 video.
Existing circuits can be visualized by simply passing a finished circuit `qc`.

Highlights
- *Minimal changes*: just call `visualize_circuit(qc, output="out.mp4")`.
- *Interpolation*: if a gate supports `.power()`, `interpolate>1` produces smooth transitions.
- *Parallelization*: automatically sets the number of CPU threads for the Aer `StatevectorSimulator`.
- *Coloring*: choose HSV (hue = phase) or monochrome (thresholded).
- *Layout*: 2^row_bits × 2^(n-row_bits); defaults to a near square.
- *FFmpeg*: pipe frames to FFmpeg for a high-quality MP4.

Notes
- The statevector method consumes a large amount of memory as qubit count increases.
- Non-unitary instructions such as `measure` are skipped (collapse visualization is out of scope).

Dependencies
- qiskit, qiskit-aer, numpy, tqdm, ffmpeg (binary)

Minimal usage
-------------
>>> from qiskit import QuantumCircuit
>>> from qc_statevis import visualize_circuit
>>> qc = QuantumCircuit(3)
>>> qc.h(0); qc.cx(0,1); qc.cx(1,2)
>>> visualize_circuit(qc, output="cx_chain.mp4", fps=8, interpolate=8)

QFT example (prepare circuit then visualize)
--------------------------------------------
>>> from qc_statevis import make_qft, visualize_circuit
>>> qc = make_qft(12, do_swaps=True)
>>> visualize_circuit(qc, output="qft.mp4", fps=8, interpolate=8,
...                   mapping="hsv_value", scale=16)

"""
from __future__ import annotations

import math
import shutil
import subprocess
import multiprocessing
from dataclasses import dataclass
from typing import Iterable, Optional, Sequence, Tuple

import numpy as np
from tqdm import tqdm

from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import Statevector
from qiskit.circuit import Instruction, Clbit, ClassicalRegister
from qiskit.circuit.library import PhaseGate, HGate, SwapGate

try:
    from qiskit_aer.backends import StatevectorSimulator
except Exception as e:  # pragma: no cover
    raise ImportError(
        "qiskit-aer is required. Run `pip install qiskit-aer`.\n"
        f"ImportError: {e}"
    )

# ------------------------------------------------------------
# Image conversion utilities
# ------------------------------------------------------------

def _hsv_to_rgb_np(hsv: np.ndarray) -> np.ndarray:
    """Minimal HSV→RGB implementation (vectorized / [0,1] range).
    hsv[...,0]=h (0..1, hue), hsv[...,1]=s, hsv[...,2]=v
    Returns RGB array in [0,1] with the same shape.
    """
    h = hsv[..., 0] * 6.0
    s = hsv[..., 1]
    v = hsv[..., 2]

    i = np.floor(h).astype(int)
    f = h - i
    p = v * (1.0 - s)
    q = v * (1.0 - s * f)
    t = v * (1.0 - s * (1.0 - f))

    r = np.empty_like(h)
    g = np.empty_like(h)
    b = np.empty_like(h)

    i_mod = i % 6
    mask0 = i_mod == 0
    mask1 = i_mod == 1
    mask2 = i_mod == 2
    mask3 = i_mod == 3
    mask4 = i_mod == 4
    mask5 = i_mod == 5

    r[mask0], g[mask0], b[mask0] = v[mask0], t[mask0], p[mask0]
    r[mask1], g[mask1], b[mask1] = q[mask1], v[mask1], p[mask1]
    r[mask2], g[mask2], b[mask2] = p[mask2], v[mask2], t[mask2]
    r[mask3], g[mask3], b[mask3] = p[mask3], q[mask3], v[mask3]
    r[mask4], g[mask4], b[mask4] = t[mask4], p[mask4], v[mask4]
    r[mask5], g[mask5], b[mask5] = v[mask5], p[mask5], q[mask5]

    return np.stack([r, g, b], axis=-1)


def magang_to_rgb(mag: np.ndarray, ang: np.ndarray, mapping: str = "hsv_value") -> np.ndarray:
    """Generate an RGB image ([0,1] float) from amplitude |ψ| and phase ∠ψ.

    Parameters
    ----------
    mag : (H,W) ndarray
        magnitude (clipped to 0..1)
    ang : (H,W) ndarray
        radians (any range)
    mapping : {"hsv_value", "hsv_saturation"}
        - "hsv_value": hue=angle, sat=1, value=mag
        - "hsv_saturation": hue=angle, sat=mag, value=1
    """
    mag = np.asarray(mag, dtype=float)
    ang = np.asarray(ang, dtype=float)
    if mag.shape != ang.shape:
        raise ValueError("mag and ang must have the same shape")

    mag = np.clip(mag, 0.0, 1.0)
    two_pi = 2.0 * np.pi
    h = (np.mod(ang, two_pi) / two_pi)  # 0..1

    if mapping == "hsv_value":
        s = np.ones_like(h)
        v = mag
    elif mapping == "hsv_saturation":
        s = mag
        v = np.ones_like(h)
    else:
        raise ValueError("mapping must be 'hsv_value' or 'hsv_saturation'")

    hsv = np.stack([h, s, v], axis=-1)
    rgb = _hsv_to_rgb_np(hsv)
    return rgb


def psi_to_rgb(
    psi: Statevector | np.ndarray,
    *,
    n_qubits: int,
    row_bits: Optional[int] = None,
    mapping: str = "hsv_value",
    mono_threshold: Optional[float] = None,
    normalize_per_frame: bool = True,
) -> np.ndarray:
    """Convert a statevector to an RGB[0..255] (uint8) image.

    - Image size is 2^row_bits × 2^(n-row_bits); defaults to a near square.
    - Map bit array to 2D via simple reshape (row_bits controls high/low assignment).
    """
    if isinstance(psi, Statevector):
        sv = np.asarray(psi.data)
    else:
        sv = np.asarray(psi)

    if row_bits is None:
        # Make shape close to square (ceil/floor)
        row_bits = n_qubits // 2

    H = 1 << row_bits
    W = 1 << (n_qubits - row_bits)

    abs_ = np.abs(sv).reshape(H, W)
    ang_ = np.angle(sv).reshape(H, W)

    if mono_threshold is not None:
        img = (abs_ > mono_threshold).astype(np.uint8)
        rgb = np.repeat(img[..., None] * 255, 3, axis=2)
        return rgb

    if normalize_per_frame:
        denom = abs_.max() if abs_.size else 1.0
        denom = denom if denom > 0 else 1.0
        abs_norm = abs_ / denom
    else:
        abs_norm = abs_

    rgb_float = magang_to_rgb(abs_norm, ang_, mapping=mapping)
    rgb_u8 = np.ascontiguousarray((rgb_float * 255.0).clip(0, 255).astype(np.uint8))
    return rgb_u8


# ------------------------------------------------------------
# FFmpeg output
# ------------------------------------------------------------

@dataclass
class VideoParams:
    fps: int = 8
    codec: str = "libx264"
    out_pix_fmt: str = "yuv420p"
    crf: int = 15
    preset: str = "veryfast"
    scale: int = 16  # nearest-neighbor scaling (pixel-art style)


class FFmpegWriter:
    def __init__(self, width: int, height: int, output: str, params: VideoParams):
        self.width = width
        self.height = height
        self.output = output
        self.params = params
        self.proc: Optional[subprocess.Popen] = None

    def __enter__(self) -> "FFmpegWriter":
        if shutil.which("ffmpeg") is None:
            raise RuntimeError("ffmpeg not found. Please install it.")
        cmd = [
            "ffmpeg",
            "-y",
            "-loglevel",
            "warning",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-s",
            f"{self.width}x{self.height}",
            "-r",
            str(self.params.fps),
            "-i",
            "pipe:0",
            "-c:v",
            self.params.codec,
            "-pix_fmt",
            self.params.out_pix_fmt,
            "-vf",
            f"scale=iw*{self.params.scale}:ih*{self.params.scale}:flags=neighbor",
            "-crf",
            str(self.params.crf),
            "-preset",
            self.params.preset,
            "-movflags",
            "+faststart",
            self.output,
        ]
        self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
        return self

    def write(self, frame_rgb_u8: np.ndarray) -> None:
        if self.proc is None or self.proc.stdin is None:
            raise RuntimeError("FFmpegWriter has not been started")
        h, w, c = frame_rgb_u8.shape
        if (w != self.width) or (h != self.height) or (c != 3):
            raise ValueError(f"Frame size mismatch: got {w}x{h}x{c}, want {self.width}x{self.height}x3")
        if frame_rgb_u8.dtype != np.uint8:
            raise TypeError("Frame must be uint8")
        self.proc.stdin.write(frame_rgb_u8.tobytes())

    def __exit__(self, exc_type, exc, tb):
        if self.proc is not None and self.proc.stdin is not None:
            try:
                self.proc.stdin.close()
            except Exception:
                pass
            ret = self.proc.wait()
            if ret != 0:
                raise RuntimeError(f"ffmpeg exited with status {ret}")


# ------------------------------------------------------------
# Measurement and conditional support (simulated on statevector)
# ------------------------------------------------------------

# def _measure_and_collapse(psi: np.ndarray, n: int, q: int, rng: np.random.Generator) -> Tuple[np.ndarray, int]:
#     """Measure qubit q in the Z basis once and project the state (sampling)."""
#     abs2 = np.abs(psi) ** 2
#     idx = np.arange(psi.size, dtype=np.uint64)
#     bit = ((idx >> q) & 1).astype(np.uint8)
#     p1 = float(abs2[bit == 1].sum())
#     p0 = max(0.0, 1.0 - p1)
#     p1 = max(0.0, p1)
#     total = p0 + p1
#     if total <= 0:
#         r = 0
#         norm = 1.0
#         out = psi.copy()
#     else:
#         r = int(rng.choice([0, 1], p=[p0 / total, p1 / total]))
#         out = psi.copy()
#         if r == 0:
#             out[bit == 1] = 0
#             norm = math.sqrt(p0) if p0 > 0 else 1.0
#         else:
#             out[bit == 0] = 0
#             norm = math.sqrt(p1) if p1 > 0 else 1.0
#     if norm <= 0:
#         norm = 1.0
#     out = out / norm
#     return out, r


# def _reset_qubit(psi: np.ndarray, n: int, q: int) -> np.ndarray:
#     """reset q to |0> (measurement then X if outcome is |1>)."""
#     abs2 = np.abs(psi) ** 2
#     idx = np.arange(psi.size, dtype=np.uint64)
#     bit = ((idx >> q) & 1).astype(np.uint8)
#     p0 = float(abs2[bit == 0].sum())
#     out = psi.copy()
#     out[bit == 1] = 0
#     norm = math.sqrt(p0) if p0 > 0 else 1.0
#     return out / norm


# def _eval_condition(inst: Instruction, classical_state: dict) -> bool:
#     cond = getattr(inst, "condition", None)
#     if not cond:
#         return True
#     target, val = cond
#     if isinstance(target, Clbit):
#         cur = int(classical_state.get(target, 0))
#         return cur == int(val)
#     bits = list(getattr(target, "bits", [])) or list(target)
#     cur = 0
#     for i, b in enumerate(bits):
#         cur |= (int(classical_state.get(b, 0)) & 1) << i
#     return cur == int(val)


# def _clear_condition(inst: Instruction) -> Instruction:
#     try:
#         g = inst.copy()
#         g.condition = None
#         return g
#     except Exception:
#         try:
#             g = inst.copy()
#             setattr(g, "_condition", None)
#             return g
#         except Exception:
#             return inst

# ------------------------------------------------------------
# Visualization core
# ------------------------------------------------------------

def _build_simulator(max_threads: Optional[int] = None) -> StatevectorSimulator:
    if max_threads is None:
        max_threads = multiprocessing.cpu_count()
    sim = StatevectorSimulator()
    sim.set_options(
        max_parallel_threads=max_threads,
        max_parallel_experiments=0,
        max_parallel_shots=1,
        statevector_parallel_threshold=max_threads,
    )
    return sim


def _run_and_get_state(sim: StatevectorSimulator, qc: QuantumCircuit) -> Statevector:
    """Run a circuit once and return the final statevector.

    To avoid Aer's "Duplicate key 'statevector'" issue,
    save with a unique key and fetch directly from index 0.
    """
    qc2 = qc.copy()
    save_key = "_sv"
    # If a save with the same name already exists, do not add another (safety)
    if not any(getattr(inst, "name", "") == "save_statevector" for (inst, _q, _c) in qc2.data):
        qc2.save_statevector(save_key)
    else:
        # Even if one exists, try to relabel to avoid key collisions
        # (try/except fallback for compatibility)
        try:
            qc2.save_statevector(save_key)
        except Exception:
            pass

    job = sim.run(transpile(qc2, sim))
    result = job.result()

    # Assuming only one execution, retrieve from index=0
    data0 = result.data(0)
    vec = data0.get(save_key)
    if vec is None:
        # Compatibility fallback: look for default key
        vec = data0.get("statevector")
        if vec is None:
            raise RuntimeError("statevector not found in result (failed to insert save_statevector)")
    # Aer may return a list, so take the last element
    if isinstance(vec, (list, tuple)):
        vec = vec[-1]
    return Statevector(vec)


def _grid_shape(n_qubits: int, row_bits: Optional[int]) -> Tuple[int, int]:
    if row_bits is None:
        row_bits = n_qubits // 2
    H = 1 << row_bits
    W = 1 << (n_qubits - row_bits)
    return H, W


def visualize_circuit(
    qc: QuantumCircuit,
    *,
    output: str = "out.mp4",
    fps: int = 8,
    interpolate: int = 8,
    mapping: str = "hsv_value",
    mono_threshold: Optional[float] = None,
    row_bits: Optional[int] = None,
    normalize_per_frame: bool = True,
    scale: int = 16,
    # do_bit_swaps_at_end: bool = False,
    max_threads: Optional[int] = None,
    show_progress: bool = True,
    initial_state: Optional[Statevector | np.ndarray] = None,
) -> None:
    """Simulate the given circuit step by step and write a video.

    Parameters
    ----------
    qc : QuantumCircuit
        The finished circuit to visualize.
    output : str
        Path to the output mp4.
    fps : int
        Frame rate of the output video. Matching `interpolate` yields smooth playback.
    interpolate : int
        Number of substeps per instruction (effective only if the instruction supports `.power()`).
    mapping : {"hsv_value", "hsv_saturation"}
        Coloring method. Hue=phase; value/saturation=amplitude.
    mono_threshold : Optional[float]
        Switch to monochrome (255/0) when amplitude exceeds this threshold.
    row_bits : Optional[int]
        Number of bits mapped to rows (columns use n-row_bits). Defaults to n//2.
    normalize_per_frame : bool
        Normalize the maximum amplitude to 1 for each frame.
    scale : int
        Nearest-neighbor scale factor passed to FFmpeg.
    # do_bit_swaps_at_end : bool
    #     Append MSB↔LSB swap display at the end (useful for QFT reordering).
    max_threads : Optional[int]
        Number of threads for Aer. All cores by default.
    show_progress : bool
        Display progress with tqdm.
    initial_state : Optional[Statevector | np.ndarray]
        Initial state; |0…0⟩ if None.
    """

    n = qc.num_qubits
    H, W = _grid_shape(n, row_bits)

    sim = _build_simulator(max_threads=max_threads)

    params = VideoParams(fps=fps, scale=scale)
    with FFmpegWriter(W, H, output, params) as writer:
        # Initial state frame
        if initial_state is None:
            psi = Statevector.from_label("0" * n)
        else:
            psi = Statevector(initial_state)
        frame = psi_to_rgb(
            psi,
            n_qubits=n,
            row_bits=row_bits,
            mapping=mapping,
            mono_threshold=mono_threshold,
            normalize_per_frame=normalize_per_frame,
        )
        writer.write(frame)

        # Classical register state and RNG (for measurement)
        classical_state = {clb: 0 for clb in getattr(qc, "clbits", [])}
        rng = np.random.default_rng(12345)

        # Iterate over instruction list
        data = list(qc.data)
        steps: Sequence[Instruction] = [inst for (inst, _q, _c) in data]
        qargs_list: Sequence[Tuple[int, ...]] = []
        cargs_list: Sequence[Tuple[Clbit, ...]] = []
        for _inst, qargs, cargs in data:
            qidxs = tuple(qc.find_bit(qb).index for qb in qargs)
            qargs_list.append(qidxs)
            cargs_list.append(tuple(cargs))

        pbar = tqdm(total=len(steps), desc="apply ops", disable=not show_progress)

        for inst, qidxs, cargs in zip(steps, qargs_list, cargs_list):
            name = inst.name
            # Measurement
            if name == "measure":
                pairs = list(zip(qidxs, list(cargs))) if cargs else [(qidxs[0], None)]
                step = 1.0 / interpolate
                for q_i, cbit in pairs:
                    for _ in range(interpolate // 2):
                        frame = psi_to_rgb(
                            psi,
                            n_qubits=n,
                            row_bits=row_bits,
                            mapping=mapping,
                            mono_threshold=mono_threshold,
                            normalize_per_frame=normalize_per_frame,
                        )
                        writer.write(frame)
                        pbar.update(step)
                    qcs = QuantumCircuit(n, n)
                    qcs.initialize(psi)
                    qcs.measure(q_i, cbit)
                    psi = _run_and_get_state(sim, qcs)
                    for _ in range(interpolate - interpolate // 2):
                        frame = psi_to_rgb(
                            psi,
                            n_qubits=n,
                            row_bits=row_bits,
                            mapping=mapping,
                            mono_threshold=mono_threshold,
                            normalize_per_frame=normalize_per_frame,
                        )
                        writer.write(frame)
                        pbar.update(step)
                continue

            # reset (optional support)
            if name == "reset":
                step = 1.0 / interpolate
                for q_i in qidxs:
                    for _ in range(interpolate // 2):
                        frame = psi_to_rgb(
                            psi,
                            n_qubits=n,
                            row_bits=row_bits,
                            mapping=mapping,
                            mono_threshold=mono_threshold,
                            normalize_per_frame=normalize_per_frame,
                        )
                        writer.write(frame)
                        pbar.update(step)
                    qcs = QuantumCircuit(n, n)
                    qcs.initialize(psi)
                    qcs.reset(q_i, cbit)
                    psi = _run_and_get_state(sim, qcs)
                    for _ in range(interpolate - interpolate // 2):
                        frame = psi_to_rgb(
                            psi,
                            n_qubits=n,
                            row_bits=row_bits,
                            mapping=mapping,
                            mono_threshold=mono_threshold,
                            normalize_per_frame=normalize_per_frame,
                        )
                        writer.write(frame)
                        pbar.update(step)
                continue

            # barrier / save only continue the frame
            if name in ("barrier", "save_statevector"):
                frame = psi_to_rgb(
                    psi,
                    n_qubits=n,
                    row_bits=row_bits,
                    mapping=mapping,
                    mono_threshold=mono_threshold,
                    normalize_per_frame=normalize_per_frame,
                )
                writer.write(frame)
                pbar.update(1)
                continue

            # Evaluate classical conditions
            # if not _eval_condition(inst, classical_state):
            #     frame = psi_to_rgb(
            #         psi,
            #         n_qubits=n,
            #         row_bits=row_bits,
            #         mapping=mapping,
            #         mono_threshold=mono_threshold,
            #         normalize_per_frame=normalize_per_frame,
            #     )
            #     writer.write(frame)
            #     continue

            # For each step, build a circuit that applies the instruction once
            if interpolate > 1 and hasattr(inst, "power"):
                try:
                    frac = inst.power(1.0 / interpolate)
                    step = 1.0 / interpolate
                    for _ in range(interpolate):
                        qcs = QuantumCircuit(n)
                        qcs.initialize(psi)
                        qcs.append(frac, qidxs)
                        psi = _run_and_get_state(sim, qcs)
                        frame = psi_to_rgb(
                            psi,
                            n_qubits=n,
                            row_bits=row_bits,
                            mapping=mapping,
                            mono_threshold=mono_threshold,
                            normalize_per_frame=normalize_per_frame,
                        )
                        writer.write(frame)
                        pbar.update(step)
                    continue
                except Exception:
                    # Fallback when power() is unsupported
                    pass

            qcs = QuantumCircuit(n)
            qcs.initialize(psi)
            qcs.append(inst, qidxs)
            psi = _run_and_get_state(sim, qcs)
            frame = psi_to_rgb(
                psi,
                n_qubits=n,
                row_bits=row_bits,
                mapping=mapping,
                mono_threshold=mono_threshold,
                normalize_per_frame=normalize_per_frame,
            )
            writer.write(frame)
            pbar.update(1)

        pbar.close()



# ------------------------------------------------------------
# Utility functions (initial states / QFT generation / measurement example)
# ------------------------------------------------------------

def generate_random(n_qubits: int, seed: int = 12345) -> Statevector:
    rng = np.random.default_rng(seed)
    vec = rng.normal(size=(1 << n_qubits,)) + 1j * rng.normal(size=(1 << n_qubits,))
    vec /= np.linalg.norm(vec)
    return Statevector(vec)


def generate_uniform(n_qubits: int) -> Statevector:
    vec = np.ones(shape=(1 << n_qubits,), dtype=np.complex128)
    vec /= np.linalg.norm(vec)
    return Statevector(vec)


def make_measurement(n_qubits: int) -> QuantumCircuit:
    """Example measurement circuit: apply H to each qubit then measure to corresponding classical bit"""
    qc = QuantumCircuit(n_qubits, n_qubits)
    for j in range(n_qubits):
        qc.h(j)
    for j in range(n_qubits):
        qc.measure(j, j)
    return qc


def make_qft(n_qubits: int, *, do_swaps: bool = True) -> QuantumCircuit:
    """Generate a straightforward QFT (Phase+H, swap at the end)."""
    qc = QuantumCircuit(n_qubits)
    for j in range(n_qubits):
        # Apply controlled phase rotations from lower indices (0..j-1) to target j
        for k in range(j):
            phi = math.pi / (2 ** (j - k))
            qc.append(PhaseGate(phi).control(1), [k, j])
        qc.append(HGate(), [j])
    if do_swaps:
        for i in range(n_qubits // 2):
            qc.append(SwapGate(), [i, n_qubits - i - 1])
    return qc


# ------------------------------------------------------------
# Simple demo when run as a script
# ------------------------------------------------------------
if __name__ == "__main__":
    # Demo of 12-qubit QFT (base image ~4096 pixels scaled ×16)
    n = 12
    qc = make_qft(n, do_swaps=True)
    visualize_circuit(
        qc,
        output="qft_demo.mp4",
        fps=8,
        interpolate=8,
        mapping="hsv_value",
        scale=16,
        normalize_per_frame=True,
        show_progress=True,
    )
