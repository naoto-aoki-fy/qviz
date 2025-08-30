# qviz

Statevector visualizer for Qiskit circuits.

## Features
- Simulate circuits step by step with optional interpolation.
- Visualize amplitude and phase using HSV coloring or monochrome.
- Export frames to MP4 through FFmpeg.
- Minimal changes required: pass an existing `QuantumCircuit` to `visualize_circuit`.

## Installation
Install the package and its dependencies with pip:

```bash
pip install git+https://github.com/naoto-aoki-fy/qviz
```

For a local checkout, run:

```bash
pip install .
```

## Minimal Example
```python
from qiskit import QuantumCircuit
from qviz import visualize_circuit

qc = QuantumCircuit(3)
qc.h(0)
qc.cx(0, 1)
qc.cx(1, 2)
visualize_circuit(qc, output="cx_chain.mp4", fps=8, interpolate=8)
```

## Requirements
- qiskit
- qiskit-aer
- numpy
- tqdm
- ffmpeg (binary)

