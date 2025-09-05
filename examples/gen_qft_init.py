import multiprocessing
num_threads = multiprocessing.cpu_count()

import qiskit
import qiskit.quantum_info
from qiskit.circuit.library import QFTGate
from qiskit import QuantumCircuit
import qiskit_aer.backends
import numpy
import numpy.random

num_qubits = 12
num_states = 1 << num_qubits
input_data = numpy.zeros(shape=(num_states,), dtype=numpy.complex128)

num_points = int(num_states * 0.1)
indices = numpy.random.randint(0, num_states - 1, size=(num_points, ))
angle = numpy.random.rand(num_points) * 2 * numpy.pi
input_data[indices] = numpy.cos(angle) + 1j * numpy.sin(angle)
input_data /= numpy.linalg.norm(input_data)

simulator = qiskit_aer.backends.StatevectorSimulator()
simulator.set_options(
    max_parallel_threads = num_threads,
    max_parallel_experiments = 0,
    max_parallel_shots = 1,
    statevector_parallel_threshold = num_threads
)

psi = qiskit.quantum_info.Statevector(input_data)

qc = QuantumCircuit(num_qubits)
qc.initialize(psi)

qc.append(QFTGate(num_qubits).inverse(), range(num_qubits))

qc.save_statevector("")

job = simulator.run(qiskit.transpile(qc, simulator))
job.wait_for_final_state(wait=0)
result = job.result()

psi = result.get_statevector(qc)

output_data = numpy.asarray(psi.data)

numpy.save("qft_init.npy", output_data)