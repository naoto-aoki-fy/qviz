import qviz
import numpy

num_qubits = 12
num_states = 1 << num_qubits

input_data = numpy.load("qft_init.npy")
assert input_data.shape == (num_states, )

qc = qviz.make_qft(12, do_swaps=True)
qviz.visualize_circuit(qc, output="qft_30.mp4", fps=1, interpolate=1, mapping="hsv_value", scale=16, initial_state=input_data)
