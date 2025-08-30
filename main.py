import qviz

qc = qviz.make_qft(12, do_swaps=True)
qviz.visualize_circuit(qc, output="qft_30.mp4", fps=30, interpolate=30, mapping="hsv_value", scale=16)

# qc = qviz.make_measurement(12)
# qviz.visualize_circuit(qc, output="measurement_30.mp4", fps=30, interpolate=30, mapping="hsv_value", scale=16)