import qc_statevis

qc = qc_statevis.make_qft(12, do_swaps=True)
qc_statevis.visualize_circuit(qc, output="qft_30.mp4", fps=30, interpolate=30, mapping="hsv_value", scale=16)

# qc = qc_statevis.make_measurement(12)
# qc_statevis.visualize_circuit(qc, output="measurement_30.mp4", fps=30, interpolate=30, mapping="hsv_value", scale=16)