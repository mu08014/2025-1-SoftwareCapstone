import os, numpy as np, time, math, tensorflow as tf
os.environ["TF_XLA_FLAGS"] = "--tf_xla_auto_jit=0 --tf_xla_cpu_global_jit=false"
tf.config.set_visible_devices([], 'GPU')

from qiskit_aer import AerSimulator
from qiskit import QuantumCircuit, transpile

N = 256
backend = AerSimulator(method="statevector", device="GPU",
                       precision="single",
                       max_parallel_experiments=1)   # ← 1 로 조정

qc = QuantumCircuit(9); qc.h(range(9))
tqc = transpile(qc, backend)
circs = [tqc]*N

t0 = time.time()
print("run…", flush=True)
backend.run(circs, shots=1).result()
print("OK", time.time()-t0, "sec")