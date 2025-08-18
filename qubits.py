from grovers_algorithm import create_quantum_circuit
from shors_algorithm import create_shors_circuit
from test_shors import test_shors
from visualize_gates import visualize

import os
os.environ['OMP_NUM_THREADS'] = '1'

def main():
    # create_quantum_circuit()
    # create_shors_circuit()
    # test_shors()
    visualize()
    
if __name__ == "__main__":
    main()