from grovers_algorithm import create_quantum_circuit
from shors_algorithm import entire_shors_algorithm
from test_shors import test_shors
from visualize_gates import visualize
import sys


import os
os.environ['OMP_NUM_THREADS'] = '1'

def main():
    argv = sys.argv
    argc = len(sys.argv)

    if "grovers" in argv or "g" in argv:
        create_quantum_circuit()
    if "shors" in argv or "s" in argv:
        entire_shors_algorithm()
    if "test_s" in argv:
        test_shors()
    if "visualize" in argv or "v" in argv:
        visualize(argv)
    
if __name__ == "__main__":
    main()