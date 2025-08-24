import qiskit
import shors_algorithm
from quantum_visualization.gui_backend.main_backend import QuantumCircuitWindow
from typing import List

def visualize(argv: List[str]) -> None:
    if "sum" in argv:
        bloch_sphere_window = QuantumCircuitWindow()
        shors_algorithm.sum_gate(visualization_window=bloch_sphere_window)

        bloch_sphere_window.animate_circuit()
        bloch_sphere_window.app.exec()

    if "carry" in argv:
        bloch_sphere_window = QuantumCircuitWindow()
        shors_algorithm.carry_gate(visualization_window=bloch_sphere_window)

        bloch_sphere_window.animate_circuit()
        bloch_sphere_window.app.exec()

    if "adder" in argv:
        bloch_sphere_window = QuantumCircuitWindow(ignore_entanglement=True)
        shors_algorithm.adder_gate(visualization_window=bloch_sphere_window)

        bloch_sphere_window.animate_circuit()
        bloch_sphere_window.app.exec()

    if "m_adder" in argv:
        bloch_sphere_window = QuantumCircuitWindow()
        shors_algorithm.modulated_adder_gate(5, visualization_window=bloch_sphere_window)

        bloch_sphere_window.animate_circuit()
        bloch_sphere_window.app.exec()

        
    coprime = 5
    product_of_primes = 3 * 2
    
    if "cmm" in argv:
        bloch_sphere_window = QuantumCircuitWindow(ignore_entanglement=True)
        shors_algorithm.controlled_mod_mult_gate(product_of_primes, 3, visualization_window=bloch_sphere_window)

        bloch_sphere_window.animate_circuit()
        bloch_sphere_window.app.exec()
    
    if "mod_exp" in argv:
        bloch_sphere_window = QuantumCircuitWindow(ignore_entanglement=True)
        shors_algorithm.modular_exponentiation(product_of_primes, coprime, visualization_window=bloch_sphere_window)

        bloch_sphere_window.animate_circuit()
        bloch_sphere_window.app.exec()

    if "qpe" in argv:
        bloch_sphere_window = QuantumCircuitWindow()
        shors_algorithm.qpe_gate(6, visualization_window=bloch_sphere_window)

        bloch_sphere_window.animate_circuit()
        bloch_sphere_window.app.exec()
    
    if "vshor" in argv:
        shors_algorithm.visualize_shors()    