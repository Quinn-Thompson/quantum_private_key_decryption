import qiskit
import shors_algorithm
from quantum_visualization.gui_backend.main_backend import QuantumCircuitWindow

def visualize():
    # bloch_sphere_window = QuantumCircuitWindow()
    # shors_algorithm.carry_gate(visualization_window=bloch_sphere_window)

    # bloch_sphere_window.animate_circuit()
    # bloch_sphere_window.app.exec()

    # bloch_sphere_window = QuantumCircuitWindow()
    # shors_algorithm.adder_gate(visualization_window=bloch_sphere_window)

    # bloch_sphere_window.animate_circuit()
    # bloch_sphere_window.app.exec()
    
    # bloch_sphere_window = QuantumCircuitWindow()
    # shors_algorithm.modulated_adder_gate(5, visualization_window=bloch_sphere_window)

    # bloch_sphere_window.animate_circuit()
    # bloch_sphere_window.app.exec()
    
    coprime = 5
    product_of_primes = 3 * 2
    
    bloch_sphere_window = QuantumCircuitWindow()
    shors_algorithm.controlled_mod_mult_gate(product_of_primes, coprime, visualization_window=bloch_sphere_window)

    bloch_sphere_window.animate_circuit()
    bloch_sphere_window.app.exec()