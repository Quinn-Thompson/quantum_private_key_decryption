import numpy as np
from typing import List, Literal
import matplotlib.pyplot as plt
import qiskit
from qiskit.quantum_info import Statevector, DensityMatrix, partial_trace
from qiskit_aer import Aer

from dataclasses import dataclass, fields
from quantum_visualization.visualize_qubits import BlochSpheres, BlochProperties


_ANCILARY_QUBITS = 1
_CIPHERTEXT_CUBITS = 1
_PLAINTEXT_QUBITS = 4
_KEY_QUBITS = 4
_FLIP_QUBIT = 1
_NUMBER_OF_QUBITS = _PLAINTEXT_QUBITS + _KEY_QUBITS + _FLIP_QUBIT
_NUMBER_OF_FRAMES = 100
_QUBIT_HILBERT_SPACE = 2
_INPUT_KEY = [0, 1, 1, 0]
_INPUT_PLAINTEXT = [1, 0, 1, 0]
_GROVER_ITERATIONS = 2


@dataclass
class QubitRegisters():
    plaintext_qubits: qiskit.QuantumRegister
    key_qubits: qiskit.QuantumRegister
    flip_qubit: qiskit.QuantumRegister


@dataclass
class BlochSpherePlots():
    plaintext_qubits_plots: BlochSpheres
    key_qubits_plots: BlochSpheres
    flip_qubit_plots: BlochSpheres

def aes_step(quantum_circuit: qiskit.QuantumCircuit, qubit_registers: QubitRegisters, bloch_spheres: BlochSpheres):
    # integrate key into plaintext encryption
    # this is the round key step
    for i in range(_KEY_QUBITS):
        quantum_circuit.cx(qubit_registers.key_qubits[i], qubit_registers.plaintext_qubits[i])
    bloch_spheres.add_circuit_state(quantum_circuit, BlochProperties(plot_name="round key"))
    # some nonlinear substritution step
    # reversable tofolli gate, where 110 = 111, 111= 110
    quantum_circuit.ccx(qubit_registers.plaintext_qubits[0], qubit_registers.plaintext_qubits[1], qubit_registers.plaintext_qubits[2])
    bloch_spheres.add_circuit_state(quantum_circuit, BlochProperties(plot_name="ccx"))

    quantum_circuit.cx(qubit_registers.plaintext_qubits[0], qubit_registers.plaintext_qubits[1])
    bloch_spheres.add_circuit_state(quantum_circuit, BlochProperties(plot_name="cx"))
    quantum_circuit.ccx(qubit_registers.plaintext_qubits[1], qubit_registers.plaintext_qubits[0], qubit_registers.plaintext_qubits[3])
    bloch_spheres.add_circuit_state(quantum_circuit, BlochProperties(plot_name="ccx"))
    
    # swap qubit 1 and 3, is basically shiftrow
    quantum_circuit.swap(qubit_registers.plaintext_qubits[2], qubit_registers.plaintext_qubits[3])
    bloch_spheres.add_circuit_state(quantum_circuit, BlochProperties(plot_name="swap"))

def aes_step_inverse(quantum_circuit: qiskit.QuantumCircuit, qubit_registers: QubitRegisters, bloch_spheres: BlochSpheres):
    # swap qubit 1 and 3, is basically shiftrow
    quantum_circuit.swap(qubit_registers.plaintext_qubits[2], qubit_registers.plaintext_qubits[3])
    bloch_spheres.add_circuit_state(quantum_circuit, BlochProperties(plot_name="swap"))
    
    # some nonlinear substritution step
    # reversable tofolli gate, where 110 = 111, 111= 110

    quantum_circuit.ccx(qubit_registers.plaintext_qubits[1], qubit_registers.plaintext_qubits[0], qubit_registers.plaintext_qubits[3])
    bloch_spheres.add_circuit_state(quantum_circuit, BlochProperties(plot_name="ccx"))
    
    quantum_circuit.cx(qubit_registers.plaintext_qubits[0], qubit_registers.plaintext_qubits[1])
    bloch_spheres.add_circuit_state(quantum_circuit, BlochProperties(plot_name="cx"))
    
    quantum_circuit.ccx(qubit_registers.plaintext_qubits[0], qubit_registers.plaintext_qubits[1], qubit_registers.plaintext_qubits[2])
    bloch_spheres.add_circuit_state(quantum_circuit, BlochProperties(plot_name="ccx"))
        # integrate key into plaintext encryption
    # this is the round key step
    for i in range(_KEY_QUBITS):
        quantum_circuit.cx(qubit_registers.key_qubits[i], qubit_registers.plaintext_qubits[i])
    bloch_spheres.add_circuit_state(quantum_circuit, BlochProperties(plot_name="round key"))
    
def aes_circuit(
    quantum_circuit: qiskit.QuantumCircuit, 
    qubit_registers: QubitRegisters, 
    bloch_spheres: BlochSpheres,
    classical_output: qiskit.ClassicalRegister,
    target_cipher: bytes
):
    for cipher_index, bit in enumerate(_INPUT_PLAINTEXT):
        if bit == 1:
            quantum_circuit.x(qubit_registers.plaintext_qubits[cipher_index])
    bloch_spheres.add_circuit_state(quantum_circuit, BlochProperties(plot_name="setup plaintext"))

    # entangle all qubits so they have the same probability
    quantum_circuit.h(qubit_registers.key_qubits)
    bloch_spheres.add_circuit_state(quantum_circuit, BlochProperties(plot_name="super position"))
        # iplace the flag qubit into a state where it can easily flip
    quantum_circuit.x(qubit_registers.flip_qubit)
    bloch_spheres.add_circuit_state(quantum_circuit, BlochProperties(plot_name="flip"))
    quantum_circuit.h(qubit_registers.flip_qubit)
    bloch_spheres.add_circuit_state(quantum_circuit, BlochProperties(plot_name="superposition flip"))
    
    for _ in range(_GROVER_ITERATIONS):
        oracle_find_matching_states(quantum_circuit, qubit_registers, bloch_spheres, target_cipher)
        grover_diffuser(quantum_circuit, qubit_registers, bloch_spheres)
    
    quantum_circuit.measure(qubit_registers.key_qubits, classical_output)
    
def oracle_find_matching_states(quantum_circuit: qiskit.QuantumCircuit, qubit_registers: QubitRegisters, bloch_spheres: BlochSpheres, target_ciper: List[int]):
    # encrypt the plaintext 
    aes_step(quantum_circuit, qubit_registers, bloch_spheres)
    
    # affect control bits so they are all 1s for MCX
    for cipher_index, bit in enumerate(target_ciper):
        if bit == 0:
            quantum_circuit.x(qubit_registers.plaintext_qubits[cipher_index])
    bloch_spheres.add_circuit_state(quantum_circuit, BlochProperties(plot_name="impose cipher on plaintext"))
    
    # based on affect of the target cipher on the paintext, if the sets are all 1, flip the output qubit
    quantum_circuit.mcx(qubit_registers.plaintext_qubits, qubit_registers.flip_qubit)
    bloch_spheres.add_circuit_state(quantum_circuit, BlochProperties(plot_name="flip qubit"))

    # reverse the change of the cipher onto the plaintext
    for cipher_index, bit in enumerate(target_ciper):
        if bit == 0:
            quantum_circuit.x(qubit_registers.plaintext_qubits[cipher_index])
    bloch_spheres.add_circuit_state(quantum_circuit, BlochProperties(plot_name="unimpose cipher on plaintext"))

    # unencrypt the plaintext
    aes_step_inverse(quantum_circuit, qubit_registers, bloch_spheres)

def grover_diffuser(quantum_circuit: qiskit.QuantumCircuit, qubit_registers: QubitRegisters, bloch_spheres: BlochSpheres):
    # reflect the amplitude over the average so matching items probabilities gain amplitude
    
    quantum_circuit.h(qubit_registers.key_qubits)
    bloch_spheres.add_circuit_state(quantum_circuit, BlochProperties(plot_name="superposition"))
    # invert over the x axis
    quantum_circuit.x(qubit_registers.key_qubits)
    bloch_spheres.add_circuit_state(quantum_circuit, BlochProperties(plot_name="invert keys"))
    # apply another hammond on the last key so we can phase flip all bits
    quantum_circuit.h(qubit_registers.key_qubits[-1])
    bloch_spheres.add_circuit_state(quantum_circuit, BlochProperties(plot_name="hernand last qubit"))
    # flip the phase of all bits relative to the last key
    quantum_circuit.mcx(qubit_registers.key_qubits[:-1], qubit_registers.key_qubits[-1])
    bloch_spheres.add_circuit_state(quantum_circuit, BlochProperties(plot_name="flip all qubits"))
    # match the state of the last key to the rest
    quantum_circuit.h(qubit_registers.key_qubits[-1])
    bloch_spheres.add_circuit_state(quantum_circuit, BlochProperties(plot_name="unhernand last qubit"))
    # move from the real states to the phase states
    quantum_circuit.x(qubit_registers.key_qubits)
    bloch_spheres.add_circuit_state(quantum_circuit, BlochProperties(plot_name="revert keys"))
    quantum_circuit.h(qubit_registers.key_qubits)
    bloch_spheres.add_circuit_state(quantum_circuit, BlochProperties(plot_name="de-superposition"))

def ccx_operation(acting_bits: bytes, acting_bit_0, acting_bit_1, target_bit, target_bits = None):
    if target_bits is None:
        target_bits = acting_bits
    
    if acting_bits[acting_bit_0] and acting_bits[acting_bit_1]:
        target_bits[target_bit] = 1-target_bits[target_bit]

def swap_bits(acting_bits, bit_to_swap_0, bit_to_swap_1):
    bit_value_0 = acting_bits[bit_to_swap_0]
    bit_value_1 = acting_bits[bit_to_swap_1]
    acting_bits[bit_to_swap_1] = bit_value_0
    acting_bits[bit_to_swap_0] = bit_value_1

def aes_ecnryption(plain_text: List[int], input_key: List[int]):
    round_key = [plain_text_bit ^ key_bit for plain_text_bit, key_bit in zip(plain_text, input_key)]
    ccx_operation(round_key, 0, 1, 2)
    round_key[1] = 1-round_key[1] if round_key[0] else round_key[1]
    ccx_operation(round_key, 1, 0, 3)
    swap_bits(round_key, 2, 3)
    return round_key

def create_quantum_circuit():
    qubit_registers = QubitRegisters(
        plaintext_qubits = qiskit.QuantumRegister(_PLAINTEXT_QUBITS, name='plaintext_qubits'),
        key_qubits = qiskit.QuantumRegister(_KEY_QUBITS, name='key_qubits'),
        flip_qubit = qiskit.QuantumRegister(_FLIP_QUBIT, name='flip_qubits')
    )
    classical_output = qiskit.ClassicalRegister(4, "classical_output")
    quantum_circuit = qiskit.QuantumCircuit(
        qubit_registers.plaintext_qubits, 
        qubit_registers.key_qubits, 
        qubit_registers.flip_qubit,
        classical_output
    )

    cipher_text = aes_ecnryption(_INPUT_PLAINTEXT, _INPUT_KEY)
    print(_INPUT_KEY)
    bloch_spheres = BlochSpheres(quantum_circuit, 50, BlochProperties(plot_name="Initialized"))


    # Hadamard gate
    aes_circuit(quantum_circuit, qubit_registers, bloch_spheres, classical_output, cipher_text)
    # mutual_entanglement(quantum_circuit)
    sim = Aer.get_backend('aer_simulator')
    compiled_circuit = qiskit.transpile(quantum_circuit, sim)
    job = sim.run(compiled_circuit, shots=1024)
    result = job.result()

    counts = result.get_counts(quantum_circuit)
    print(counts)
    bloch_spheres.animate_bloch_sphere()



def main():
    create_quantum_circuit()
    
if __name__ == "__main__":
    main()