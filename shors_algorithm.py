import qiskit
from qiskit_aer import AerSimulator
from typing import List, Iterable
from dataclasses import dataclass
import numpy as np
from functools import partial
from qiskit.circuit import Qubit
import matplotlib.pylab as plt
import math
import time
from fractions import Fraction
from qiskit.visualization import plot_histogram
from copy import deepcopy

from quantum_visualization.gui_backend.sub_backend.visualize_qubits import DisplayProperties
from quantum_visualization.gui_backend.main_backend import QuantumCircuitWindow

_FUNDAMENTAL_BASIS = 4

_COUNTING_QUBITS = 6 # should be 2N, but can cause bloat
_MULTIPLIER_REGISTER = _FUNDAMENTAL_BASIS
_LEFT_OPERAND_REGISTER = _FUNDAMENTAL_BASIS
_RIGHT_OPERAND_REGISTER = _FUNDAMENTAL_BASIS+1
_CARRY_REGISTER = _FUNDAMENTAL_BASIS
_TEMPORARY_REGISTER = _FUNDAMENTAL_BASIS
_CONTROL_QUBIT = 1

@dataclass
class QubitRegisters():
    left_operand_qubits: qiskit.QuantumRegister
    right_operand_qubits: qiskit.QuantumRegister
    carry_qubits: qiskit.QuantumRegister
    temporary_qubits: qiskit.QuantumRegister
    control_qubit: qiskit.QuantumRegister
    multiplier_qubits: qiskit.QuantumRegister
    counting_qubits: qiskit.QuantumRegister

    def __iter__(self) -> Iterable[Qubit]:
        return iter(
            list(self.left_operand_qubits) 
            + list(self.right_operand_qubits) 
            + list(self.carry_qubits) 
            + list(self.temporary_qubits) 
            + list(self.control_qubit) 
            + list(self.multiplier_qubits)
            + list(self.counting_qubits)
        )

@dataclass
class AdderRegisters():
    left_operand_qubits: qiskit.QuantumRegister
    right_operand_qubits: qiskit.QuantumRegister
    carry_qubits: qiskit.QuantumRegister

@dataclass
class ModulatedAdderRegisters():
    left_operand_qubits: qiskit.QuantumRegister
    right_operand_qubits: qiskit.QuantumRegister
    carry_qubits: qiskit.QuantumRegister
    temporary_qubits: qiskit.QuantumRegister
    control_qubit: qiskit.QuantumRegister

    def __iter__(self) -> Iterable[Qubit]:
        return iter(
            list(self.left_operand_qubits) 
            + list(self.right_operand_qubits) 
            + list(self.carry_qubits) 
        )

@dataclass
class ControlModMultRegisters():
    left_operand_qubits: qiskit.QuantumRegister
    right_operand_qubits: qiskit.QuantumRegister
    carry_qubits: qiskit.QuantumRegister
    temporary_qubits: qiskit.QuantumRegister
    control_qubit: qiskit.QuantumRegister
    multiplier_qubits: qiskit.QuantumRegister
    application_qubit: qiskit.QuantumRegister
    
    def __iter__(self) -> Iterable[Qubit]:
        return iter(
            list(self.left_operand_qubits) 
            + list(self.right_operand_qubits) 
            + list(self.carry_qubits) 
            + list(self.temporary_qubits) 
            + list(self.control_qubit) 
        )

@dataclass
class ModularExponentiationRegisters():
    left_operand_qubits: qiskit.QuantumRegister
    right_operand_qubits: qiskit.QuantumRegister
    carry_qubits: qiskit.QuantumRegister
    temporary_qubits: qiskit.QuantumRegister
    control_qubit: qiskit.QuantumRegister
    multiplier_qubits: qiskit.QuantumRegister
    counting_qubits: qiskit.QuantumRegister
    
    def __getitem__(self, item: int) -> List[Qubit]:
        return (
            list(self.left_operand_qubits) 
            + list(self.right_operand_qubits) 
            + list(self.carry_qubits) 
            + list(self.temporary_qubits) 
            + list(self.control_qubit) 
            + list(self.multiplier_qubits)
            + [list(self.counting_qubits)[item]]
        )

def carry_gate(invert: bool = False):
    carry_circuit = qiskit.QuantumCircuit(4, name="carry_gate")
    # this is for carrying the values of both registers and the previous into another
    gates = [
        partial(carry_circuit.ccx, 1, 2, 3), 
        partial(carry_circuit.cx, 1, 2), 
        partial(carry_circuit.ccx, 0, 2, 3), 
    ]
    if invert:
        gates.reverse()
        
    for gate in gates:
        gate()
        
    custom_gate = carry_circuit.to_gate()
    custom_gate.label = f"Carry{'†' if invert else ''}"
    return custom_gate

        
def sum_gate(invert: bool = False):
    carry_circuit = qiskit.QuantumCircuit(3, name="sum_gate")
    # this is for checking what the actual output of the addition is
    gates = [
        partial(carry_circuit.cx, 1, 2), 
        partial(carry_circuit.cx, 0, 2), 
    ]
    if invert:
        gates.reverse()
        
    for gate in gates:
        gate()
        
    custom_gate = carry_circuit.to_gate()
    custom_gate.label = f"Sum{'†' if invert else ''}"
    return custom_gate
    
def adder_gate(invert: bool = False):
    adder_registers = AdderRegisters(
        left_operand_qubits = qiskit.QuantumRegister(_LEFT_OPERAND_REGISTER, name="left_operand_qubits"),
        right_operand_qubits = qiskit.QuantumRegister(_RIGHT_OPERAND_REGISTER, name="right_operand_qubits"),
        carry_qubits = qiskit.QuantumRegister(_CARRY_REGISTER, name="carry_qubits")
    )
    adder_circuit = qiskit.QuantumCircuit(adder_registers.left_operand_qubits, adder_registers.right_operand_qubits, adder_registers.carry_qubits, name="adder_gate")
    # this is the full adder circuit for providing the output
    # the carry values are handled for each bit, until b[3] holds the final remainder bit
    # then the sum is handled and the carry transformations are undone, so that the right operand becomes a+b
    gates = []
    for location in range(_FUNDAMENTAL_BASIS-1):
        gates.append(partial(adder_circuit.append, carry_gate(invert), [adder_registers.carry_qubits[location], adder_registers.left_operand_qubits[location], adder_registers.right_operand_qubits[location], adder_registers.carry_qubits[location+1]])), 
    gates.append(partial(adder_circuit.append, carry_gate(invert), [adder_registers.carry_qubits[_FUNDAMENTAL_BASIS-1], adder_registers.left_operand_qubits[_FUNDAMENTAL_BASIS-1], adder_registers.right_operand_qubits[_FUNDAMENTAL_BASIS-1], adder_registers.right_operand_qubits[_FUNDAMENTAL_BASIS]])), 
    gates.append(partial(adder_circuit.cx, adder_registers.left_operand_qubits[_FUNDAMENTAL_BASIS-1], adder_registers.right_operand_qubits[_FUNDAMENTAL_BASIS-1]))
    for location in range(_FUNDAMENTAL_BASIS-1, 0, -1):
        gates.append(partial(adder_circuit.append, sum_gate(invert), [adder_registers.carry_qubits[location], adder_registers.left_operand_qubits[location], adder_registers.right_operand_qubits[location]])), 
        gates.append(partial(adder_circuit.append, carry_gate(not invert), [adder_registers.carry_qubits[location-1], adder_registers.left_operand_qubits[location-1], adder_registers.right_operand_qubits[location-1], adder_registers.carry_qubits[location]])), 
    gates.append(partial(adder_circuit.append, sum_gate(invert), [adder_registers.carry_qubits[0], adder_registers.left_operand_qubits[0], adder_registers.right_operand_qubits[0]])), 

    
    if invert:
        gates.reverse()
        
    for gate in gates:
        gate()
        
    custom_gate = adder_circuit.to_gate()
    custom_gate.label = f"Adder{'†' if invert else ''}"
    return custom_gate

def modulated_adder_gate(classical_modulated_value: int, invert: bool=False):
    modulated_adder_registers = ModulatedAdderRegisters(
        left_operand_qubits = qiskit.QuantumRegister(_LEFT_OPERAND_REGISTER, name="left_operand_qubits"),
        right_operand_qubits = qiskit.QuantumRegister(_RIGHT_OPERAND_REGISTER, name="right_operand_qubits"),
        carry_qubits = qiskit.QuantumRegister(_CARRY_REGISTER, name="carry_qubits"),
        temporary_qubits = qiskit.QuantumRegister(_TEMPORARY_REGISTER, name="remainder_qubits"),
        control_qubit = qiskit.QuantumRegister(_CONTROL_QUBIT, name="control_qubit"), 
        
    )
    mod_adder_circuit = qiskit.QuantumCircuit(
        modulated_adder_registers.left_operand_qubits, 
        modulated_adder_registers.right_operand_qubits, 
        modulated_adder_registers.carry_qubits, 
        modulated_adder_registers.temporary_qubits,
        modulated_adder_registers.control_qubit,
        name="modulated_adder_gate"
    )
    # the last two adders are just to uncompute the control qubit
    
    # a+b
    gates = [
        partial(mod_adder_circuit.append, adder_gate(invert), list(modulated_adder_registers))
    ]
    # a is now N, and N is now a
    swap_gates = []
    for location in range(_FUNDAMENTAL_BASIS):
        gates.append(partial(mod_adder_circuit.swap, modulated_adder_registers.temporary_qubits[location], modulated_adder_registers.left_operand_qubits[location]))
        swap_gates.append(gates[-1])
    # subtract N from (a+b), because a < N and b < N from being factored in by N, 
    # a + b - N will never falsely trigger the overflow qubit
    # as such, if the overflow is zero after subtraction, then the value a + b - N is positive
    # thus it is the end result from modulation
    # however if the overflow is 1, then we have dipped into 2's complement, so we must add N back
    # to modulate
    gates.extend(
        [
            partial(mod_adder_circuit.append, adder_gate(not invert), list(modulated_adder_registers)), 
            partial(mod_adder_circuit.x, modulated_adder_registers.right_operand_qubits[_FUNDAMENTAL_BASIS]),
            partial(mod_adder_circuit.cx, modulated_adder_registers.right_operand_qubits[_FUNDAMENTAL_BASIS], modulated_adder_registers.control_qubit[0]),
            partial(mod_adder_circuit.x, modulated_adder_registers.right_operand_qubits[_FUNDAMENTAL_BASIS]),
        ]
    )
    # zero out N bits if control qubit is 1
    zero_gates = []
    modulation_bits: str = "".join(reversed(format(classical_modulated_value, f"0{_FUNDAMENTAL_BASIS}b"))) 
    for location, bit in enumerate(modulation_bits):
        if bit == "1":
            gates.append(partial(mod_adder_circuit.cx, modulated_adder_registers.control_qubit[0], modulated_adder_registers.left_operand_qubits[location]))
            zero_gates.append(gates[-1])
    
    # add N back in if it hasn't been zerod out
    gates.append(
        partial(mod_adder_circuit.append, adder_gate(invert), list(modulated_adder_registers)), 
    )
    
    # reset the N bits
    zero_gates.reverse()
    gates.extend(zero_gates)
    
    # swap N back to N register, and a back to a register
    swap_gates.reverse()
    gates.extend(swap_gates)
    
    # block 2, used just to reset the control qubit
    gates.extend(
        [
            partial(mod_adder_circuit.append, adder_gate(not invert), list(modulated_adder_registers)), 
            partial(mod_adder_circuit.cx, modulated_adder_registers.right_operand_qubits[_FUNDAMENTAL_BASIS], modulated_adder_registers.control_qubit[0]),
            partial(mod_adder_circuit.append, adder_gate(invert), list(modulated_adder_registers)), 
        ]
    )
    
    if invert:
        gates.reverse()
        
    for gate in gates:
        gate()
        
    custom_gate = mod_adder_circuit.to_gate()
    custom_gate.label = f"ModAdder{'†' if invert else ''}"
    return custom_gate

def controlled_mod_mult_gate(classical_modulated_value: int, classical_base_value: int, invert: bool=False):
    cntrl_mod_mult_registers = ControlModMultRegisters(
        left_operand_qubits = qiskit.QuantumRegister(_LEFT_OPERAND_REGISTER, name="left_operand_qubits"),
        right_operand_qubits = qiskit.QuantumRegister(_RIGHT_OPERAND_REGISTER, name="right_operand_qubits"),
        carry_qubits = qiskit.QuantumRegister(_CARRY_REGISTER, name="carry_qubits"),
        temporary_qubits = qiskit.QuantumRegister(_TEMPORARY_REGISTER, name="remainder_qubits"),
        control_qubit = qiskit.QuantumRegister(_CONTROL_QUBIT, name="control_qubit"), 
        multiplier_qubits= qiskit.QuantumRegister(_MULTIPLIER_REGISTER, name="multiplier_qubits"),
        application_qubit= qiskit.QuantumRegister(1, name="application_qubit"),
    )
    cntrl_mod_mult_circuit = qiskit.QuantumCircuit(
        cntrl_mod_mult_registers.left_operand_qubits, 
        cntrl_mod_mult_registers.right_operand_qubits, 
        cntrl_mod_mult_registers.carry_qubits, 
        cntrl_mod_mult_registers.temporary_qubits,
        cntrl_mod_mult_registers.control_qubit,
        cntrl_mod_mult_registers.multiplier_qubits,
        cntrl_mod_mult_registers.application_qubit,
        name="controlled_mod_mult_gate"
    )
    
    # for the hadamard state, we are considering all possible exponentials from the application qubit
    # so, in the case where the application qubit is 1, we are using this qubit in the exponential
    # in this case, the modular multiplication is used because z 
    gates = []
    for exponent in range(_FUNDAMENTAL_BASIS):
        modulated_coprime = classical_base_value * (2 ** exponent) % classical_modulated_value
        control_gates = []
        for location, bit in enumerate(reversed(format(modulated_coprime, f"0{_FUNDAMENTAL_BASIS}b"))):
            if bit == "1":
                gates.append(partial(cntrl_mod_mult_circuit.ccx, cntrl_mod_mult_registers.application_qubit[0], cntrl_mod_mult_registers.multiplier_qubits[exponent], cntrl_mod_mult_registers.left_operand_qubits[location]))
                control_gates.append(gates[-1])
        
        gates.append(partial(cntrl_mod_mult_circuit.append, modulated_adder_gate(classical_modulated_value, invert), list(cntrl_mod_mult_registers)))
        
        control_gates.reverse()
        gates.extend(control_gates)

    # in the case in which the control bit is zero, flip it, and set the b register to z so it is maintained
    # within the circuit
    gates.append(partial(cntrl_mod_mult_circuit.x, cntrl_mod_mult_registers.application_qubit[0]))
    for location in range(_FUNDAMENTAL_BASIS):
        gates.append(partial(cntrl_mod_mult_circuit.ccx, cntrl_mod_mult_registers.application_qubit[0], cntrl_mod_mult_registers.multiplier_qubits[location], cntrl_mod_mult_registers.right_operand_qubits[location]))
    gates.append(partial(cntrl_mod_mult_circuit.x, cntrl_mod_mult_registers.application_qubit[0]))

    
    if invert:
        gates.reverse()
    
    for gate in gates:
        gate()
        
    custom_gate = cntrl_mod_mult_circuit.to_gate()
    return custom_gate

def modular_exponentiation(classical_modulated_value: int, classical_base_value: int):
    mod_exp_registers = ModularExponentiationRegisters(
        left_operand_qubits = qiskit.QuantumRegister(_LEFT_OPERAND_REGISTER, name="left_operand_qubits"),
        right_operand_qubits = qiskit.QuantumRegister(_RIGHT_OPERAND_REGISTER, name="right_operand_qubits"),
        carry_qubits = qiskit.QuantumRegister(_CARRY_REGISTER, name="carry_qubits"),
        temporary_qubits = qiskit.QuantumRegister(_TEMPORARY_REGISTER, name="remainder_qubits"),
        control_qubit = qiskit.QuantumRegister(_CONTROL_QUBIT, name="control_qubit"), 
        multiplier_qubits= qiskit.QuantumRegister(_MULTIPLIER_REGISTER, name="multiplier_qubits"),
        counting_qubits= qiskit.QuantumRegister(_COUNTING_QUBITS, name="counting_qubits"),
    )
    mod_exp_circuit = qiskit.QuantumCircuit(
        mod_exp_registers.left_operand_qubits, 
        mod_exp_registers.right_operand_qubits, 
        mod_exp_registers.carry_qubits, 
        mod_exp_registers.temporary_qubits,
        mod_exp_registers.control_qubit,
        mod_exp_registers.multiplier_qubits,
        mod_exp_registers.counting_qubits,
        name="modular_exponentiation"
    )
    gates = []
    for counting_qubit_number in range(_COUNTING_QUBITS):
        gates.append(partial(mod_exp_circuit.append, controlled_mod_mult_gate(classical_modulated_value, classical_base_value), mod_exp_registers[counting_qubit_number]))

        for location in range(_FUNDAMENTAL_BASIS):
            gates.append(partial(mod_exp_circuit.cswap, mod_exp_registers.counting_qubits[counting_qubit_number], mod_exp_registers.multiplier_qubits[location], mod_exp_registers.right_operand_qubits[location]))
        gates.append(partial(mod_exp_circuit.append, controlled_mod_mult_gate(classical_modulated_value, classical_base_value, invert=True), mod_exp_registers[counting_qubit_number]))
            
    
    for gate in gates:
        gate()

    custom_gate = mod_exp_circuit.to_gate()
    custom_gate.label = "ModExp"
    return custom_gate

def inverse_qft(number_of_qubits: int):
    invert_quantum_fourier_circuit = qiskit.QuantumCircuit(number_of_qubits)  
    for counting_qubit in range(number_of_qubits):
        for k in range(counting_qubit):
            angle = -np.pi / (2 ** (counting_qubit - k))
            invert_quantum_fourier_circuit.cp(angle, counting_qubit, k)
        invert_quantum_fourier_circuit.h(counting_qubit)
        
    for i in range(number_of_qubits // 2):
        invert_quantum_fourier_circuit.swap(i, number_of_qubits - i - 1)

    invert_quantum_fourier_circuit.name = "QFT†"
    custom_gate = invert_quantum_fourier_circuit.to_gate()
    return custom_gate

def create_shors_circuit():
    # bloch_sphere_window = QuantumCircuitWindow()
    qubit_registers = QubitRegisters(
        left_operand_qubits = qiskit.QuantumRegister(_LEFT_OPERAND_REGISTER, name="left_operand_qubits"),
        right_operand_qubits = qiskit.QuantumRegister(_RIGHT_OPERAND_REGISTER, name="right_operand_qubits"),
        carry_qubits = qiskit.QuantumRegister(_CARRY_REGISTER, name="carry_qubits"),
        temporary_qubits = qiskit.QuantumRegister(_TEMPORARY_REGISTER, name="remainder_qubits"),
        control_qubit = qiskit.QuantumRegister(_CONTROL_QUBIT, name="control_qubit"), 
        multiplier_qubits= qiskit.QuantumRegister(_MULTIPLIER_REGISTER, name="multiplier_qubits"),
        counting_qubits= qiskit.QuantumRegister(_COUNTING_QUBITS, name="counting_qubits"),
    )
    classical_output = qiskit.ClassicalRegister(_COUNTING_QUBITS, "classical_output")
    quantum_circuit = qiskit.QuantumCircuit(
        qubit_registers.left_operand_qubits, 
        qubit_registers.right_operand_qubits, 
        qubit_registers.carry_qubits, 
        qubit_registers.temporary_qubits,
        qubit_registers.control_qubit,
        qubit_registers.multiplier_qubits,
        qubit_registers.counting_qubits,
        classical_output,
        name="circuit"
    )
    prime_1 = 3
    prime_2 = 5
    coprime = 7
    product_of_primes = prime_1 * prime_2
    # bloch_sphere_window.initialize_circuit_properties(quantum_circuit, 33, DisplayProperties(plot_name="Initialized"))
    quantum_circuit.h(qubit_registers.counting_qubits)
    quantum_circuit.x(qubit_registers.multiplier_qubits[0])
    for location, bit in enumerate(format(product_of_primes, f"0{_FUNDAMENTAL_BASIS}b")):
        if bit == "1":
            quantum_circuit.x(qubit_registers.temporary_qubits[location])
    quantum_circuit.append(modular_exponentiation(product_of_primes, coprime), list(qubit_registers))
    quantum_circuit.append(inverse_qft(_COUNTING_QUBITS), qubit_registers.counting_qubits)
    quantum_circuit.measure(qubit_registers.counting_qubits, classical_output)
    # bloch_sphere_window.add_circuit_state(quantum_circuit, DisplayProperties(plot_name="modular_exp"))
    # quantum_circuit.decompose().draw('mpl')
    # plt.show()
    sim = AerSimulator(method='matrix_product_state', matrix_product_state_max_bond_dimension=4096, seed_simulator=42)
    compiled_circuit = qiskit.transpile(quantum_circuit, sim)
    counts = {}
    job = sim.run(compiled_circuit, shots=100000)
    previous_status = None
    while not job.done():
        status = job.status()
        if status != previous_status:
            print("Waiting... job status:", status, flush=True)
            previous_status = status
        time.sleep(1)
    result = job.result()
    counts |= result.get_counts(quantum_circuit)

    print(counts)

    for measure in sorted(counts.items(), key=lambda item: item[1], reverse=True)[:3]:
        # implemented by chatgpt
        measured_int = int(measure[0], 2)
        phase = measured_int / 2**3
        frac = Fraction(phase).limit_denominator(product_of_primes)
        r = frac.denominator
        if pow(coprime, r, product_of_primes) == 1:
            x = pow(coprime, r // 2, product_of_primes)
            factors = [math.gcd(x - 1, product_of_primes), math.gcd(x + 1, product_of_primes)]
            
            if 1 < factors[0] < product_of_primes:
                print(f"Factor found: {factors[0]}")
            if 1 < factors[1] < product_of_primes:
                print(f"Factor found: {factors[1]}")
    plot_histogram(counts)
    plt.show()
    # bloch_sphere_window.animate_circuit()
    # bloch_sphere_window.app.exec()
