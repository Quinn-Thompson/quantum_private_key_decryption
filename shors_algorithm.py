import qiskit
from qiskit_aer import AerSimulator
from typing import List, Iterable, Optional, Callable
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

_FUNDAMENTAL_BASIS = 3

_COUNTING_QUBITS = _FUNDAMENTAL_BASIS*2 # should be 2N, but can cause bloat
_MULTIPLIER_REGISTER = _FUNDAMENTAL_BASIS
_LEFT_OPERAND_REGISTER = _FUNDAMENTAL_BASIS
_RIGHT_OPERAND_REGISTER = _FUNDAMENTAL_BASIS+1
_CARRY_REGISTER = _FUNDAMENTAL_BASIS
_TEMPORARY_REGISTER = _FUNDAMENTAL_BASIS
_CONTROL_QUBIT = 1

@dataclass
class GateInfo():
    gate_operation: Callable
    name: Optional[str]

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

def flip_bits(quantum_circuit: qiskit.QuantumCircuit, quantum_register: qiskit.QuantumRegister, register_values: List[int]) -> int:
    classical_value = 0
    for location, bit in enumerate(reversed(register_values)):
        if bit == 1:
            classical_value += 2 ** location                
            quantum_circuit.x(quantum_register[location])
    return classical_value


def carry_gate(invert: bool = False, visualization_window: Optional[QuantumCircuitWindow] = None):
    carry_circuit = qiskit.QuantumCircuit(4, name="carry_gate")
    if visualization_window is not None:
        carry_circuit.x(0)
        carry_circuit.x(1)
        carry_circuit.x(2)
        visualization_window.initialize_circuit_properties(carry_circuit, 33, DisplayProperties(plot_name="Initialized"))
    # this is for carrying the values of both registers and the previous into another
    # if both bits are 1, then a carry bit is used, however if either b or a is 1, but not both
    # then b will contain a 1
    # then if the previous carry existed, it will check if b is also 1, and then the
    # carry is flipped, so in the case where either a or b being 1 and the carry is 1, the carry is again 1
    # l   r   c
    # 0 + 0 + 0 = 0
    # 1 + 0 + 0 = 0
    # 0 + 1 + 0 = 0
    # 1 + 1 + 0 = 1
    # 0 + 0 + 1 = 0
    # 1 + 0 + 1 = 1
    # 0 + 1 + 1 = 1
    # 1 + 1 + 1 = 1
    
    # this gate only controls the carry bits, as 1 + 1 + 1 loses information
    
    gates: List[GateInfo] = [
        GateInfo(partial(carry_circuit.ccx, 1, 2, 3), "Carry if R and C are 1"),
        GateInfo(partial(carry_circuit.cx, 1, 2), "Xor L and R"), 
        GateInfo(partial(carry_circuit.ccx, 0, 2, 3), "Our second Carry")
    ]
    if invert:
        gates.reverse()
        
    for gate_info in gates:
        gate_info.gate_operation()
        if visualization_window is not None and gate_info.name is not None:
            visualization_window.add_circuit_state(
                carry_circuit, 
                DisplayProperties(plot_name=gate_info.name),
                fast_state=True,
            )
            
    if visualization_window is None:
        custom_gate = carry_circuit.to_gate()
        custom_gate.label = f"Carry{'†' if invert else ''}"
        return custom_gate

        
def sum_gate(invert: bool = False):
    carry_circuit = qiskit.QuantumCircuit(3, name="sum_gate")
    # this is for checking what the actual output of the addition is
    # because the carry gates only handles the carry bits
    # l + r
    # 0 + 0 = 0
    # 0 + 1 = 1
    # 1 + 0 = 1
    # 1 + 1 = 0
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
    
def adder_gate(invert: bool = False, visualization_window: Optional[QuantumCircuitWindow] = None):
    adder_registers = AdderRegisters(
        left_operand_qubits = qiskit.QuantumRegister(_LEFT_OPERAND_REGISTER, name="left_operand_qubits"),
        right_operand_qubits = qiskit.QuantumRegister(_RIGHT_OPERAND_REGISTER, name="right_operand_qubits"),
        carry_qubits = qiskit.QuantumRegister(_CARRY_REGISTER, name="carry_qubits")
    )
    adder_circuit = qiskit.QuantumCircuit(adder_registers.left_operand_qubits, adder_registers.right_operand_qubits, adder_registers.carry_qubits, name="adder_gate")
    if visualization_window is not None:
        flip_bits(adder_circuit, adder_registers.left_operand_qubits, [1, 0, 1])
        flip_bits(adder_circuit, adder_registers.right_operand_qubits, [0, 1, 1])
        visualization_window.initialize_circuit_properties(adder_circuit, 33, DisplayProperties(plot_name="Initialized"))
    # this is the full adder circuit for providing the output
    # the carry values are handled for each bit, until b[3] holds the final remainder bit
    # then the sum is handled and all but the last carry transforms are undone, so that the right operand becomes a+b, 
    # the carry's are empty except for the next operation and there is a carry overflow bit leftover
    gates: List[GateInfo] = []
    for location in range(_FUNDAMENTAL_BASIS-1):
        gates.append(GateInfo(partial(
            adder_circuit.append, carry_gate(invert), 
            [
                adder_registers.carry_qubits[location], 
                adder_registers.left_operand_qubits[location], 
                adder_registers.right_operand_qubits[location], 
                adder_registers.carry_qubits[location+1]
            ]), f"Compute Carry for Bit {location}")
        )
    gates.append(GateInfo(partial(
        adder_circuit.append, carry_gate(invert), 
        [
            adder_registers.carry_qubits[_FUNDAMENTAL_BASIS-1], 
            adder_registers.left_operand_qubits[_FUNDAMENTAL_BASIS-1], 
            adder_registers.right_operand_qubits[_FUNDAMENTAL_BASIS-1], 
            adder_registers.right_operand_qubits[_FUNDAMENTAL_BASIS]
        ]), "Compute Last Carry Bit")
    ) 

    gates.append(GateInfo(partial(adder_circuit.cx, adder_registers.left_operand_qubits[_FUNDAMENTAL_BASIS-1], adder_registers.right_operand_qubits[_FUNDAMENTAL_BASIS-1]), "Maintain Last Carry"))
    for location in range(_FUNDAMENTAL_BASIS-1, 0, -1):
        gates.append(GateInfo(partial(
            adder_circuit.append, sum_gate(invert), 
            [
                adder_registers.carry_qubits[location], 
                adder_registers.left_operand_qubits[location], 
                adder_registers.right_operand_qubits[location]
            ]), f"Sum Actual Location {location} Bit Values") 
        )
        gates.append(GateInfo(partial(
            adder_circuit.append, carry_gate(not invert), 
            [
                adder_registers.carry_qubits[location-1], 
                adder_registers.left_operand_qubits[location-1], 
                adder_registers.right_operand_qubits[location-1], 
                adder_registers.carry_qubits[location]
            ]), f"Undo Carry {location} to Restore State")
        )
    gates.append(GateInfo(partial(
        adder_circuit.append, sum_gate(invert), 
        [
            adder_registers.carry_qubits[0], 
            adder_registers.left_operand_qubits[0], 
            adder_registers.right_operand_qubits[0]
        ]), "Sum First Bits")
    ) 

    
    if invert:
        gates.reverse()
        
    for gate_info in gates:
        gate_info.gate_operation()
        if visualization_window is not None and gate_info.name is not None:
            visualization_window.add_circuit_state(
                adder_circuit, 
                DisplayProperties(plot_name=gate_info.name),
                fast_state=True,
            )
        
    if visualization_window is None:
        custom_gate = adder_circuit.to_gate()
        custom_gate.label = f"Adder{'†' if invert else ''}"
        return custom_gate

def modulated_adder_gate(classical_modulated_value: int, invert: bool=False, visualization_window: Optional[QuantumCircuitWindow] = None):
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
    if visualization_window is not None:
        flip_bits(mod_adder_circuit, modulated_adder_registers.left_operand_qubits, [1, 1, 0])
        flip_bits(mod_adder_circuit, modulated_adder_registers.right_operand_qubits, [1, 1, 0])
        flip_bits(mod_adder_circuit, modulated_adder_registers.temporary_qubits, [1, 0, 1])
        visualization_window.initialize_circuit_properties(mod_adder_circuit, 33, DisplayProperties(plot_name="Initialized"))
    # the last two adders are just to uncompute the control qubit
    
    # a+b
    gates: List[GateInfo] = [
        GateInfo(partial(mod_adder_circuit.append, adder_gate(invert), list(modulated_adder_registers)), "Add A and B")
    ]
    # a is now N, and N is now a
    swap_gates: List[GateInfo] = []
    operation_name = None
    for location in range(_FUNDAMENTAL_BASIS):
        if location == _FUNDAMENTAL_BASIS - 1:
            operation_name = "Swap A for N"
        gates.append(GateInfo(partial(mod_adder_circuit.swap, modulated_adder_registers.temporary_qubits[location], modulated_adder_registers.left_operand_qubits[location]), operation_name))
        swap_gates.append(gates[-1])
    # subtract N from (a+b), because a < N and b < N from being factored in by N, 
    # a + b - N will never falsely trigger the overflow qubit
    # as such, if the overflow is zero after subtraction, then the value a + b - N is positive
    # thus it is the end result from modulation
    # however if the overflow is 1, then we have dipped into 2's complement, so we must add N back
    # to modulate
    gates.extend(
        [
            GateInfo(partial(mod_adder_circuit.append, adder_gate(not invert), list(modulated_adder_registers)), "Subtract N from B"), 
            GateInfo(partial(mod_adder_circuit.x, modulated_adder_registers.right_operand_qubits[_FUNDAMENTAL_BASIS]), None),
            GateInfo(partial(mod_adder_circuit.cx, modulated_adder_registers.right_operand_qubits[_FUNDAMENTAL_BASIS], modulated_adder_registers.control_qubit[0]), None),
            GateInfo(partial(mod_adder_circuit.x, modulated_adder_registers.right_operand_qubits[_FUNDAMENTAL_BASIS]), "Flip Control for Re-Adding"),
        ]
    )
    # zero out N bits if control qubit is 1
    zero_gates: List[GateInfo] = []
    modulation_bits: str = "".join(reversed(format(classical_modulated_value, f"0{_FUNDAMENTAL_BASIS}b"))) 
    for location, bit in enumerate(modulation_bits):
        if bit == "1":
            gates.append(GateInfo(partial(mod_adder_circuit.cx, modulated_adder_registers.control_qubit[0], modulated_adder_registers.left_operand_qubits[location]), f"Zero Out Bit {location}"))
            zero_gates.append(gates[-1])
    
    # add N back in if it hasn't been zerod out
    gates.append(
        GateInfo(partial(mod_adder_circuit.append, adder_gate(invert), list(modulated_adder_registers)), "Add N if Not Zeroed")
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
            GateInfo(partial(mod_adder_circuit.append, adder_gate(not invert), list(modulated_adder_registers)), None), 
            GateInfo(partial(mod_adder_circuit.cx, modulated_adder_registers.right_operand_qubits[_FUNDAMENTAL_BASIS], modulated_adder_registers.control_qubit[0]), "Reset Control"),
            GateInfo(partial(mod_adder_circuit.append, adder_gate(invert), list(modulated_adder_registers)), None), 
        ]
    )
    
    if invert:
        gates.reverse()
        
    for gate_info in gates:
        gate_info.gate_operation()
        if visualization_window is not None and gate_info.name is not None:
            visualization_window.add_circuit_state(
                mod_adder_circuit, 
                DisplayProperties(plot_name=gate_info.name),
                fast_state=True,
            )
        
    if visualization_window is None:
        custom_gate = mod_adder_circuit.to_gate()
        custom_gate.label = f"ModAdder{'†' if invert else ''}"
        return custom_gate

def controlled_mod_mult_gate(classical_modulated_value: int, classical_base_value: int, invert: bool=False, visualization_window: Optional[QuantumCircuitWindow] = None):
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
    
    if visualization_window is not None:
        flip_bits(cntrl_mod_mult_circuit, cntrl_mod_mult_registers.application_qubit, [0])
        flip_bits(cntrl_mod_mult_circuit, cntrl_mod_mult_registers.multiplier_qubits, [0, 1, 0])
        flip_bits(cntrl_mod_mult_circuit, cntrl_mod_mult_registers.temporary_qubits, [1, 0, 1])
        visualization_window.initialize_circuit_properties(cntrl_mod_mult_circuit, 33, DisplayProperties(plot_name="Initialized"))
    
    # for the hadamard state, we are considering all possible exponential from the application qubit
    # so, in the case where the application qubit is 1, we are using this qubit in the exponential
    # in this case, the modular multiplication can be thought of as zm = 2**0 * m * z0 + 2**1 * m * z1
    # so, 0101 * 0110 = 2**0 * 0110 * 1 + 2**1 * 0110 * 0 + 2**2 * 0110 * 1 + 2**3 * 0110 * 0
    # this means that z is conditional, as it is either 1 or 0, and alongside this, m can be made exponential
    # by moving those cx gates along the adder circuit register, as when it 
    # moves a significant bit it now is a power of two greater during the addition process
    # so, we calculate 2**k * m and create gates located where the output is based on the zk value also conditioned on c
    # however, we need to ensure it both stays within the bounds and also follows z*m mod N, so instead of
    # calculating z*m, z*m mod N is used to calculate gate positions on the adder gate
    gates: List[GateInfo] = []
    for exponent in range(_FUNDAMENTAL_BASIS):
        modulated_coprime = classical_base_value * (2 ** exponent) % classical_modulated_value
        control_gates = []
        for location, bit in enumerate(reversed(format(modulated_coprime, f"0{_FUNDAMENTAL_BASIS}b"))):
            if bit == "1":
                gates.append(GateInfo(partial(
                    cntrl_mod_mult_circuit.ccx, 
                    cntrl_mod_mult_registers.application_qubit[0], 
                    cntrl_mod_mult_registers.multiplier_qubits[exponent], 
                    cntrl_mod_mult_registers.left_operand_qubits[location]
                    ), "Control Bit Location Multiplier")
                )
                control_gates.append(gates[-1])
        
        gates.append(GateInfo(partial(cntrl_mod_mult_circuit.append, modulated_adder_gate(classical_modulated_value, invert), list(cntrl_mod_mult_registers)), "Modulated Addition"))
        
        control_gates.reverse()
        gates.extend(control_gates)

    # in the case in which the control bit is zero, flip it, and set the b register to z so it is maintained
    # within the circuit
    gates.append(GateInfo(partial(cntrl_mod_mult_circuit.x, cntrl_mod_mult_registers.application_qubit[0]), None))
    for location in range(_FUNDAMENTAL_BASIS):
        gates.append(GateInfo(partial(cntrl_mod_mult_circuit.ccx, cntrl_mod_mult_registers.application_qubit[0], cntrl_mod_mult_registers.multiplier_qubits[location], cntrl_mod_mult_registers.right_operand_qubits[location]), None))
    gates.append(GateInfo(partial(cntrl_mod_mult_circuit.x, cntrl_mod_mult_registers.application_qubit[0]), "Push Z into B"))

    if invert:
        gates.reverse()
        
    for gate_info in gates:
        gate_info.gate_operation()
        if visualization_window is not None and gate_info.name is not None:
            visualization_window.add_circuit_state(
                cntrl_mod_mult_circuit, 
                DisplayProperties(plot_name=gate_info.name),
                fast_state=True,
            )
        
    if visualization_window is None:
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
    # x is the control for the exponential value
    # first, the multiplication mod gate is used on Z, with m being a and Z being 00...01, meaning we get a Mod N as our b register 
    # this is then swapped with Z, so Z holds a mod N and b holds 1, thus when the inverse is applied, it reverse the global state
    # (besides the control gate) back to it's original form.
    # now, however, there is an entangled interference kickback to the control register, which reflects the periodicity of x + r
    # for all possible x's

    for counting_qubit_number in range(_COUNTING_QUBITS):
        exponential_component = classical_base_value ** (2 ** counting_qubit_number)
        gates.append(partial(mod_exp_circuit.append, controlled_mod_mult_gate(classical_modulated_value, exponential_component), mod_exp_registers[counting_qubit_number]))

        for location in range(_FUNDAMENTAL_BASIS):
            gates.append(partial(mod_exp_circuit.cswap, mod_exp_registers.counting_qubits[counting_qubit_number], mod_exp_registers.multiplier_qubits[location], mod_exp_registers.right_operand_qubits[location]))
        gates.append(partial(mod_exp_circuit.append, controlled_mod_mult_gate(classical_modulated_value, pow(exponential_component, -1, classical_modulated_value), invert=True), mod_exp_registers[counting_qubit_number]))
            
    
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
    bloch_sphere_window = QuantumCircuitWindow()

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
    prime_1 = 2
    prime_2 = 3
    coprime = 5
    product_of_primes = prime_1 * prime_2

    bloch_sphere_window.initialize_circuit_properties(quantum_circuit, 33, DisplayProperties(plot_name="Initialized"))
    print("Running Shors Algorithm")
    quantum_circuit.h(qubit_registers.counting_qubits)
    quantum_circuit.x(qubit_registers.multiplier_qubits[0])
    for location, bit in enumerate(format(product_of_primes, f"0{_FUNDAMENTAL_BASIS}b")):
        if bit == "1":
            quantum_circuit.x(qubit_registers.temporary_qubits[location])
    quantum_circuit.append(modular_exponentiation(product_of_primes, coprime), list(qubit_registers))
    bloch_sphere_window.add_circuit_state(
        quantum_circuit, 
        DisplayProperties(plot_name="modular_exp"),
        fast_state=True,
    )
    quantum_circuit.append(inverse_qft(_COUNTING_QUBITS), qubit_registers.counting_qubits)
    bloch_sphere_window.add_circuit_state(
        quantum_circuit, 
        DisplayProperties(plot_name="inverse_QPE"),
        fast_state=True,
    )
    # quantum_circuit.measure(qubit_registers.counting_qubits, classical_output)

    # sim = AerSimulator(method='matrix_product_state', matrix_product_state_max_bond_dimension=4096, seed_simulator=42)
    # compiled_circuit = qiskit.transpile(quantum_circuit, sim)
    # counts = {}
    # job = sim.run(compiled_circuit, shots=1024)
    # previous_status = None
    # while not job.done():
    #     status = job.status()
    #     if status != previous_status:
    #         print("Waiting... job status:", status, flush=True)
    #         previous_status = status
    #     time.sleep(1)
    # result = job.result()
    # counts |= result.get_counts(quantum_circuit)
    print("Finished Shors Algorithm")
    # print(counts)
    # found_factor = False
    # attempt = 0
    # for measure_key, measure_value in sorted(counts.items(), key=lambda item: item[1], reverse=True)[:7]:
    #     measured_int = int(measure_key, 2)
    #     if measured_int == 0 or measured_int == 2**_COUNTING_QUBITS - 1:
    #         continue
        
    #     phase = measured_int / 2**_COUNTING_QUBITS
    #     frac = Fraction(phase).limit_denominator(product_of_primes)
    #     r = frac.denominator
    #     if r % 2 != 0 or r <= 1:
    #         continue  # skip odd periods
    #     attempt += 1
    #     if pow(coprime, r, product_of_primes) == 1:
    #         x = pow(coprime, r // 2, product_of_primes)
    #         factors = [math.gcd(x - 1, product_of_primes), math.gcd(x + 1, product_of_primes)]
            
    #         if 1 < factors[0] < product_of_primes:
    #             print(f"Factor found: {factors[0]}")
    #             found_factor = True
    #         if 1 < factors[1] < product_of_primes:
    #             print(f"Factor found: {factors[1]}")
    #             found_factor = True
    #     if found_factor:
    #         print(f"Found factor on proper phase attempt {attempt} for value {measure_key}")
    #         break
    # plot_histogram(counts)
    # plt.show()
    bloch_sphere_window.animate_circuit()
    bloch_sphere_window.app.exec()
