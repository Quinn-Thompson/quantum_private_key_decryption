import qiskit
from qiskit_aer import AerSimulator
from typing import List, Iterable, Optional, Callable, Dict
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
class SumRegisters():
    left_operand_qubit: qiskit.QuantumRegister
    right_operand_qubit: qiskit.QuantumRegister
    output_qubit: qiskit.QuantumRegister

@dataclass
class CarryRegisters():
    input_carry_qubit: qiskit.QuantumRegister
    left_operand_qubit: qiskit.QuantumRegister
    right_operand_qubit: qiskit.QuantumRegister
    output_carry_qubit: qiskit.QuantumRegister


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

def int_to_binary_list(n: int, length: Optional[int] = None) -> List[int]:
    binary_str = bin(n)[2:] 
    if length:
        binary_str = binary_str.zfill(length)
    return [int(bit) for bit in binary_str]

def flip_bits(quantum_circuit: qiskit.QuantumCircuit, quantum_register: qiskit.QuantumRegister, register_values: List[int]) -> int:
    classical_value = 0
    for location, bit in enumerate(reversed(register_values)):
        if bit == 1:
            classical_value += 2 ** location                
            quantum_circuit.x(quantum_register[location])
    return classical_value


def convert_to_gates(
    quantum_circuit: qiskit.QuantumCircuit, 
    invert: bool, 
    gates: List[GateInfo], 
    name: str,
    visualization_window: Optional[QuantumCircuitWindow] = None    
):
    if invert:
        gates.reverse()

    for gate_info in gates:
        gate_info.gate_operation()
        if visualization_window is not None and gate_info.name is not None:
            print(f"Adding Visualization for {gate_info.name}")
            visualization_window.add_circuit_state(
                quantum_circuit, 
                DisplayProperties(plot_name=gate_info.name),
                fast_state=True,
            )
            
    if visualization_window is None:
        custom_gate = quantum_circuit.to_gate()
        custom_gate.label = f"{name}{'†' if invert else ''}"
        return custom_gate

def carry_gate(invert: bool = False, visualization_window: Optional[QuantumCircuitWindow] = None):
    carry_registers = CarryRegisters(
        input_carry_qubit = qiskit.QuantumRegister(1, name="Input Carry"),
        left_operand_qubit = qiskit.QuantumRegister(1, name="Left Operand"),
        right_operand_qubit = qiskit.QuantumRegister(1, name="Right Operand"),
        output_carry_qubit = qiskit.QuantumRegister(1, name="Output Carry")
    )
    carry_circuit = qiskit.QuantumCircuit(
        carry_registers.input_carry_qubit, carry_registers.left_operand_qubit, carry_registers.right_operand_qubit, carry_registers.output_carry_qubit, name="Carry Gate"
    )
    if visualization_window is not None:
        carry_circuit.x(carry_registers.input_carry_qubit)
        carry_circuit.x(carry_registers.left_operand_qubit)
        carry_circuit.x(carry_registers.right_operand_qubit)
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
        GateInfo(partial(carry_circuit.ccx, carry_registers.left_operand_qubit, carry_registers.right_operand_qubit, carry_registers.output_carry_qubit), "Carry if R and L are 1"),
        GateInfo(partial(carry_circuit.cx, carry_registers.left_operand_qubit, carry_registers.right_operand_qubit), "Disarm Right Operand if Both are 1"), 
        GateInfo(partial(carry_circuit.ccx, carry_registers.input_carry_qubit, carry_registers.right_operand_qubit, carry_registers.output_carry_qubit), "Carry if R and C input are 1")
    ]

    return convert_to_gates(carry_circuit, invert, gates, "Carry", visualization_window)
        
def sum_gate(invert: bool = False,  visualization_window: Optional[QuantumCircuitWindow] = None):
    sum_registers = SumRegisters(
        left_operand_qubit = qiskit.QuantumRegister(1, name="Left Operand"),
        right_operand_qubit = qiskit.QuantumRegister(1, name="Right Operand"),
        output_qubit = qiskit.QuantumRegister(1, name="Output")
    )
    sum_circuit = qiskit.QuantumCircuit(sum_registers.left_operand_qubit, sum_registers.right_operand_qubit, sum_registers.output_qubit, name="Sum Gate")
    # this is for checking what the actual output of the addition is
    # because the carry gates only handles the carry bits
    # l + r
    # 0 + 0 = 0
    # 0 + 1 = 1
    # 1 + 0 = 1
    # 1 + 1 = 0
    if visualization_window is not None:
        sum_circuit.x(sum_registers.left_operand_qubit)
        sum_circuit.x(sum_registers.right_operand_qubit)
        visualization_window.initialize_circuit_properties(sum_circuit, 33, DisplayProperties(plot_name="Initialized"))
        
    gates = [
        GateInfo(partial(sum_circuit.cx, sum_registers.left_operand_qubit, sum_registers.output_qubit), "Check L Value"), 
        GateInfo(partial(sum_circuit.cx, sum_registers.right_operand_qubit, sum_registers.output_qubit), "Check R Value"), 
    ]
    return convert_to_gates(sum_circuit, invert, gates, "Sum", visualization_window)
    
def adder_gate(invert: bool = False, visualization_window: Optional[QuantumCircuitWindow] = None):
    adder_registers = AdderRegisters(
        left_operand_qubits = qiskit.QuantumRegister(_LEFT_OPERAND_REGISTER, name="Left Operand Qubits"),
        right_operand_qubits = qiskit.QuantumRegister(_RIGHT_OPERAND_REGISTER, name="Right Operand Qubits"),
        carry_qubits = qiskit.QuantumRegister(_CARRY_REGISTER, name="Carry Qubits")
    )
    adder_circuit = qiskit.QuantumCircuit(adder_registers.left_operand_qubits, adder_registers.right_operand_qubits, adder_registers.carry_qubits, name="Adder Gate")
    if visualization_window is not None:
        flip_bits(adder_circuit, adder_registers.left_operand_qubits, int_to_binary_list(5, length=_FUNDAMENTAL_BASIS))
        flip_bits(adder_circuit, adder_registers.right_operand_qubits, int_to_binary_list(3, length=_FUNDAMENTAL_BASIS))
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
    # this cx gate is applied to preserve the last carry bit
    gates.append(GateInfo(partial(adder_circuit.cx, adder_registers.left_operand_qubits[_FUNDAMENTAL_BASIS-1], adder_registers.right_operand_qubits[_FUNDAMENTAL_BASIS-1]), "Preserve Last Carry"))
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
    return convert_to_gates(adder_circuit, invert, gates, "Adder", visualization_window)

def modulated_adder_gate(classical_modulated_value: int, invert: bool=False, visualization_window: Optional[QuantumCircuitWindow] = None):
    modulated_adder_registers = ModulatedAdderRegisters(
        left_operand_qubits = qiskit.QuantumRegister(_LEFT_OPERAND_REGISTER, name="Left Operand Qubits"),
        right_operand_qubits = qiskit.QuantumRegister(_RIGHT_OPERAND_REGISTER, name="Right Operand Qubits"),
        carry_qubits = qiskit.QuantumRegister(_CARRY_REGISTER, name="Carry Qubits"),
        temporary_qubits = qiskit.QuantumRegister(_TEMPORARY_REGISTER, name="Modulation Value Qubits"),
        control_qubit = qiskit.QuantumRegister(_CONTROL_QUBIT, name="Control Qubit"), 
        
    )
    mod_adder_circuit = qiskit.QuantumCircuit(
        modulated_adder_registers.left_operand_qubits, 
        modulated_adder_registers.right_operand_qubits, 
        modulated_adder_registers.carry_qubits, 
        modulated_adder_registers.temporary_qubits,
        modulated_adder_registers.control_qubit,
        name="Modulated Adder Gate"
    )
    if visualization_window is not None:
        flip_bits(mod_adder_circuit, modulated_adder_registers.left_operand_qubits, int_to_binary_list(3, length=_FUNDAMENTAL_BASIS))
        flip_bits(mod_adder_circuit, modulated_adder_registers.right_operand_qubits, int_to_binary_list(3, length=_FUNDAMENTAL_BASIS))
        flip_bits(mod_adder_circuit, modulated_adder_registers.temporary_qubits, int_to_binary_list(5, length=_FUNDAMENTAL_BASIS))
        visualization_window.initialize_circuit_properties(mod_adder_circuit, 33, DisplayProperties(plot_name="Initialized"))
    # the last two adders are just to uncompute the control qubit
    
    # a+b
    gates: List[GateInfo] = [
        GateInfo(partial(mod_adder_circuit.append, adder_gate(invert), list(modulated_adder_registers)), "Add Left and Right")
    ]
    # a is now N, and N is now a
    swap_gates: List[GateInfo] = []
    operation_name = None
    for location in range(_FUNDAMENTAL_BASIS):
        if location == _FUNDAMENTAL_BASIS - 1:
            operation_name = "Swap Left for N"
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
            GateInfo(partial(mod_adder_circuit.append, adder_gate(not invert), list(modulated_adder_registers)), "Subtract N from L + R"), 
            GateInfo(partial(mod_adder_circuit.x, modulated_adder_registers.right_operand_qubits[_FUNDAMENTAL_BASIS]), None),
            GateInfo(partial(mod_adder_circuit.cx, modulated_adder_registers.right_operand_qubits[_FUNDAMENTAL_BASIS], modulated_adder_registers.control_qubit[0]), None),
            GateInfo(partial(mod_adder_circuit.x, modulated_adder_registers.right_operand_qubits[_FUNDAMENTAL_BASIS]), "Flip Control for Zeroing"),
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
        GateInfo(partial(mod_adder_circuit.append, adder_gate(invert), list(modulated_adder_registers)), "Add N to L + R - N if Not Zeroed")
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
    
    return convert_to_gates(mod_adder_circuit, invert, gates, "ModAdder", visualization_window)

def controlled_mod_mult_gate(classical_modulated_value: int, modulated_coprime: int, invert: bool=False, visualization_window: Optional[QuantumCircuitWindow] = None):
    cntrl_mod_mult_registers = ControlModMultRegisters(
        left_operand_qubits = qiskit.QuantumRegister(_LEFT_OPERAND_REGISTER, name="Left Operand Qubits"),
        right_operand_qubits = qiskit.QuantumRegister(_RIGHT_OPERAND_REGISTER, name="Right Operand Qubits"),
        carry_qubits = qiskit.QuantumRegister(_CARRY_REGISTER, name="Carry Qubits"),
        temporary_qubits = qiskit.QuantumRegister(_TEMPORARY_REGISTER, name="Modulation Value Qubits"),
        control_qubit = qiskit.QuantumRegister(_CONTROL_QUBIT, name="Control Qubit"), 
        multiplier_qubits= qiskit.QuantumRegister(_MULTIPLIER_REGISTER, name="Multiplier Qubits"),
        application_qubit= qiskit.QuantumRegister(1, name="Counting Qubit"),
    )
    cntrl_mod_mult_circuit = qiskit.QuantumCircuit(
        cntrl_mod_mult_registers.left_operand_qubits, 
        cntrl_mod_mult_registers.right_operand_qubits, 
        cntrl_mod_mult_registers.carry_qubits, 
        cntrl_mod_mult_registers.temporary_qubits,
        cntrl_mod_mult_registers.control_qubit,
        cntrl_mod_mult_registers.multiplier_qubits,
        cntrl_mod_mult_registers.application_qubit,
        name="CMMG"
    )
    
    if visualization_window is not None:
        flip_bits(cntrl_mod_mult_circuit, cntrl_mod_mult_registers.application_qubit, [1])
        flip_bits(cntrl_mod_mult_circuit, cntrl_mod_mult_registers.multiplier_qubits, int_to_binary_list(2, length=_FUNDAMENTAL_BASIS))
        flip_bits(cntrl_mod_mult_circuit, cntrl_mod_mult_registers.temporary_qubits, int_to_binary_list(classical_modulated_value, length=_FUNDAMENTAL_BASIS))
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
    for bit_shift_amount in range(_FUNDAMENTAL_BASIS):
        bit_segment_of_both_components = modulated_coprime * (2 ** bit_shift_amount) % classical_modulated_value
        control_gates = []
        for location, bit in enumerate(reversed(format(bit_segment_of_both_components, f"0{_FUNDAMENTAL_BASIS}b"))):
            if bit == "1":
                gates.append(GateInfo(partial(
                    cntrl_mod_mult_circuit.ccx, 
                    cntrl_mod_mult_registers.application_qubit[0], 
                    cntrl_mod_mult_registers.multiplier_qubits[bit_shift_amount], 
                    cntrl_mod_mult_registers.left_operand_qubits[location]
                    ), None)
                )
                control_gates.append(gates[-1])
        gates[-1].name = "Control Bit Location Multiplier"
        
        gates.append(GateInfo(partial(cntrl_mod_mult_circuit.append, modulated_adder_gate(classical_modulated_value, invert), list(cntrl_mod_mult_registers)), f"Modulated Addition for bit {bit_shift_amount}"))

        control_gates.reverse()
        gates.extend(control_gates)
        gates[-1].name = "Undo Control Bit Location Multiplier"

    # in the case in which the control bit is zero, flip it, and set the b register to z so it is maintained
    # within the circuit
    gates.append(GateInfo(partial(cntrl_mod_mult_circuit.x, cntrl_mod_mult_registers.application_qubit[0]), None))
    for location in range(_FUNDAMENTAL_BASIS):
        gates.append(GateInfo(partial(cntrl_mod_mult_circuit.ccx, cntrl_mod_mult_registers.application_qubit[0], cntrl_mod_mult_registers.multiplier_qubits[location], cntrl_mod_mult_registers.right_operand_qubits[location]), None))
    gates.append(GateInfo(partial(cntrl_mod_mult_circuit.x, cntrl_mod_mult_registers.application_qubit[0]), "Push Z into B"))

    return convert_to_gates(cntrl_mod_mult_circuit, invert, gates, "CntrlModMult", visualization_window)

def modular_exponentiation(classical_modulated_value: int, classical_base_value: int, visualization_window: Optional[QuantumCircuitWindow] = None):
    mod_exp_registers = ModularExponentiationRegisters(
        left_operand_qubits = qiskit.QuantumRegister(_LEFT_OPERAND_REGISTER, name="Left Operand Qubits"),
        right_operand_qubits = qiskit.QuantumRegister(_RIGHT_OPERAND_REGISTER, name="Right Operand Qubits"),
        carry_qubits = qiskit.QuantumRegister(_CARRY_REGISTER, name="Carry Qubits"),
        temporary_qubits = qiskit.QuantumRegister(_TEMPORARY_REGISTER, name="Modulation Value Qubits"),
        control_qubit = qiskit.QuantumRegister(_CONTROL_QUBIT, name="Control Qubit"), 
        multiplier_qubits= qiskit.QuantumRegister(_MULTIPLIER_REGISTER, name="Multiplier Qubits"),
        counting_qubits= qiskit.QuantumRegister(_COUNTING_QUBITS, name="Counting Qubits"),
    )
    mod_exp_circuit = qiskit.QuantumCircuit(
        mod_exp_registers.left_operand_qubits, 
        mod_exp_registers.right_operand_qubits, 
        mod_exp_registers.carry_qubits, 
        mod_exp_registers.temporary_qubits,
        mod_exp_registers.control_qubit,
        mod_exp_registers.multiplier_qubits,
        mod_exp_registers.counting_qubits,
        name="Modular Exponentiation"
    )
    
    if visualization_window is not None:
        flip_bits(mod_exp_circuit, mod_exp_registers.counting_qubits, int_to_binary_list(7, length=_FUNDAMENTAL_BASIS*2))
        flip_bits(mod_exp_circuit, mod_exp_registers.multiplier_qubits, int_to_binary_list(1, length=_FUNDAMENTAL_BASIS))
        flip_bits(mod_exp_circuit, mod_exp_registers.temporary_qubits, int_to_binary_list(classical_modulated_value, length=_FUNDAMENTAL_BASIS))
        visualization_window.initialize_circuit_properties(mod_exp_circuit, 33, DisplayProperties(plot_name="Initialized"))
    
    gates: List[GateInfo] = []
    # x is the control for the exponential value
    # first, the multiplication mod gate is used on Z, with m being a and Z being 00...01, meaning we get a Mod N as our b register 
    # this is then swapped with Z, so Z holds a mod N and b holds 1, thus when the inverse is applied, it reverse the global state
    # (besides the control gate) back to it's original form.
    # now, however, there is an entangled interference kickback to the control register, which reflects the periodicity of x + r
    # for all possible x's

    for counting_qubit_number in range(_COUNTING_QUBITS):
        exponential_component = classical_base_value ** (2 ** counting_qubit_number)
        gates.append(GateInfo(partial(
            mod_exp_circuit.append, 
            controlled_mod_mult_gate(classical_modulated_value, exponential_component), 
            mod_exp_registers[counting_qubit_number]
        ), f"Multiply Z with y^2^k For CB {counting_qubit_number}"))
        # switch output back with Z register
        for location in range(_FUNDAMENTAL_BASIS):

            gates.append(GateInfo(partial(
                mod_exp_circuit.cswap, 
                mod_exp_registers.counting_qubits[counting_qubit_number], 
                mod_exp_registers.multiplier_qubits[location], 
                mod_exp_registers.right_operand_qubits[location]
            ), None))
        gates[-1].name =  f"Swap Adder Output With Z For CB {counting_qubit_number}"
        # revert back to Z 
        gates.append(GateInfo(partial(
            mod_exp_circuit.append, 
            controlled_mod_mult_gate(classical_modulated_value, pow(exponential_component, -1, classical_modulated_value), invert=True), 
            mod_exp_registers[counting_qubit_number]
        ), f"Remove y^2^k For CB {counting_qubit_number}"))
            
    return convert_to_gates(mod_exp_circuit, False, gates, "CntrlModMult", visualization_window)

def qpe_gate(number_of_qubits: int, invert: bool = False, visualization_window: Optional[QuantumCircuitWindow] = None):
    qpe_register= qiskit.QuantumRegister(number_of_qubits, name="QPE Qubit Register")
    quantum_phase_estimation_circuit = qiskit.QuantumCircuit(qpe_register, name="Quantum Phase Estimation")  
    gates: List[GateInfo] = []
    if visualization_window is not None:
        flip_bits(quantum_phase_estimation_circuit, [0, 3, 5], int_to_binary_list(7, length=_FUNDAMENTAL_BASIS*2))
        visualization_window.initialize_circuit_properties(quantum_phase_estimation_circuit, 33, DisplayProperties(plot_name="Initialized"))
    
    
    # reverse order, as qubit 0 is the LSB, and the paper stat "reverse the order of the qubits with a string of SWAP" to fix this
    for qubit_number in range(number_of_qubits // 2):
        # reverse the order of the qubits 
        gates.append(GateInfo(partial(quantum_phase_estimation_circuit.swap, qubit_number, number_of_qubits - qubit_number - 1), None))
    
    gates[-1].name = "Reverse Qubit Ordering"
    for controlling_qubit_number in reversed(range(number_of_qubits)):
        # provides the superposition basis
        gates.append(GateInfo(partial(quantum_phase_estimation_circuit.h, controlling_qubit_number), f"Place Qubit {controlling_qubit_number} Into Superposition"))
        for qubit_number in reversed(range(controlling_qubit_number)):
            # phase angle decreases with qubit order
            phase_angle = np.pi / (2 ** (controlling_qubit_number - qubit_number))
            # invert angle if inverse QPE
            # rotates the counting qubit 
            gates.append(GateInfo(partial(quantum_phase_estimation_circuit.cp, phase_angle * -1 if invert else phase_angle, controlling_qubit_number, qubit_number), None))
        gates[-1].name = f"Decompose Qubit {controlling_qubit_number}"
            
    return convert_to_gates(quantum_phase_estimation_circuit, invert, gates, "QPE", visualization_window)

def visualize_shors():
    bloch_sphere_window = QuantumCircuitWindow(ignore_entanglement=False)

    qubit_registers = QubitRegisters(
        left_operand_qubits = qiskit.QuantumRegister(_LEFT_OPERAND_REGISTER, name="Left Operand Qubits"),
        right_operand_qubits = qiskit.QuantumRegister(_RIGHT_OPERAND_REGISTER, name="Right Operand Qubits"),
        carry_qubits = qiskit.QuantumRegister(_CARRY_REGISTER, name="Carry Qubits"),
        temporary_qubits = qiskit.QuantumRegister(_TEMPORARY_REGISTER, name="Modulation Value Qubits"),
        control_qubit = qiskit.QuantumRegister(_CONTROL_QUBIT, name="Control Qubit"), 
        multiplier_qubits= qiskit.QuantumRegister(_MULTIPLIER_REGISTER, name="Multiplier Qubits"),
        counting_qubits= qiskit.QuantumRegister(_COUNTING_QUBITS, name="Counting Qubits"),
    )
    classical_output = qiskit.ClassicalRegister(_COUNTING_QUBITS, "Classical Output")
    quantum_circuit = qiskit.QuantumCircuit(
        qubit_registers.left_operand_qubits, 
        qubit_registers.right_operand_qubits, 
        qubit_registers.carry_qubits, 
        qubit_registers.temporary_qubits,
        qubit_registers.control_qubit,
        qubit_registers.multiplier_qubits,
        qubit_registers.counting_qubits,
        classical_output,
        name="Shor's Algorithm"
    )
    prime_1 = 2
    prime_2 = 2
    coprime = 3
    product_of_primes = prime_1 * prime_2

    bloch_sphere_window.initialize_circuit_properties(quantum_circuit, 33, DisplayProperties(plot_name="Initialized"))
    print("Running Shors Algorithm")
    quantum_circuit.h(qubit_registers.counting_qubits)
    quantum_circuit.x(qubit_registers.multiplier_qubits[0])
    # flip N register so that N can be used in the modulated adder
    for location, bit in enumerate(format(product_of_primes, f"0{_FUNDAMENTAL_BASIS}b")):
        if bit == "1":
            quantum_circuit.x(qubit_registers.temporary_qubits[location])
    # pass in coprime a so it can be pre-calculated during the multiplcation process
    quantum_circuit.append(modular_exponentiation(product_of_primes, coprime), list(qubit_registers))
    bloch_sphere_window.add_circuit_state(
        quantum_circuit, 
        DisplayProperties(plot_name="Modular Exponentiation"),
        fast_state=True,
    )
    quantum_circuit.append(qpe_gate(_COUNTING_QUBITS,invert=True), qubit_registers.counting_qubits)
    bloch_sphere_window.add_circuit_state(
        quantum_circuit, 
        DisplayProperties(plot_name="Inverse QPE"),
        fast_state=True,
    )
    bloch_sphere_window.animate_circuit()
    bloch_sphere_window.app.exec()


def entire_shors_algorithm():

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

    print("Running Shors Algorithm")
    quantum_circuit.h(qubit_registers.counting_qubits)
    quantum_circuit.x(qubit_registers.multiplier_qubits[0])
    # flip N register so that N can be used in the modulated adder
    for location, bit in enumerate(format(product_of_primes, f"0{_FUNDAMENTAL_BASIS}b")):
        if bit == "1":
            quantum_circuit.x(qubit_registers.temporary_qubits[location])
    # pass in coprime a so it can be pre-calculated during the multiplcation process
    quantum_circuit.append(modular_exponentiation(product_of_primes, coprime), list(qubit_registers))
    quantum_circuit.append(qpe_gate(_COUNTING_QUBITS,invert=True), qubit_registers.counting_qubits)
    quantum_circuit.measure(qubit_registers.counting_qubits, classical_output)

    sim = AerSimulator(method='matrix_product_state', matrix_product_state_max_bond_dimension=4096, seed_simulator=42)
    compiled_circuit = qiskit.transpile(quantum_circuit, sim)
    counts = {}
    job = sim.run(compiled_circuit, shots=1024)
    previous_status = None
    while not job.done():
        status = job.status()
        if status != previous_status:
            print("Waiting... job status:", status, flush=True)
            previous_status = status
        time.sleep(1)
    result = job.result()
    counts |= result.get_counts(quantum_circuit)
    print("Finished Shors Algorithm")
    print(counts)
    found_factor = False
    attempt = 0
    found_factors = [None, None]
    for measure_key, measure_value in sorted(counts.items(), key=lambda item: item[1], reverse=True)[:7]:
        measured_int = int(measure_key, 2)
        if measured_int == 0 or measured_int == 2**_COUNTING_QUBITS - 1:
            continue
        
        phase = measured_int / 2**_COUNTING_QUBITS
        frac = Fraction(phase).limit_denominator(product_of_primes)
        r = frac.denominator
        if r % 2 != 0 or r <= 1:
            continue  # skip odd periods
        attempt += 1
        if pow(coprime, r, product_of_primes) == 1:
            x = pow(coprime, r // 2, product_of_primes)
            factors = [math.gcd(x - 1, product_of_primes), math.gcd(x + 1, product_of_primes)]
            
            if 1 < factors[0] < product_of_primes:
                print(f"Factor found: {factors[0]}")
                found_factors[0] = factors[0]
                found_factor = True
            if 1 < factors[1] < product_of_primes:
                print(f"Factor found: {factors[1]}")
                found_factors[1] = factors[1]
                found_factor = True
        if found_factor:
            print(f"Found factor on proper phase attempt {attempt} for value {measure_key}")
            break
    sorted_plot_histogram(counts, found_factors[0], found_factors[1], attempt, measure_key)
    plt.show()

def sorted_plot_histogram(
    counts: Dict[str, int], 
    factor_1: Optional[int] = None, 
    factor_2: Optional[int] = None, 
    attempt: Optional[int] = None, 
    measure_key: Optional[int] = None,
):
    sorted_counts = dict(sorted(counts.items(), key=lambda item: int(item[0], 2)))
    plot_histogram(
        sorted_counts,
        title="Shor's Algorithm Probability Distribution",
        figsize=(10,5),
        color="midnightblue",
        bar_labels=False  # set to True if you want value labels on bars
    )

    if factor_1 is None and factor_2 is None:
        factor_string = "Found No Factors"
    else:
        factor_string = "Found: "
        if factor_1:
            factor_string += f"Factor 1 ({factor_1}), "
        if factor_2:
            factor_string += f"Factor 2 ({factor_2}), "
        factor_string += f"on Valid Phase Attempt {attempt} for Value {measure_key}"
            
    plt.suptitle(factor_string, fontsize=10, y=1.0)
