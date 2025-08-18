import qiskit
import shors_algorithm
import itertools
from qiskit_aer import AerSimulator
from typing import Dict, List
import random
import math

def test_shors():
    test_adder()
    test_mod_adder()
    test_controlled_mod_mult()
    test_mod_exp()
    
def verify_value(
    quantum_circuit: qiskit.QuantumCircuit, 
    measurement_results: Dict[str, int], 
    expected_value: int, 
    classical_register: qiskit.ClassicalRegister,
    strict: bool = True
):
    classical_bits = quantum_circuit.clbits
    
    bit_indices = [classical_bits.index(bit) for bit in classical_register]
    for bitstring in measurement_results:
        truncated_bitstring = bitstring.replace(" ", "")
        if bit_indices[0] == 0:
            bit_value = truncated_bitstring[-bit_indices[-1]-1:]
        else:
            bit_value = truncated_bitstring[-bit_indices[-1]-1:-bit_indices[0]]
        value = 0
        for location, bit in enumerate(reversed(bit_value)):
            if bit == "1":
                value += 2 ** location
                
        if strict:
            if value != expected_value:
                print(f"{classical_register.name} gotten {value} != expected {expected_value}")
                # raise ValueError("Not matching elements")
            return bit_value

def convert_to_int(register_values: List[int]) -> int:
    classical_value = 0
    for location, bit in enumerate(reversed(register_values)):
        if bit == 1:
            classical_value += 2 ** location                
    return classical_value  


def test_adder():
    left_operand_comb = list(itertools.product([0, 1], repeat=shors_algorithm._LEFT_OPERAND_REGISTER))
    right_operand_comb = list(itertools.product([0, 1], repeat=shors_algorithm._RIGHT_OPERAND_REGISTER-1))

    for left_reg in left_operand_comb:
        for right_reg in right_operand_comb:
            adder_registers = shors_algorithm.AdderRegisters(
                left_operand_qubits = qiskit.QuantumRegister(shors_algorithm._LEFT_OPERAND_REGISTER, name="left_operand_qubits"),
                right_operand_qubits = qiskit.QuantumRegister(shors_algorithm._RIGHT_OPERAND_REGISTER, name="right_operand_qubits"),
                carry_qubits = qiskit.QuantumRegister(shors_algorithm._CARRY_REGISTER, name="carry_qubits")
            )
            left_operand_output = qiskit.ClassicalRegister(shors_algorithm._LEFT_OPERAND_REGISTER, "left_operand_bits")
            right_operand_output = qiskit.ClassicalRegister(shors_algorithm._RIGHT_OPERAND_REGISTER, "right_operand_bits")
            carry_output = qiskit.ClassicalRegister(shors_algorithm._CARRY_REGISTER, "carry_bits")
            adder_circuit = qiskit.QuantumCircuit(
                adder_registers.left_operand_qubits, 
                adder_registers.right_operand_qubits, 
                adder_registers.carry_qubits,
                left_operand_output,
                right_operand_output,
                carry_output,
                name="adder_gate"
            )

            right_register_classical = shors_algorithm.flip_bits(adder_circuit, adder_registers.right_operand_qubits, right_reg)
            left_register_classical = shors_algorithm.flip_bits(adder_circuit, adder_registers.left_operand_qubits, left_reg)

            expected_right_register = left_register_classical + right_register_classical
            expected_left_register = left_register_classical
            expected_carry_register = 0
                
            adder_circuit.append(shors_algorithm.adder_gate(), list(adder_registers.left_operand_qubits) + list(adder_registers.right_operand_qubits) + list(adder_registers.carry_qubits))
            adder_circuit.measure(adder_registers.carry_qubits, carry_output)
            adder_circuit.measure(adder_registers.right_operand_qubits, right_operand_output)
            adder_circuit.measure(adder_registers.left_operand_qubits, left_operand_output)
            
            sim = AerSimulator(method='matrix_product_state', matrix_product_state_max_bond_dimension=4096, seed_simulator=42)
            compiled_circuit = qiskit.transpile(adder_circuit, sim)
            job = sim.run(compiled_circuit, shots=1)
            result = job.result()
            counts = result.get_counts(adder_circuit)
            print(counts)
            verify_value(adder_circuit, counts, expected_left_register, left_operand_output)
            output_value = verify_value(adder_circuit, counts, expected_right_register, right_operand_output)
            verify_value(adder_circuit, counts, expected_carry_register, carry_output)
            adder_circuit.reset(range(len(adder_circuit.qubits)))
            adder_circuit.data.clear()
            
            sub_right_reg = [int(bit) for bit in output_value]
            sub_left_reg = [0,1,1,1]
            right_register_classical = shors_algorithm.flip_bits(adder_circuit, adder_registers.right_operand_qubits, sub_right_reg)
            left_register_classical = shors_algorithm.flip_bits(adder_circuit, adder_registers.left_operand_qubits, sub_left_reg)
            print(f"right {right_register_classical}, left: {left_register_classical}")

            expected_right_register = (right_register_classical - left_register_classical) % (2 ** shors_algorithm._RIGHT_OPERAND_REGISTER)
            expected_left_register = left_register_classical
            expected_carry_register = 0
 
            adder_circuit.append(shors_algorithm.adder_gate(invert=True), list(adder_registers.left_operand_qubits) + list(adder_registers.right_operand_qubits) + list(adder_registers.carry_qubits))
            adder_circuit.measure(adder_registers.carry_qubits, carry_output)
            adder_circuit.measure(adder_registers.right_operand_qubits, right_operand_output)
            adder_circuit.measure(adder_registers.left_operand_qubits, left_operand_output)
 
            sim = AerSimulator(method='matrix_product_state', matrix_product_state_max_bond_dimension=4096, seed_simulator=42)
            compiled_circuit = qiskit.transpile(adder_circuit, sim)
            job = sim.run(compiled_circuit, shots=1)
            result = job.result()
            counts = result.get_counts(adder_circuit)
            print(counts)
            print()
            verify_value(adder_circuit, counts, expected_left_register, left_operand_output)
            verify_value(adder_circuit, counts, expected_right_register, right_operand_output)
            verify_value(adder_circuit, counts, expected_carry_register, carry_output)
            adder_circuit.reset(range(len(adder_circuit.qubits)))
            adder_circuit.data.clear()
                       
 
def test_mod_adder():
    modulator_operand = list(itertools.product([0, 1], repeat=shors_algorithm._TEMPORARY_REGISTER))
    modulator_operand = modulator_operand[2 ** 2:]
    left_operand_comb = list(itertools.product([0, 1], repeat=shors_algorithm._LEFT_OPERAND_REGISTER))
    right_operand_comb = list(itertools.product([0, 1], repeat=shors_algorithm._RIGHT_OPERAND_REGISTER-1))
             
    for mod_reg in modulator_operand:
        modulated_adder_registers = shors_algorithm.ModulatedAdderRegisters(
            left_operand_qubits = qiskit.QuantumRegister(shors_algorithm._LEFT_OPERAND_REGISTER, name="left_operand_qubits"),
            right_operand_qubits = qiskit.QuantumRegister(shors_algorithm._RIGHT_OPERAND_REGISTER, name="right_operand_qubits"),
            carry_qubits = qiskit.QuantumRegister(shors_algorithm._CARRY_REGISTER, name="carry_qubits"),
            temporary_qubits = qiskit.QuantumRegister(shors_algorithm._TEMPORARY_REGISTER, name="remainder_qubits"),
            control_qubit = qiskit.QuantumRegister(shors_algorithm._CONTROL_QUBIT, name="control_qubit"), 
        )
        left_operand_output = qiskit.ClassicalRegister(shors_algorithm._LEFT_OPERAND_REGISTER, "left_operand_bits")
        right_operand_output = qiskit.ClassicalRegister(shors_algorithm._RIGHT_OPERAND_REGISTER, "right_operand_bits")
        carry_output = qiskit.ClassicalRegister(shors_algorithm._CARRY_REGISTER, "carry_bits")
        temporary_output = qiskit.ClassicalRegister(shors_algorithm._TEMPORARY_REGISTER, "temporary_bits")
        control_output = qiskit.ClassicalRegister(shors_algorithm._CONTROL_QUBIT, "control_bit")
        mod_adder_circuit = qiskit.QuantumCircuit(
            modulated_adder_registers.left_operand_qubits, 
            modulated_adder_registers.right_operand_qubits, 
            modulated_adder_registers.carry_qubits, 
            modulated_adder_registers.temporary_qubits,
            modulated_adder_registers.control_qubit,
            left_operand_output,
            right_operand_output,
            carry_output,
            temporary_output,
            control_output,
            name="modulated_adder_gate"
        )
        
        mod_register_classical = shors_algorithm.flip_bits(mod_adder_circuit, modulated_adder_registers.temporary_qubits, mod_reg)
        right_reg = random.choice([right_op for right_op in right_operand_comb if convert_to_int(right_op) < mod_register_classical])
        left_reg = random.choice([left_op for left_op in left_operand_comb if convert_to_int(left_op) < mod_register_classical])
        print(f"Chose: L {left_reg}, R: {right_reg}")
        right_register_classical = shors_algorithm.flip_bits(mod_adder_circuit, modulated_adder_registers.right_operand_qubits, right_reg)
        left_register_classical = shors_algorithm.flip_bits(mod_adder_circuit, modulated_adder_registers.left_operand_qubits, left_reg)

        expected_right_register = (left_register_classical + right_register_classical) % mod_register_classical
        expected_left_register = left_register_classical
        expected_carry_register = 0
        expected_mod_register = mod_register_classical 
        expected_control_register = 0
            
        mod_adder_circuit.append(shors_algorithm.modulated_adder_gate(mod_register_classical), list(modulated_adder_registers.left_operand_qubits) + list(modulated_adder_registers.right_operand_qubits) + list(modulated_adder_registers.carry_qubits) + list(modulated_adder_registers.temporary_qubits) + list(modulated_adder_registers.control_qubit))
        mod_adder_circuit.measure(modulated_adder_registers.control_qubit, control_output)
        mod_adder_circuit.measure(modulated_adder_registers.temporary_qubits, temporary_output)
        mod_adder_circuit.measure(modulated_adder_registers.carry_qubits, carry_output)
        mod_adder_circuit.measure(modulated_adder_registers.right_operand_qubits, right_operand_output)
        mod_adder_circuit.measure(modulated_adder_registers.left_operand_qubits, left_operand_output)
        
        sim = AerSimulator(method='matrix_product_state', matrix_product_state_max_bond_dimension=4096, seed_simulator=42)
        compiled_circuit = qiskit.transpile(mod_adder_circuit, sim)
        job = sim.run(compiled_circuit, shots=1)
        result = job.result()
        counts = result.get_counts(mod_adder_circuit)
        print(counts)
        verify_value(mod_adder_circuit, counts, expected_left_register, left_operand_output)
        verify_value(mod_adder_circuit, counts, expected_right_register, right_operand_output)
        verify_value(mod_adder_circuit, counts, expected_carry_register, carry_output)
        verify_value(mod_adder_circuit, counts, expected_mod_register, temporary_output)
        verify_value(mod_adder_circuit, counts, expected_control_register, control_output)


def inner_loop_mod_mult(mod_reg, application_reg, mult_reg):
    cntrl_mod_mult_registers = shors_algorithm.ControlModMultRegisters(
        left_operand_qubits = qiskit.QuantumRegister(shors_algorithm._LEFT_OPERAND_REGISTER, name="left_operand_qubits"),
        right_operand_qubits = qiskit.QuantumRegister(shors_algorithm._RIGHT_OPERAND_REGISTER, name="right_operand_qubits"),
        carry_qubits = qiskit.QuantumRegister(shors_algorithm._CARRY_REGISTER, name="carry_qubits"),
        temporary_qubits = qiskit.QuantumRegister(shors_algorithm._TEMPORARY_REGISTER, name="remainder_qubits"),
        control_qubit = qiskit.QuantumRegister(shors_algorithm._CONTROL_QUBIT, name="control_qubit"), 
        multiplier_qubits= qiskit.QuantumRegister(shors_algorithm._MULTIPLIER_REGISTER, name="multiplier_qubits"),
        application_qubit= qiskit.QuantumRegister(1, name="application_qubit"),
    )

    left_operand_output = qiskit.ClassicalRegister(shors_algorithm._LEFT_OPERAND_REGISTER, "left_operand_bits")
    right_operand_output = qiskit.ClassicalRegister(shors_algorithm._RIGHT_OPERAND_REGISTER, "right_operand_bits")
    carry_output = qiskit.ClassicalRegister(shors_algorithm._CARRY_REGISTER, "carry_bits")
    temporary_output = qiskit.ClassicalRegister(shors_algorithm._TEMPORARY_REGISTER, "temporary_bits")
    control_output = qiskit.ClassicalRegister(shors_algorithm._CONTROL_QUBIT, "control_bit")
    multiplier_output = qiskit.ClassicalRegister(shors_algorithm._MULTIPLIER_REGISTER, "multiplier_bits")
    application_output = qiskit.ClassicalRegister(1, "application_bit")
    cntrl_mod_mult_circuit = qiskit.QuantumCircuit(
        cntrl_mod_mult_registers.left_operand_qubits, 
        cntrl_mod_mult_registers.right_operand_qubits, 
        cntrl_mod_mult_registers.carry_qubits, 
        cntrl_mod_mult_registers.temporary_qubits,
        cntrl_mod_mult_registers.control_qubit,
        cntrl_mod_mult_registers.multiplier_qubits,
        cntrl_mod_mult_registers.application_qubit,
        left_operand_output,
        right_operand_output,
        carry_output,
        temporary_output,
        control_output,
        multiplier_output,
        application_output,
        name="controlled_mod_mult_gate"
    )
    
    mod_register_classical = shors_algorithm.flip_bits(cntrl_mod_mult_circuit, cntrl_mod_mult_registers.temporary_qubits, mod_reg)
    mult_register_classical = shors_algorithm.flip_bits(cntrl_mod_mult_circuit, cntrl_mod_mult_registers.multiplier_qubits, mult_reg)
    appl_register_classical = shors_algorithm.flip_bits(cntrl_mod_mult_circuit, cntrl_mod_mult_registers.application_qubit, application_reg)

    coprimes = [coprime for coprime in range(1, mod_register_classical) if math.gcd(coprime, mod_register_classical) == 1]
    
    chosen_coprime = random.choice(coprimes)

    expected_right_register = mult_register_classical*chosen_coprime % mod_register_classical if appl_register_classical else mult_register_classical
    expected_left_register = 0
    expected_carry_register = 0
    expected_mod_register = mod_register_classical 
    expected_control_register = 0
    expected_mult_register = mult_register_classical
    expected_appl_register = appl_register_classical
        
    cntrl_mod_mult_circuit.append(shors_algorithm.controlled_mod_mult_gate(mod_register_classical, chosen_coprime), list(
        cntrl_mod_mult_registers.left_operand_qubits) 
        + list(cntrl_mod_mult_registers.right_operand_qubits) 
        + list(cntrl_mod_mult_registers.carry_qubits) 
        + list(cntrl_mod_mult_registers.temporary_qubits) 
        + list(cntrl_mod_mult_registers.control_qubit)
        + list(cntrl_mod_mult_registers.multiplier_qubits)
        + list(cntrl_mod_mult_registers.application_qubit)
    )
    cntrl_mod_mult_circuit.measure(cntrl_mod_mult_registers.application_qubit, application_output)
    cntrl_mod_mult_circuit.measure(cntrl_mod_mult_registers.multiplier_qubits, multiplier_output)
    cntrl_mod_mult_circuit.measure(cntrl_mod_mult_registers.control_qubit, control_output)
    cntrl_mod_mult_circuit.measure(cntrl_mod_mult_registers.temporary_qubits, temporary_output)
    cntrl_mod_mult_circuit.measure(cntrl_mod_mult_registers.carry_qubits, carry_output)
    cntrl_mod_mult_circuit.measure(cntrl_mod_mult_registers.right_operand_qubits, right_operand_output)
    cntrl_mod_mult_circuit.measure(cntrl_mod_mult_registers.left_operand_qubits, left_operand_output)
    
    sim = AerSimulator(method='matrix_product_state', matrix_product_state_max_bond_dimension=4096, seed_simulator=42)
    compiled_circuit = qiskit.transpile(cntrl_mod_mult_circuit, sim)
    job = sim.run(compiled_circuit, shots=1)
    result = job.result()
    counts = result.get_counts(cntrl_mod_mult_circuit)
    cntrl_mod_mult_circuit.reset(range(len(cntrl_mod_mult_circuit.qubits)))
    cntrl_mod_mult_circuit.data.clear()
    print(f"Forward: {counts}")
    verify_value(cntrl_mod_mult_circuit, counts, expected_left_register, left_operand_output)
    verify_value(cntrl_mod_mult_circuit, counts, expected_right_register, right_operand_output)
    verify_value(cntrl_mod_mult_circuit, counts, expected_carry_register, carry_output)
    verify_value(cntrl_mod_mult_circuit, counts, expected_mod_register, temporary_output)
    verify_value(cntrl_mod_mult_circuit, counts, expected_control_register, control_output)
    output_value = verify_value(cntrl_mod_mult_circuit, counts, expected_mult_register, multiplier_output)
    verify_value(cntrl_mod_mult_circuit, counts, expected_appl_register, application_output)


def test_controlled_mod_mult():
    modulator_operand = list(itertools.product([0, 1], repeat=shors_algorithm._TEMPORARY_REGISTER))
    multiplier_comb = list(itertools.product([0, 1], repeat=shors_algorithm._LEFT_OPERAND_REGISTER))
    modulator_operand = list(itertools.product([0, 1], repeat=shors_algorithm._TEMPORARY_REGISTER))
    modulator_operand = modulator_operand[2 ** 3:]


    for mult_reg in multiplier_comb:
        for application_reg in [[1], [0]]:
            for mod_reg in modulator_operand:
                inner_loop_mod_mult(mod_reg, application_reg, mult_reg)
                

def test_mod_exp():
    modulator_operand = list(itertools.product([0, 1], repeat=shors_algorithm._TEMPORARY_REGISTER))
    modulator_operand = modulator_operand[2 ** 3:]
    exp_comb = list(itertools.product([0, 1], repeat=shors_algorithm._TEMPORARY_REGISTER))

    for mod_reg in modulator_operand:
        for exp_reg in exp_comb:
            mod_exp_registers = shors_algorithm.ModularExponentiationRegisters(
                left_operand_qubits = qiskit.QuantumRegister(shors_algorithm._LEFT_OPERAND_REGISTER, name="left_operand_qubits"),
                right_operand_qubits = qiskit.QuantumRegister(shors_algorithm._RIGHT_OPERAND_REGISTER, name="right_operand_qubits"),
                carry_qubits = qiskit.QuantumRegister(shors_algorithm._CARRY_REGISTER, name="carry_qubits"),
                temporary_qubits = qiskit.QuantumRegister(shors_algorithm._TEMPORARY_REGISTER, name="remainder_qubits"),
                control_qubit = qiskit.QuantumRegister(shors_algorithm._CONTROL_QUBIT, name="control_qubit"), 
                multiplier_qubits= qiskit.QuantumRegister(shors_algorithm._MULTIPLIER_REGISTER, name="multiplier_qubits"),
                counting_qubits= qiskit.QuantumRegister(shors_algorithm._COUNTING_QUBITS, name="counting_qubits"),
            )

            left_operand_output = qiskit.ClassicalRegister(shors_algorithm._LEFT_OPERAND_REGISTER, "left_operand_bits")
            right_operand_output = qiskit.ClassicalRegister(shors_algorithm._RIGHT_OPERAND_REGISTER, "right_operand_bits")
            carry_output = qiskit.ClassicalRegister(shors_algorithm._CARRY_REGISTER, "carry_bits")
            temporary_output = qiskit.ClassicalRegister(shors_algorithm._TEMPORARY_REGISTER, "temporary_bits")
            control_output = qiskit.ClassicalRegister(shors_algorithm._CONTROL_QUBIT, "control_bit")
            multiplier_output = qiskit.ClassicalRegister(shors_algorithm._MULTIPLIER_REGISTER, "multiplier_bits")
            counting_output = qiskit.ClassicalRegister(shors_algorithm._COUNTING_QUBITS, "counting_bits")
            mod_exp_circuit = qiskit.QuantumCircuit(
                mod_exp_registers.left_operand_qubits, 
                mod_exp_registers.right_operand_qubits, 
                mod_exp_registers.carry_qubits, 
                mod_exp_registers.temporary_qubits,
                mod_exp_registers.control_qubit,
                mod_exp_registers.multiplier_qubits,
                mod_exp_registers.counting_qubits,
                left_operand_output,
                right_operand_output,
                carry_output,
                temporary_output,
                control_output,
                multiplier_output,
                counting_output,
                name="mod_exp_gate"
            )
            
            mod_register_classical = shors_algorithm.flip_bits(mod_exp_circuit, mod_exp_registers.temporary_qubits, [1,1,1,1])
            shors_algorithm.flip_bits(mod_exp_circuit, mod_exp_registers.multiplier_qubits, [0, 0, 1])
            exp_register_classical = shors_algorithm.flip_bits(mod_exp_circuit, mod_exp_registers.counting_qubits, exp_reg)

            coprimes = [coprime for coprime in range(1, mod_register_classical) if math.gcd(coprime, mod_register_classical) == 1]
            
            chosen_coprime = random.choice([coprime for coprime in coprimes if coprime != 1])

            expected_right_register = 0
            expected_left_register = 0
            expected_carry_register = 0
            expected_mod_register = mod_register_classical 
            expected_control_register = 0
            expected_mult_register = (chosen_coprime ** exp_register_classical) % mod_register_classical 
            expected_counting_register = exp_register_classical
                
            mod_exp_circuit.append(shors_algorithm.modular_exponentiation(mod_register_classical, chosen_coprime), 
                list(mod_exp_registers.left_operand_qubits) 
                + list(mod_exp_registers.right_operand_qubits) 
                + list(mod_exp_registers.carry_qubits) 
                + list(mod_exp_registers.temporary_qubits) 
                + list(mod_exp_registers.control_qubit)
                + list(mod_exp_registers.multiplier_qubits)
                + list(mod_exp_registers.counting_qubits)
            )
            mod_exp_circuit.measure(mod_exp_registers.counting_qubits, counting_output)
            mod_exp_circuit.measure(mod_exp_registers.multiplier_qubits, multiplier_output)
            mod_exp_circuit.measure(mod_exp_registers.control_qubit, control_output)
            mod_exp_circuit.measure(mod_exp_registers.temporary_qubits, temporary_output)
            mod_exp_circuit.measure(mod_exp_registers.carry_qubits, carry_output)
            mod_exp_circuit.measure(mod_exp_registers.right_operand_qubits, right_operand_output)
            mod_exp_circuit.measure(mod_exp_registers.left_operand_qubits, left_operand_output)
            
            sim = AerSimulator(method='matrix_product_state', matrix_product_state_max_bond_dimension=4096, seed_simulator=42)
            compiled_circuit = qiskit.transpile(mod_exp_circuit, sim)
            job = sim.run(compiled_circuit, shots=1)
            result = job.result()
            counts = result.get_counts(mod_exp_circuit)
            print(f"Forward: {counts}")
            verify_value(mod_exp_circuit, counts, expected_left_register, left_operand_output)
            verify_value(mod_exp_circuit, counts, expected_right_register, right_operand_output)
            verify_value(mod_exp_circuit, counts, expected_carry_register, carry_output)
            verify_value(mod_exp_circuit, counts, expected_mod_register, temporary_output)
            verify_value(mod_exp_circuit, counts, expected_control_register, control_output)
            verify_value(mod_exp_circuit, counts, expected_mult_register, multiplier_output)
            verify_value(mod_exp_circuit, counts, expected_counting_register, counting_output)
