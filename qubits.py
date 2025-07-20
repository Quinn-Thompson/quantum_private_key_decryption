import numpy as np
from typing import List, Union
import matplotlib.pyplot as plt
from matplotlib.quiver import Quiver
import qiskit
from numpy.typing import NDArray
from qiskit.quantum_info import Statevector, DensityMatrix, partial_trace, entropy
from qiskit_aer import Aer
from qiskit.visualization import plot_bloch_vector
from functools import partial
from matplotlib.animation import FuncAnimation
import timeit
from scipy.spatial.transform import Rotation as R, Slerp
from dataclasses import dataclass, fields
from copy import deepcopy


_ANCILARY_QUBITS = 1
_CIPHERTEXT_CUBITS = 1
_PLAINTEXT_QUBITS = 4
_KEY_QUBITS = 4
_FLIP_QUBIT = 1
_NUMBER_OF_QUBITS = _PLAINTEXT_QUBITS + _KEY_QUBITS + _FLIP_QUBIT
_NUMBER_OF_FRAMES = 33
_QUBIT_HILBERT_SPACE = 2
_INPUT_KEY = [0, 1, 1, 0]
_INPUT_PLAINTEXT = [1, 0, 1, 0]
_GROVER_ITERATIONS = 2

@dataclass
class QubitMatrices():
    state_matrices: List[List[NDArray[np.float64]]]
    mixed_matrices_1: List[List[NDArray[np.float64]]]
    mixed_matrices_2: List[List[NDArray[np.float64]]]

@dataclass
class Quivers():
    state_matrices: List[Quiver]
    mixed_matrices_1: List[Quiver]
    mixed_matrices_2: List[Quiver]
    state_matrices_color: str = "black"
    mixed_matrices_1_color: str = "cyan"
    mixed_matrices_2_color: str = "purple"
    state_matrices_alpha: float = 1.0
    mixed_matrices_1_alpha: float = 0.3
    mixed_matrices_2_alpha: float = 0.3

@dataclass
class QubitRegisters():
    plaintext_qubits: qiskit.QuantumRegister
    key_qubits: qiskit.QuantumRegister
    flip_qubit: qiskit.QuantumRegister




def get_x_y_z(matrix):
    x = 2 * np.real(matrix[0, 1])
    y = 2 * np.imag(matrix[0, 1])
    z = np.real(matrix[0, 0] - matrix[1, 1])
    return np.array([x, y, z])

def get_eigen_bloch_vector(eigen_vectors):
    bloch_vectors = []
    for i in range(_QUBIT_HILBERT_SPACE):
        eigen_vector = eigen_vectors[:, i]

        function = Statevector(eigen_vector)
        
        operator = np.array([[[0, 1], [1, 0]], [[0, -1j], [1j, 0]], [[1, 0], [0, -1]]])

        bloch_vector = np.real(function.data.conj().T @ operator @ function.data)
        bloch_vectors.append(bloch_vector)
    return bloch_vectors

def append_matrices(matrices: QubitMatrices, matrix_to_add, qubit_number):
    matrices.state_matrices[qubit_number].append(get_x_y_z(matrix_to_add))

    eigen_values, eigen_vectors = np.linalg.eigh(matrix_to_add)
    bloch_vectors = get_eigen_bloch_vector(eigen_vectors)
    matrices.mixed_matrices_1[qubit_number].append(bloch_vectors[0])
    matrices.mixed_matrices_2[qubit_number].append(bloch_vectors[1])

class bloch_spheres():
    def __init__(self, init_matrices: List[NDArray[np.float64]], number_of_qubits: int):
        self.number_of_qubits = number_of_qubits
        self.figure = plt.figure()
        self.axes = []
        self.bloch_vectors = Quivers([], [], [])
        empty_list = [[] for _ in range(self.number_of_qubits)]
        self.matrices = QubitMatrices(deepcopy(empty_list), deepcopy(empty_list), deepcopy(empty_list))
        self.current_matrix_index = 0
        self.last_known_rotation = QubitMatrices(deepcopy(empty_list), deepcopy(empty_list), deepcopy(empty_list))
        for qubit_number in range(self.number_of_qubits):
            self.axes.append(
                self.figure.add_subplot(int(np.ceil(np.sqrt(self.number_of_qubits))), int(np.ceil(np.sqrt(self.number_of_qubits))), qubit_number+1, projection='3d')
            )
            append_matrices(self.matrices, init_matrices[qubit_number], qubit_number)
            for field in fields(self.matrices):
                name = field.name
                matrices = getattr(self.matrices, name)
                last_known_rot_matrix = getattr(self.last_known_rotation, name)
                last_known_rot_matrix[qubit_number] = matrices[qubit_number][0] / np.linalg.norm(matrices[qubit_number][0])
                
        self.create_bloch_sphere()

    @staticmethod
    def normalize(v):
        return v / np.linalg.norm(v)

    def rotation_slerp(self, start_matrix, end_matrix, t):
        # written by chatGPT, needs to be reworked to utilize a half transform for correct rotations
        start_matrix = start_matrix / np.linalg.norm(start_matrix)
        end_matrix = end_matrix / np.linalg.norm(end_matrix)
        dot = np.dot(start_matrix, end_matrix)

        if np.isclose(dot, 1.0):
            return start_matrix 
        elif np.isclose(dot, -1.0):
            axis = np.cross(start_matrix, np.array([1, 0, 0]))
            if np.linalg.norm(axis) < 1e-6:
                axis = np.cross(start_matrix, np.array([0, 1, 0]))
            axis = axis / np.linalg.norm(axis)
            rot = R.from_rotvec(np.pi * axis)
        else:
            axis = np.cross(start_matrix, end_matrix)
            axis = axis / np.linalg.norm(axis)
            angle = np.arccos(dot)
            rot = R.from_rotvec(angle * axis)
        slerp = Slerp([0, 1], R.concatenate([R.identity(), rot]))
        rot_t = slerp([t])[0]
        return rot_t.apply(start_matrix)

    def update(self, frame: int):
        for qubit_number in range(self.number_of_qubits):
            for field in fields(self.matrices):
                name = field.name
                matrices = getattr(self.matrices, name)
                last_known_rot_matrix = getattr(self.last_known_rotation, name)
                bloch_quiver = getattr(self.bloch_vectors, name)
                current_matrix = matrices[qubit_number][self.current_matrix_index]
                next_matrix = matrices[qubit_number][self.current_matrix_index+1]
                interpolation_ratio = (frame+1) / _NUMBER_OF_FRAMES
                if (np.linalg.norm(current_matrix) == 0.0 and np.linalg.norm(next_matrix) != 0.0):
                    current_matrix += last_known_rot_matrix[qubit_number]*0.01
                elif (np.linalg.norm(current_matrix) != 0.0 and np.linalg.norm(next_matrix) == 0.0):
                    next_matrix += last_known_rot_matrix[qubit_number]*0.01
                    
                transition_matrix = self.rotation_slerp(current_matrix/np.linalg.norm(current_matrix), next_matrix/np.linalg.norm(next_matrix), interpolation_ratio)
                    
                # print(f"post_rotation {qubit_number}: {transition_matrix}")
                transition_matrix = transition_matrix * (np.linalg.norm(current_matrix) + (np.linalg.norm(next_matrix) - np.linalg.norm(current_matrix)) * interpolation_ratio)
                # print(f"post_magnitude {qubit_number}: {transition_matrix}")
                quiver_to_remove = bloch_quiver[qubit_number]
                quiver_to_remove.remove()
                
                bloch_quiver[qubit_number] = self.axes[qubit_number].quiver(
                    0, 0, 0, transition_matrix[0], transition_matrix[1], transition_matrix[2], color=getattr(self.bloch_vectors, f"{name}_color"), arrow_length_ratio=0.1, alpha = getattr(self.bloch_vectors, f"{name}_alpha")
                )
        if frame == (_NUMBER_OF_FRAMES) - 1:
            self.current_matrix_index += 1
            for field in fields(self.matrices):
                name = field.name
                last_known_rot_matrix = getattr(self.last_known_rotation, name)
                last_known_rot_matrix[qubit_number] = next_matrix / np.linalg.norm(next_matrix)
            
    def create_bloch_sphere(self):
        for qubit_number in range(self.number_of_qubits):
            u = np.linspace(0, 2 * np.pi, 20)
            v = np.linspace(0, np.pi, 20)
            x = np.outer(np.cos(u), np.sin(v))
            y = np.outer(np.sin(u), np.sin(v))
            z = np.outer(np.ones_like(u), np.cos(v))
            self.axes[qubit_number].plot_wireframe(x, y, z, color='lightblue', alpha=0.1, zorder=3)
            
            self.axes[qubit_number].quiver(0, 0, 0, 0.77, 0, 0, color='r', arrow_length_ratio=0.1, alpha=0.3, zorder=2)
            self.axes[qubit_number].quiver(0, 0, 0, 0, 0.77, 0, color='g', arrow_length_ratio=0.1, alpha=0.3, zorder=2)
            self.axes[qubit_number].quiver(0, 0, 0, 0, 0, 0.77, color='b', arrow_length_ratio=0.1, alpha=0.3, zorder=2)

            # Settings
            self.axes[qubit_number].set_xlim([-1, 1])
            self.axes[qubit_number].set_ylim([-1, 1])
            self.axes[qubit_number].set_zlim([-1, 1])
            self.axes[qubit_number].set_box_aspect([1,1,1])
            self.axes[qubit_number].axis('off')
            
            self.draw_quivers(0, qubit_number)
            
            self.axes[qubit_number].text(x=0.0, y=0.0, z=1.2, s='|0⟩', color='black', fontsize=8, zorder=1)
            self.axes[qubit_number].text(x=0.0, y=0.0, z=-1.4, s='|1⟩', color='black', fontsize=8, zorder=1)
            self.axes[qubit_number].text(x=0.0, y=1.1, z=0.0, s='y', color='black', fontsize=8, zorder=1)
            self.axes[qubit_number].text(x=1.1, y=0.0, z=0.0, s='x', color='black', fontsize=8, zorder=1)


    def draw_quivers(self, matrix_number, qubit_number):
        for field in fields(self.matrices):
            name = field.name
            matrices = getattr(self.matrices, name)
            bloch_quiver = getattr(self.bloch_vectors, name)
            # Initial Bloch vector
            bloch_quiver.append(
                self.axes[qubit_number].quiver(
                    0, 
                    0, 
                    0, 
                    matrices[qubit_number][matrix_number][0], 
                    matrices[qubit_number][matrix_number][1], 
                    matrices[qubit_number][matrix_number][2], 
                    color=getattr(self.bloch_vectors, f"{name}_color"), 
                    arrow_length_ratio=0.1,
                    alpha = getattr(self.bloch_vectors, f"{name}_alpha")
                )
            )            

@dataclass
class BlochSpherePlots():
    plaintext_qubits_plots: bloch_spheres
    key_qubits_plots: bloch_spheres
    flip_qubit_plots: bloch_spheres

def add_to_bloch_spheres(bloch_spheres: bloch_spheres, quantum_circuit: qiskit.QuantumCircuit, register: qiskit.QuantumRegister):
    quantum_vector = Statevector.from_instruction(quantum_circuit)
    qubits_in_register = [quantum_circuit.qubits.index(qubit) for qubit in register]
    keep_indices = [qubit for qubit in list(range(len(quantum_circuit.qubits))) if qubit not in qubits_in_register]
    reduced_state_vector = partial_trace(quantum_vector, keep_indices)
    for qubit_number in range(len(qubits_in_register)): 
        qubit_trace = [qubit for qubit in list(range(len(qubits_in_register)))]
        qubit_trace.remove(qubit_number)
        partial_trace_values = partial_trace(DensityMatrix(reduced_state_vector), qubit_trace).data
        
        append_matrices(bloch_spheres.matrices, partial_trace_values, qubit_number)

def von_neuman_info(qubit_trace_1, qubit_trace_2, qubit_trace_1_2):
    # compute von neumon entropy
    entropy_qubit_1 = entropy(qubit_trace_1, base=2)
    entropy_qubit_2 = entropy(qubit_trace_2, base=2)
    entropy_qubits_state = entropy(qubit_trace_1_2, base=2)

    # calculate mutal info
    mutal_information = (entropy_qubit_1 + entropy_qubit_2) - entropy_qubits_state
    return mutal_information

def mutual_entanglement(quantum_circuit: qiskit.QuantumCircuit):
    combination_matrices = np.empty((_NUMBER_OF_QUBITS, _NUMBER_OF_QUBITS), dtype=np.float64)
    density_matrix = DensityMatrix.from_instruction(quantum_circuit).data
    for qubit_number_1 in range(_NUMBER_OF_QUBITS):
        combination_qubit = list(range(_NUMBER_OF_QUBITS))
        qubit_trace_1 = list(range(_NUMBER_OF_QUBITS))
        qubit_trace_1.remove(qubit_number_1)
        combination_qubit.remove(qubit_number_1)
        for qubit_number_2 in range(_NUMBER_OF_QUBITS):
            if qubit_number_1 == qubit_number_2:
                continue
            qubit_trace_2 = list(range(_NUMBER_OF_QUBITS))
            qubit_trace_2.remove(qubit_number_2)
            combination_qubit.remove(qubit_number_2)
            von_neuman = von_neuman_info(
                partial_trace(density_matrix, qubit_trace_1).data, 
                partial_trace(density_matrix, qubit_trace_2).data, 
                partial_trace(density_matrix, combination_qubit).data, 
            )
            combination_matrices[qubit_number_1, qubit_number_2] = von_neuman
    print(combination_matrices)

def animate_bloch_sphere(bloch_spheres: bloch_spheres):
    number_of_frames = (len(bloch_spheres.matrices.state_matrices[0])-1)*_NUMBER_OF_FRAMES
    def update(frame):
        frame = frame % _NUMBER_OF_FRAMES
        bloch_spheres.update(frame)
        if frame == number_of_frames-1:
            plt.close(bloch_spheres.figure)

    _ = FuncAnimation(bloch_spheres.figure, update, frames=number_of_frames, interval=5, blit=False, repeat=False)
    plt.show()

def aes_step(quantum_circuit: qiskit.QuantumCircuit, qubit_registers: QubitRegisters, bloch_sphere_plots: BlochSpherePlots):
    # integrate key into plaintext encryption
    # this is the round key step
    for i in range(_KEY_QUBITS):
        quantum_circuit.cx(qubit_registers.key_qubits[i], qubit_registers.plaintext_qubits[i])
    add_to_bloch_spheres(bloch_sphere_plots.key_qubits_plots, quantum_circuit, qubit_registers.key_qubits)
    # some nonlinear substritution step
    # reversable tofolli gate, where 110 = 111, 111= 110
    quantum_circuit.ccx(qubit_registers.plaintext_qubits[0], qubit_registers.plaintext_qubits[1], qubit_registers.plaintext_qubits[2])
    add_to_bloch_spheres(bloch_sphere_plots.plaintext_qubits_plots, quantum_circuit, qubit_registers.plaintext_qubits)

    quantum_circuit.cx(qubit_registers.plaintext_qubits[0], qubit_registers.plaintext_qubits[1])
    add_to_bloch_spheres(bloch_sphere_plots.plaintext_qubits_plots, quantum_circuit, qubit_registers.plaintext_qubits)
    quantum_circuit.ccx(qubit_registers.plaintext_qubits[1], qubit_registers.plaintext_qubits[0], qubit_registers.plaintext_qubits[3])
    add_to_bloch_spheres(bloch_sphere_plots.plaintext_qubits_plots, quantum_circuit, qubit_registers.plaintext_qubits)
    
    # swap qubit 1 and 3, is basically shiftrow
    quantum_circuit.swap(qubit_registers.plaintext_qubits[2], qubit_registers.plaintext_qubits[3])
    add_to_bloch_spheres(bloch_sphere_plots.plaintext_qubits_plots, quantum_circuit, qubit_registers.plaintext_qubits)

def aes_step_inverse(quantum_circuit: qiskit.QuantumCircuit, qubit_registers: QubitRegisters, bloch_sphere_plots: BlochSpherePlots):
    # swap qubit 1 and 3, is basically shiftrow
    quantum_circuit.swap(qubit_registers.plaintext_qubits[2], qubit_registers.plaintext_qubits[3])
    add_to_bloch_spheres(bloch_sphere_plots.plaintext_qubits_plots, quantum_circuit, qubit_registers.plaintext_qubits)
    
    # some nonlinear substritution step
    # reversable tofolli gate, where 110 = 111, 111= 110

    quantum_circuit.ccx(qubit_registers.plaintext_qubits[1], qubit_registers.plaintext_qubits[0], qubit_registers.plaintext_qubits[3])
    add_to_bloch_spheres(bloch_sphere_plots.plaintext_qubits_plots, quantum_circuit, qubit_registers.plaintext_qubits)
    
    quantum_circuit.cx(qubit_registers.plaintext_qubits[0], qubit_registers.plaintext_qubits[1])
    add_to_bloch_spheres(bloch_sphere_plots.plaintext_qubits_plots, quantum_circuit, qubit_registers.plaintext_qubits)
    quantum_circuit.ccx(qubit_registers.plaintext_qubits[0], qubit_registers.plaintext_qubits[1], qubit_registers.plaintext_qubits[2])
    add_to_bloch_spheres(bloch_sphere_plots.plaintext_qubits_plots, quantum_circuit, qubit_registers.plaintext_qubits)
    # integrate key into plaintext encryption
    # this is the round key step
    for i in range(_KEY_QUBITS):
        quantum_circuit.cx(qubit_registers.key_qubits[i], qubit_registers.plaintext_qubits[i])
    add_to_bloch_spheres(bloch_sphere_plots.key_qubits_plots, quantum_circuit, qubit_registers.key_qubits)

def aes_circuit(
    quantum_circuit: qiskit.QuantumCircuit, 
    qubit_registers: QubitRegisters, 
    bloch_sphere_plots: BlochSpherePlots,
    classical_output: qiskit.ClassicalRegister,
    target_cipher: bytes
):
    for cipher_index, bit in enumerate(_INPUT_PLAINTEXT):
        if bit == 1:
            quantum_circuit.x(qubit_registers.plaintext_qubits[cipher_index])

    # entangle all qubits so they have the same probability
    quantum_circuit.h(qubit_registers.key_qubits)
    add_to_bloch_spheres(bloch_sphere_plots.key_qubits_plots, quantum_circuit, qubit_registers.key_qubits)
    # iplace the flag qubit into a state where it can easily flip
    quantum_circuit.x(qubit_registers.flip_qubit)
    quantum_circuit.h(qubit_registers.flip_qubit)
    
    for _ in range(_GROVER_ITERATIONS):
        oracle_find_matching_states(quantum_circuit, qubit_registers, bloch_sphere_plots, target_cipher)
        grover_diffuser(quantum_circuit, qubit_registers, bloch_sphere_plots)
    
    quantum_circuit.measure(qubit_registers.key_qubits, classical_output)
    
def oracle_find_matching_states(quantum_circuit: qiskit.QuantumCircuit, qubit_registers: QubitRegisters, bloch_sphere_plots: BlochSpherePlots, target_ciper: List[int]):
    # encrypt the plaintext 
    aes_step(quantum_circuit, qubit_registers, bloch_sphere_plots)
    
    # affect control bits so they are all 1s for MCX
    for cipher_index, bit in enumerate(target_ciper):
        if bit == 0:
            quantum_circuit.x(qubit_registers.plaintext_qubits[cipher_index])
    add_to_bloch_spheres(bloch_sphere_plots.plaintext_qubits_plots, quantum_circuit, qubit_registers.plaintext_qubits)

    # based on affect of the target cipher on the paintext, if the sets are all 1, flip the output qubit
    quantum_circuit.mcx(qubit_registers.plaintext_qubits, qubit_registers.flip_qubit)

    # reverse the change of the cipher onto the plaintext
    for cipher_index, bit in enumerate(target_ciper):
        if bit == 0:
            quantum_circuit.x(qubit_registers.plaintext_qubits[cipher_index])
    add_to_bloch_spheres(bloch_sphere_plots.plaintext_qubits_plots, quantum_circuit, qubit_registers.plaintext_qubits)

    # unencrypt the plaintext
    aes_step_inverse(quantum_circuit, qubit_registers, bloch_sphere_plots)

def grover_diffuser(quantum_circuit: qiskit.QuantumCircuit, qubit_registers: QubitRegisters, bloch_sphere_plots: BlochSpherePlots):
    # reflect the amplitude over the average so matching items probabilities gain amplitude
    
    quantum_circuit.h(qubit_registers.key_qubits)
    add_to_bloch_spheres(bloch_sphere_plots.key_qubits_plots, quantum_circuit, qubit_registers.key_qubits)
    # invert over the x axis
    quantum_circuit.x(qubit_registers.key_qubits)
    add_to_bloch_spheres(bloch_sphere_plots.key_qubits_plots, quantum_circuit, qubit_registers.key_qubits)
    # apply another hammond on the last key so we can phase flip all bits
    quantum_circuit.h(qubit_registers.key_qubits[-1])
    add_to_bloch_spheres(bloch_sphere_plots.key_qubits_plots, quantum_circuit, qubit_registers.key_qubits)
    # flip the phase of all bits relative to the last key
    quantum_circuit.mcx(qubit_registers.key_qubits[:-1], qubit_registers.key_qubits[-1])
    add_to_bloch_spheres(bloch_sphere_plots.key_qubits_plots, quantum_circuit, qubit_registers.key_qubits)
    # match the state of the last key to the rest
    quantum_circuit.h(qubit_registers.key_qubits[-1])
    add_to_bloch_spheres(bloch_sphere_plots.key_qubits_plots, quantum_circuit, qubit_registers.key_qubits)
    # move from the real states to the phase states
    quantum_circuit.x(qubit_registers.key_qubits)
    add_to_bloch_spheres(bloch_sphere_plots.key_qubits_plots, quantum_circuit, qubit_registers.key_qubits)
    quantum_circuit.h(qubit_registers.key_qubits)
    add_to_bloch_spheres(bloch_sphere_plots.key_qubits_plots, quantum_circuit, qubit_registers.key_qubits)

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

    
    quantum_vector = Statevector.from_instruction(quantum_circuit)
    bloch_sphere_dict = {}
    for field in fields(qubit_registers):
        name = field.name
        register = getattr(qubit_registers, name)
        qubits_in_register = [quantum_circuit.qubits.index(qubit) for qubit in register]
        initial_matrices = []
        keep_indices = [qubit for qubit in list(range(len(quantum_circuit.qubits))) if qubit not in qubits_in_register]
        reduced_state_vector = partial_trace(quantum_vector, keep_indices)
        for qubit_number in range(len(qubits_in_register)): 
            qubit_trace = [qubit for qubit in list(range(len(qubits_in_register)))]
            qubit_trace.remove(qubit_number)
            initial_matrices.append(partial_trace(DensityMatrix(reduced_state_vector), qubit_trace).data)
            
        bloch_sphere_dict[f"{name}_plots"] = bloch_spheres(initial_matrices, len(qubits_in_register))
    cipher_text = aes_ecnryption(_INPUT_PLAINTEXT, _INPUT_KEY)
    print(_INPUT_KEY)
    bloch_sphere_plots = BlochSpherePlots(**bloch_sphere_dict)

    # Hadamard gate
    aes_circuit(quantum_circuit, qubit_registers, bloch_sphere_plots, classical_output, cipher_text)
    # mutual_entanglement(quantum_circuit)
    sim = Aer.get_backend('aer_simulator')
    compiled_circuit = qiskit.transpile(quantum_circuit, sim)
    job = sim.run(compiled_circuit, shots=1024)
    result = job.result()

    counts = result.get_counts(quantum_circuit)
    print(counts)
    animate_bloch_sphere(bloch_sphere_plots.key_qubits_plots)
    # animate_bloch_sphere(bloch_sphere_plots.plaintext_qubits_plots)


def main():
    create_quantum_circuit()
    
if __name__ == "__main__":
    main()