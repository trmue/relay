# (C) Copyright IBM 2025
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

import stim


def detect_data_qubits(circuit: stim.Circuit) -> list[int]:
    """Detect data qubits as those that are only measured once in a circuit.

    Warning: This is hacky and likely will only work with your typical memory circuits.
    """
    qubit_times_measured = [0 for qubit in range(circuit.num_qubits)]

    for inst in circuit:
        if inst.name.startswith("M") and not inst.gate_args_copy():
            for qubit in inst.targets_copy():
                qubit_times_measured[qubit.qubit_value] += 1

    return [
        qubit
        for qubit, times_measured in enumerate(qubit_times_measured)
        if times_measured == 1
    ]


def insert_uniform_academic_circuit_noise(
    circuit: stim.Circuit,
    p: float,
    apply_to_final_data_qubit_measurement: bool = False,
    is_repeat_block: bool = False,
    data_qubits: list[int] | None = None,
) -> stim.Circuit:
    """The noise model presented in arXiv:2308.07915, adapted to the PhysicalCircuit gate set.

    > It assumes that each operation in the circuit is ideal or faulty with the probability 1 −p or p respectively.
      Here p is a model parameter called the error rate. Faults on different operations occur independently.
      We define faulty operations as follows.  A faulty CNOT is an ideal CNOT followed by one of 15
      non-identity Pauli errors on the control and the target qubits picked uniformly at random.
      A faulty initialization prepares a single-qubit state orthogonal to the correct one. A faulty
      measurement is an ideal measurement followed by a classical bit-flip error applied to the
      measurement outcome. A faulty idle qubit suffers from a Pauli error X or Y or Z picked
      uniformly at random.

    Args:
        circuit: The circuit to insert noise to.
        p: The uniform noise strength.
        apply_to_final_data_qubit_measurement: Apply the noise to the final measurements on data qubits.
        is_repeat_block: Is this a recursive call into a repeat block?
        data_qubits: Circuit data qubits.

    Returns:
        A new stim circuit with noise inserted to the original.
    """
    assert 0 <= p <= 1

    noisy_circuit = stim.Circuit()

    total_num_measurements = circuit.num_measurements
    measurements_processed = 0

    if data_qubits is None:
        data_qubits = detect_data_qubits(circuit)

    idle_qubits: set[int] = set()
    tick_contains_non_trivial = False

    for instr in circuit:
        if isinstance(instr, stim.CircuitRepeatBlock):
            noisy_block_circuit = insert_uniform_academic_circuit_noise(
                instr.body_copy(),
                p,
                apply_to_final_data_qubit_measurement=True,
                is_repeat_block=True,
                data_qubits=data_qubits,
            )
            noisy_circuit.append(
                stim.CircuitRepeatBlock(instr.repeat_count, noisy_block_circuit)
            )
            measurements_processed += instr.num_measurements
        else:
            measurements_processed += instr.num_measurements
            name = instr.name

            idle_qubits -= set(target.qubit_value for target in instr.targets_copy())

            if name == "TICK":
                if idle_qubits and tick_contains_non_trivial:
                    noisy_circuit.append("DEPOLARIZE1", list(idle_qubits), [p])

                noisy_circuit.append(instr)
                idle_qubits = set(range((noisy_circuit.num_qubits)))
                tick_contains_non_trivial = False
            elif name == "M":
                apply_measurement(
                    noisy_circuit,
                    "M",
                    instr.targets_copy(),
                    p,
                    data_qubits,
                    apply_to_final_data_qubit_measurement,
                )
                tick_contains_non_trivial = True
            elif name == "MZ":
                apply_measurement(
                    noisy_circuit,
                    "MZ",
                    instr.targets_copy(),
                    p,
                    data_qubits,
                    apply_to_final_data_qubit_measurement,
                )
                tick_contains_non_trivial = True
            elif name == "MX":
                apply_measurement(
                    noisy_circuit,
                    "MX",
                    instr.targets_copy(),
                    p,
                    data_qubits,
                    apply_to_final_data_qubit_measurement,
                )
                tick_contains_non_trivial = True
            elif name == "MY":
                apply_measurement(
                    noisy_circuit,
                    "MY",
                    instr.targets_copy(),
                    p,
                    data_qubits,
                    apply_to_final_data_qubit_measurement,
                )
                tick_contains_non_trivial = True
            elif name == "MR":
                apply_measurement(
                    noisy_circuit,
                    "MR",
                    instr.targets_copy(),
                    p,
                    data_qubits,
                    apply_to_final_data_qubit_measurement,
                )
                tick_contains_non_trivial = True
            elif name == "R":
                noisy_circuit.append("R", instr.targets_copy(), instr.gate_args_copy())
                noisy_circuit.append("X_ERROR", instr.targets_copy(), [p])
                tick_contains_non_trivial = True
            elif name == "RZ":
                noisy_circuit.append("RZ", instr.targets_copy(), instr.gate_args_copy())
                noisy_circuit.append("X_ERROR", instr.targets_copy(), [p])
                tick_contains_non_trivial = True
            elif name == "RX":
                noisy_circuit.append("RX", instr.targets_copy(), instr.gate_args_copy())
                noisy_circuit.append("Z_ERROR", instr.targets_copy(), [p])
                tick_contains_non_trivial = True
            elif name == "RY":
                noisy_circuit.append("RY", instr.targets_copy(), instr.gate_args_copy())
                noisy_circuit.append("X_ERROR", instr.targets_copy(), [p])
                noisy_circuit.append("Z_ERROR", instr.targets_copy(), [p])
                tick_contains_non_trivial = True
            elif name == "CZ":
                noisy_circuit.append("CZ", instr.targets_copy())
                noisy_circuit.append("DEPOLARIZE2", instr.targets_copy(), [p])
                tick_contains_non_trivial = True
            elif name == "CX":
                noisy_circuit.append("CX", instr.targets_copy())
                noisy_circuit.append("DEPOLARIZE2", instr.targets_copy(), [p])
                tick_contains_non_trivial = True
            elif name == "CY":
                noisy_circuit.append("CY", instr.targets_copy())
                noisy_circuit.append("DEPOLARIZE2", instr.targets_copy(), [p])
                tick_contains_non_trivial = True
            else:
                noisy_circuit.append(instr)

    if is_repeat_block and idle_qubits and tick_contains_non_trivial:
        noisy_circuit.append("DEPOLARIZE1", list(idle_qubits), [p])

    return noisy_circuit


def apply_measurement(
    circuit, measurement, targets, p, data_qubits, apply_to_final_data_qubit_measurement
):
    if apply_to_final_data_qubit_measurement:
        circuit.append(measurement, targets, [p])
        return

    is_data_qubit_target = [target.value in data_qubits for target in targets]
    for target, is_data_qubit_target_ in zip(targets, is_data_qubit_target):
        circuit.append(
            measurement, target.value, None if is_data_qubit_target_ else [p]
        )
