# (C) Copyright IBM 2025
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

import copy
from math import prod

import numpy as np
import stim
import sinter


def filter_detectors_by_basis(
    circuit: stim.Circuit,
    basis: str,
    qubits: list[int] | None = None,
) -> stim.Circuit | tuple[stim.Circuit, list[str]]:
    """Return a new circuit filtering any detectors which do not detect the specified basis for the input qubits.

    Args:
        circuit: The original circuit
        basis: "X" or "Z"
        qubits: Data qubits to inject test errors on. Should typically be data qubits. Defaults
            to automatically detected data qubits which may not be robust.

    returns:
        The filtered circuit
    """
    assert basis in ("X", "Z")

    pauli_error = "Z" if basis == "X" else "X"

    circuit = circuit.flattened()

    noiseless_circuit = circuit.without_noise()
    sampler = noiseless_circuit.compile_detector_sampler()
    reference_detectors, reference_observables = sampler.sample(
        1, separate_observables=True
    )
    reference_detectors = reference_detectors[0, :]
    reference_observables = reference_observables[0, :]
    num_detectors = len(reference_detectors)

    detector_is_sensitive = np.full(num_detectors, False, dtype=bool)

    if qubits is None:
        to_test = detect_data_qubits(noiseless_circuit)
    else:
        to_test = qubits

    to_test_set = set(to_test)

    inst_idx = 0
    while to_test:
        for qubit in to_test:
            injected_circuit = stim.Circuit()
            injected_circuit += noiseless_circuit
            injected_circuit.insert(
                inst_idx,
                stim.CircuitInstruction(f"{pauli_error}_ERROR", [qubit], [1.0]),
            )

            injected_sampler = injected_circuit.compile_detector_sampler()
            injected_detectors, injected_observables = injected_sampler.sample(
                1, separate_observables=True
            )
            injected_detectors = injected_detectors[0, :]
            injected_observables = injected_observables[0, :]

            detectors_flipped = np.where(reference_detectors != injected_detectors)
            detector_is_sensitive[detectors_flipped] = True

        to_test = []
        for inst in noiseless_circuit[inst_idx:]:
            # Is a reset we must inject errors after
            inst_idx += 1
            if inst.name.startswith("R") or inst.name.startswith("M"):
                to_test = list(to_test_set)
                break

    filtered_circuit = stim.Circuit()
    detector_idx = 0
    for inst in circuit:
        if inst.name == "DETECTOR":
            to_insert = detector_is_sensitive[detector_idx]
            detector_idx += 1
            if not to_insert:
                continue
        filtered_circuit.append(inst)
    return filtered_circuit


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


def prob_at_least_one_error(marginals):
    return 1 - prod(1 - p for p in marginals)


def synthesize_joint_samples(stats, circuits):
    """Create synthetic decoding samples for joint decoding of both the X & Z basis from X/Z memory circuits.

    This is useful for comparing X/Z basis stats.
    See https://github.com/quantumlib/Stim/blob/f566b83c5da89ab94d7280178e2bd642350180c3/glue/sample/src/sinter/_data/_task_stats.py#L121
    for more information.
    """
    groups: dict[tuple[str, str], list] = {}
    for stat in stats:
        json_metadata = copy.copy(stat.json_metadata)
        circuit = json_metadata.pop("circuit", None)
        if circuit is None or circuit not in circuits:
            continue
        key = (stat.decoder, str(json_metadata))
        group = groups.setdefault(key, [])
        group.append(stat)

    # Merge together stats
    # assuming trials are independent and iid.
    new_stats = []
    for key, group in groups.items():
        new_strong_id = ""
        shots = np.inf
        discards = 0
        seconds = 0
        custom_counts = None

        marginals = []
        for stat in group:
            new_strong_id += stat.strong_id
            p_marginal = stat.errors / stat.shots
            marginals.append(p_marginal)
            shots = min(stat.shots, shots)

            discards += stat.discards
            seconds += stat.seconds
            if custom_counts is None:
                custom_counts = stat.custom_counts
            else:
                custom_counts += stat.custom_counts

        errors_either = max(int(prob_at_least_one_error(marginals) * shots), 0)

        new_metadata = copy.copy(group[0].json_metadata)
        new_metadata.pop("circuit", None)
        new_stats.append(
            sinter.TaskStats(
                decoder=group[0].decoder,
                strong_id=new_strong_id,
                json_metadata=new_metadata,
                shots=shots,
                errors=errors_either,
                discards=discards,
                seconds=seconds,
                custom_counts=custom_counts,
            )
        )
    return new_stats
