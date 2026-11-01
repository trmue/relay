# (C) Copyright IBM 2025
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

import numpy as np
from scipy.sparse import csc_matrix
import pathlib
import sinter
import stim
import tempfile

from relay_bp.stim import (
    SinterDecoder_RelayBP,
    sinter_decoders,
    CheckMatrices,
)
from relay_bp.stim.sinter.check_matrices import combine_independent_priors

from testdata import (
    get_test_circuit,
    get_all_test_circuits,
    filter_detectors_by_basis,
)


def test_check_matrix_pruning():
    """Test decoding of the surface code via files."""
    circuit = stim.Circuit.generated(
        rounds=11,
        distance=11,
        after_clifford_depolarization=0.003,
        code_task=f"surface_code:rotated_memory_z",
    )
    dem = circuit.detector_error_model(decompose_errors=True)

    check_matrices = CheckMatrices.from_dem(
        dem, decomposed_hyperedges=True, prune_decided_errors=False
    )
    check_matrices_pruned = CheckMatrices.from_dem(
        dem, decomposed_hyperedges=True, prune_decided_errors=True
    )

    assert (
        check_matrices.check_matrix.shape[0]
        == check_matrices_pruned.check_matrix.shape[0]
    )
    assert (
        check_matrices.check_matrix.shape[1]
        > check_matrices_pruned.check_matrix.shape[1]
    )
    assert (
        check_matrices.observables_matrix.shape[0]
        == check_matrices_pruned.observables_matrix.shape[0]
    )
    assert (
        check_matrices.observables_matrix.shape[1]
        > check_matrices_pruned.observables_matrix.shape[1]
    )


def _circuit_with_randomized_qubit(
    distance: int, qubit: int, tick: int
) -> stim.Circuit:
    """Fully depolarize a single qubit after one TICK, a crude model of leakage."""
    circuit = stim.Circuit.generated(
        rounds=distance,
        distance=distance,
        after_clifford_depolarization=0.003,
        before_measure_flip_probability=0.003,
        code_task="surface_code:rotated_memory_z",
    )
    randomized = stim.Circuit()
    ticks = 0
    for instruction in circuit.flattened():
        randomized.append(instruction)
        if instruction.name == "TICK":
            ticks += 1
            if ticks == tick:
                randomized.append("DEPOLARIZE1", [qubit], 0.75)
    return randomized


def test_check_matrix_priors_combined_and_clamped():
    """Priors of mechanisms sharing an edge combine to at most 1/2 rather than summing."""
    circuit = _circuit_with_randomized_qubit(distance=3, qubit=8, tick=10)
    dem = circuit.detector_error_model(
        decompose_errors=True, approximate_disjoint_errors=True
    )

    check_matrices = CheckMatrices.from_dem(dem, decomposed_hyperedges=True)
    assert np.all(check_matrices.error_priors > 0.0)
    assert np.all(check_matrices.error_priors < 0.5)
    assert check_matrices.syndrome_bias is None

    decoder = SinterDecoder_RelayBP().compile_decoder_for_dem(dem=dem)
    shots, observables = circuit.compile_detector_sampler(seed=0).sample(
        200, separate_observables=True, bit_packed=True
    )
    predictions = decoder.decode_shots_bit_packed(bit_packed_detection_event_data=shots)
    assert np.mean(np.any(predictions != observables, axis=1)) < 0.1


def test_combine_independent_priors():
    combine_matrix = csc_matrix(np.array([[1, 0, 1], [0, 1, 1], [1, 1, 1]]))
    combined = combine_independent_priors(combine_matrix, np.array([0.1, 0.5, 0.2]))
    assert np.allclose(combined, [0.1 * 0.8 + 0.9 * 0.2, 0.5, 0.5])

    combined = combine_independent_priors(combine_matrix, np.array([0.1, 0.0, 0.2]))
    assert np.allclose(combined, [0.26, 0.2, 0.26])

    # Mechanisms close to 1/2 never combine above 1/2
    combined = combine_independent_priors(combine_matrix, np.array([0.45, 0.48, 0.49]))
    assert np.all(combined < 0.5)


def test_check_matrix_decided_error_bias():
    """Biases from decided errors are 1-D uint8 arrays."""
    check_matrices = CheckMatrices(
        check_matrix=csc_matrix(np.array([[1, 1, 0], [0, 1, 1]], dtype=np.uint8)),
        observables_matrix=csc_matrix(np.array([[1, 0, 1]], dtype=np.uint8)),
        error_priors=np.array([1.0, 0.1, 0.1]),
    ).prune_decided_errors()

    assert isinstance(check_matrices.syndrome_bias, np.ndarray)
    assert check_matrices.syndrome_bias.dtype == np.uint8
    assert np.all(check_matrices.syndrome_bias == np.array([1, 0]))
    assert np.all(check_matrices.observables_bias == np.array([1]))
    assert check_matrices.check_matrix.shape == (2, 2)


def test_sinter_relay_bp_decoder_integration():
    """Test decoding of the surface code with sinter."""

    def generate_example_tasks():
        for p in [0.0001]:
            for d in [3]:
                yield sinter.Task(
                    circuit=stim.Circuit.generated(
                        rounds=d,
                        distance=d,
                        after_clifford_depolarization=p,
                        code_task=f"surface_code:rotated_memory_x",
                    ),
                    json_metadata={
                        "p": p,
                        "d": d,
                    },
                )

    samples = sinter.collect(
        num_workers=2,
        max_shots=1_00,
        tasks=generate_example_tasks(),
        decoders=["relay-bp"],
        custom_decoders=sinter_decoders(),
    )
    assert samples[0].decoder == "relay-bp"
    assert samples[0].errors <= 5
    assert samples[0].shots == 100


def test_sinter_msl_bp_decoder_integration():
    """Test decoding of the surface code with sinter."""

    def generate_example_tasks():
        for p in [0.0001]:
            for d in [3]:
                yield sinter.Task(
                    circuit=stim.Circuit.generated(
                        rounds=d,
                        distance=d,
                        after_clifford_depolarization=p,
                        code_task=f"surface_code:rotated_memory_x",
                    ),
                    json_metadata={
                        "p": p,
                        "d": d,
                    },
                )

    samples = sinter.collect(
        num_workers=2,
        max_shots=1_00,
        tasks=generate_example_tasks(),
        decoders=["msl-bp"],
        custom_decoders=sinter_decoders(),
    )
    assert samples[0].decoder == "msl-bp"
    assert samples[0].errors <= 10
    assert samples[0].shots == 100


def test_sinter_mem_bp_decoder_integration():
    """Test decoding of the surface code with sinter."""

    def generate_example_tasks():
        for p in [0.0001]:
            for d in [3]:
                yield sinter.Task(
                    circuit=stim.Circuit.generated(
                        rounds=d,
                        distance=d,
                        after_clifford_depolarization=p,
                        code_task=f"surface_code:rotated_memory_x",
                    ),
                    json_metadata={
                        "p": p,
                        "d": d,
                    },
                )

    # Collect the samples (takes a few minutes).
    samples = sinter.collect(
        num_workers=2,
        max_shots=1_00,
        tasks=generate_example_tasks(),
        decoders=["mem-bp"],
        custom_decoders=sinter_decoders(),
    )
    assert samples[0].decoder == "mem-bp"
    assert samples[0].errors <= 20
    assert samples[0].shots == 100


def test_sinter_decode_via_files():
    """Test decoding of the surface code via files."""
    circuit = stim.Circuit.generated(
        rounds=3,
        distance=3,
        after_clifford_depolarization=0.0001,
        code_task=f"surface_code:rotated_memory_x",
    )
    dem = circuit.detector_error_model()

    with tempfile.TemporaryDirectory() as d:
        testdir = pathlib.Path(d)

        circuit.compile_detector_sampler().sample_write(
            shots=100,
            filepath=testdir / "detectors.b8",
            format="b8",
        )

        dem.to_file(testdir / "dem.dem")

        SinterDecoder_RelayBP(parallel=True).decode_via_files(
            num_shots=10,
            num_dets=dem.num_detectors,
            num_obs=dem.num_observables,
            dem_path=testdir / "dem.dem",
            dets_b8_in_path=testdir / "detectors.b8",
            obs_predictions_b8_out_path=testdir / "observable_predictions.b8",
            tmp_dir=testdir,
        )

        predictions = stim.read_shot_data_file(
            path=testdir / "observable_predictions.b8",
            format="b8",
            num_observables=dem.num_observables,
        )
        assert np.sum(predictions) <= 5


def test_get_testdata_circuit():
    """Test getting test circuit and decoding."""
    circuit = get_test_circuit("bicycle_bivariate_18_4_3_memory_Z", 0.001)
    tasks = [sinter.Task(circuit=circuit)]

    samples = sinter.collect(
        num_workers=2,
        max_shots=1_00,
        tasks=tasks,
        decoders=["relay-bp"],
        custom_decoders=sinter_decoders(),
    )

    assert samples[0].decoder == "relay-bp"
    assert samples[0].errors <= 10
    assert samples[0].shots == 100


def test_get_all_testdata_circuit():
    """Test getting test circuit and decoding."""
    circuits = get_all_test_circuits("*", 0.001)
    for name, circuit in circuits.items():
        assert isinstance(name, str)
        assert isinstance(circuit, stim.Circuit)

    assert len(circuits) > 1


def test_filter_detectors_by_basis():
    """Test getting test circuit and decoding."""
    circuit = get_test_circuit("bicycle_bivariate_18_4_3_memory_Z", 0.001)

    dem = circuit.detector_error_model()
    check_matrices = CheckMatrices.from_dem(dem)

    assert check_matrices.check_matrix.shape == (54, 1800)

    z_circuit = filter_detectors_by_basis(circuit, "Z")
    z_dem = z_circuit.detector_error_model()
    z_check_matrices = CheckMatrices.from_dem(z_dem)

    assert z_check_matrices.check_matrix.shape == (36, 288)
