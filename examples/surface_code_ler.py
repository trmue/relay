# (C) Copyright IBM 2026
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
"""Logical error rate of the rotated surface code with unconverged shots counted as failures.

Sinter only compares the predicted observables with the actual observable flips, so a shot
whose decoding did not converge counts as a success when its prediction happens to be right.
This script decodes with ``ObservableDecoderRunner.decode_detailed_batch``, which reports
convergence for every shot, and counts a shot as a failure if its predicted observables are
wrong or its decoding did not converge. The Sinter count, which only checks the observables,
is reported alongside.

The circuits are the rotated surface code memory experiments from ``tests/testdata`` with
distance and rounds 11 under uniform circuit noise. The X and Z memory experiments are decoded
separately and combined per shot as p_L = 1 - (1 - p_X)(1 - p_Z). The decoders are plain
min-sum BP, Relay-BP with XZ decoding and Relay-BP with XYZ decoding, and optionally
PyMatching as a reference.

Each basis stops after ``--max-errors`` failures or at its shot cap. Results are appended to
the CSV given by ``--out`` as they are computed, and points already in the file are skipped,
so an interrupted run can be resumed.

Usage::

    python examples/surface_code_ler.py --test
    python examples/surface_code_ler.py --out surface_code_ler.csv
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np

import relay_bp
from relay_bp.stim import CheckMatrices

sys.path.append(str(Path(__file__).resolve().parent.parent / "tests"))
import testdata  # noqa: E402

CIRCUITS = ["rotated_surface_code_memory_X", "rotated_surface_code_memory_Z"]
ERROR_RATES = [0.001, 0.002, 0.003, 0.004, 0.005]

RELAY = dict(
    alpha=1.0,
    gamma0=0.55,
    gamma_dist_interval=(-0.1746, 0.7979),
    pre_iter=80,
    num_sets=600,
    set_max_iter=60,
    stop_nconv=5,
    stopping_criterion="nconv",
)
BP_MAX_ITER = 10000

# Decoder label: (kind, XYZ decoding)
ALL_DECODERS = {
    "BP": ("bp", False),
    "Relay-BP-5": ("relay", False),
    "XYZ-Relay-BP-5": ("relay", True),
    "Matching": ("matching", False),
}
DEFAULT_DECODERS = ["BP", "Relay-BP-5", "XYZ-Relay-BP-5"]

# Maximum number of shots per basis, by physical error rate
SHOTS_XZ = {
    0.001: 10_000_000,
    0.002: 3_000_000,
    0.003: 300_000,
    0.004: 100_000,
    0.005: 30_000,
}
SHOTS_XYZ = {
    0.001: 2_000_000,
    0.002: 2_000_000,
    0.003: 1_000_000,
    0.004: 300_000,
    0.005: 100_000,
}
SHOTS_MATCHING = {
    0.001: 1_000_000_000,
    0.002: 20_000_000,
    0.003: 2_000_000,
    0.004: 500_000,
    0.005: 200_000,
}
SHOTS = {
    "BP": SHOTS_XZ,
    "Relay-BP-5": SHOTS_XZ,
    "XYZ-Relay-BP-5": SHOTS_XYZ,
    "Matching": SHOTS_MATCHING,
}

MAX_ERRORS = 100
CHUNK = 100_000
MIN_CHUNK = 200
MATCHING_CHUNK = 1_000_000
MATCHING_MIN_CHUNK = 10_000

CSV_FIELDS = [
    "decoder",
    "xyz",
    "p",
    "d",
    "max_shots",
    "shots_X",
    "shots_Z",
    "ler_shot",
    "ler_per_cycle",
    "ler_X",
    "ler_Z",
    "errs_X",
    "errs_Z",
    "nonconv_X",
    "nonconv_Z",
    "wrong_obs_X",
    "wrong_obs_Z",
    "ler_per_cycle_obs_only",
    "gamma0",
    "low",
    "high",
    "num_sets",
    "seconds",
]


def build_runner(
    kind: str, check_matrices: CheckMatrices
) -> relay_bp.ObservableDecoderRunner:
    """Build the observable decoder for one decoder kind.

    Args:
        kind: ``"bp"`` for plain min-sum BP or ``"relay"`` for Relay-BP.
        check_matrices: The decoding problem.

    Returns:
        The observable decoder runner.

    Raises:
        ValueError: If ``kind`` is not a known decoder kind.
    """
    if kind == "bp":
        decoder = relay_bp.MinSumBPDecoderF64(
            check_matrices.check_matrix,
            error_priors=check_matrices.error_priors,
            max_iter=BP_MAX_ITER,
            alpha=RELAY["alpha"],
            gamma0=None,
        )
    elif kind == "relay":
        decoder = relay_bp.RelayDecoderF64(
            check_matrices.check_matrix,
            error_priors=check_matrices.error_priors,
            alpha=None if RELAY["alpha"] == 0.0 else RELAY["alpha"],
            gamma0=RELAY["gamma0"],
            pre_iter=RELAY["pre_iter"],
            num_sets=RELAY["num_sets"],
            set_max_iter=RELAY["set_max_iter"],
            gamma_dist_interval=RELAY["gamma_dist_interval"],
            stop_nconv=RELAY["stop_nconv"],
            stopping_criterion=RELAY["stopping_criterion"],
        )
    else:
        raise ValueError(f"Unknown decoder kind: {kind}")
    return relay_bp.ObservableDecoderRunner(
        decoder, check_matrices.observables_matrix, include_decode_result=False
    )


def next_chunk_size(
    fails: int, done: int, max_errors: int, min_chunk: int, max_chunk: int
) -> int:
    """Choose the number of shots for the next chunk.

    The chunk aims at the remaining failures at the failure rate seen so far, grows by at most
    a factor of two per chunk, and stays between ``min_chunk`` and ``max_chunk``.

    Args:
        fails: Failures seen so far.
        done: Shots decoded so far.
        max_errors: Number of failures at which the basis stops.
        min_chunk: Smallest chunk size.
        max_chunk: Largest chunk size.

    Returns:
        The number of shots for the next chunk.
    """
    if fails == 0:
        target = max_chunk
    else:
        target = int((max_errors - fails) * done / fails * 1.1) + 1
    return max(min_chunk, min(max_chunk, done, target))


def run_bp(
    circuit_name: str,
    kind: str,
    xyz: bool,
    distance: int,
    p: float,
    num_shots: int,
    seed: int,
    max_errors: int | None,
) -> tuple[int, int, int, int]:
    """Decode one memory experiment with BP or Relay-BP.

    Args:
        circuit_name: Name of the test circuit.
        kind: ``"bp"`` or ``"relay"``.
        xyz: Decode X and Z detectors together instead of the detectors of one basis.
        distance: Code distance, also used as the number of rounds.
        p: Physical error rate.
        num_shots: Maximum number of shots.
        seed: Seed of the stim sampler.
        max_errors: Stop after this many failures, or ``None`` to decode all shots.

    Returns:
        The number of shots, the failures with unconverged shots counted as failures, the
        unconverged shots, and the shots with wrong predicted observables.
    """
    circuit = testdata.get_test_circuit(
        circuit_name, distance=distance, rounds=distance, error_rate=p
    )
    if not xyz:
        circuit = testdata.filter_detectors_by_basis(circuit, circuit_name[-1])
    dem = circuit.detector_error_model(decompose_errors=not xyz)
    check_matrices = CheckMatrices.from_dem(
        dem, prune_decided_errors=True, threshold=0.0
    )
    runner = build_runner(kind, check_matrices)

    sampler = circuit.compile_detector_sampler(seed=seed)
    fails = unconverged = wrong_obs = done = 0
    n = CHUNK if max_errors is None else MIN_CHUNK
    while done < num_shots and (max_errors is None or fails < max_errors):
        n = min(n, num_shots - done)
        detectors, observables = sampler.sample(n, separate_observables=True)
        syndromes = detectors.astype(np.uint8)
        if check_matrices.syndrome_bias is not None:
            syndromes = (syndromes + check_matrices.syndrome_bias) % 2
        results = runner.decode_detailed_batch(
            syndromes, parallel=True, progress_bar=False
        )
        decodings = np.array([result.decoding for result in results], dtype=np.uint8)
        converged = np.array([result.success for result in results], dtype=bool)
        predictions = np.asarray(check_matrices.observables_matrix @ decodings.T).T % 2
        if check_matrices.observables_bias is not None:
            predictions = (predictions + check_matrices.observables_bias) % 2
        mismatch = np.any(predictions.astype(bool) != observables, axis=1)
        fails += int(np.count_nonzero(mismatch | ~converged))
        unconverged += int(np.count_nonzero(~converged))
        wrong_obs += int(np.count_nonzero(mismatch))
        done += n
        if max_errors is not None:
            n = next_chunk_size(fails, done, max_errors, MIN_CHUNK, CHUNK)
    return done, fails, unconverged, wrong_obs


def run_matching(
    circuit_name: str,
    distance: int,
    p: float,
    num_shots: int,
    seed: int,
    max_errors: int | None,
) -> tuple[int, int, int, int]:
    """Decode one memory experiment with PyMatching on the decomposed detector error model.

    PyMatching always returns a correction, so there are no unconverged shots.

    Args:
        circuit_name: Name of the test circuit.
        distance: Code distance, also used as the number of rounds.
        p: Physical error rate.
        num_shots: Maximum number of shots.
        seed: Seed of the stim sampler.
        max_errors: Stop after this many failures, or ``None`` to decode all shots.

    Returns:
        The number of shots, the failures, zero unconverged shots, and the failures again as
        the shots with wrong predicted observables.
    """
    import pymatching

    circuit = testdata.get_test_circuit(
        circuit_name, distance=distance, rounds=distance, error_rate=p
    )
    matcher = pymatching.Matching.from_detector_error_model(
        circuit.detector_error_model(decompose_errors=True)
    )
    sampler = circuit.compile_detector_sampler(seed=seed)
    fails = done = 0
    n = MATCHING_CHUNK if max_errors is None else MATCHING_MIN_CHUNK
    while done < num_shots and (max_errors is None or fails < max_errors):
        n = min(n, num_shots - done)
        detectors, observables = sampler.sample(
            n, separate_observables=True, bit_packed=True
        )
        predictions = matcher.decode_batch(
            detectors, bit_packed_shots=True, bit_packed_predictions=True
        )
        fails += int(np.count_nonzero(np.any(predictions != observables, axis=1)))
        done += n
        if max_errors is not None:
            n = next_chunk_size(
                fails, done, max_errors, MATCHING_MIN_CHUNK, MATCHING_CHUNK
            )
    return done, fails, 0, fails


def per_cycle(p_shot: float, rounds: int) -> float:
    """Convert a logical error rate per shot into a logical error rate per round.

    Args:
        p_shot: Logical error rate per shot.
        rounds: Number of rounds per shot.

    Returns:
        The logical error rate per round.
    """
    return 1 - (1 - p_shot) ** (1.0 / rounds) if p_shot < 1 else 1.0


def read_done_points(path: Path) -> set[tuple[str, float]]:
    """Read the (decoder, physical error rate) points already in a results file.

    Args:
        path: The CSV file.

    Returns:
        The points in the file, empty if the file does not exist or is empty.
    """
    if not path.is_file() or path.stat().st_size == 0:
        return set()
    with open(path, newline="") as file:
        reader = csv.DictReader(file)
        if reader.fieldnames != CSV_FIELDS:
            sys.exit(f"{path} has different columns, use a new --out file")
        return {(row["decoder"], float(row["p"])) for row in reader}


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--test", action="store_true", help="quick check with d=5 and few shots"
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="CSV file, existing (decoder, p) points are skipped",
    )
    parser.add_argument(
        "--decoders", nargs="+", default=DEFAULT_DECODERS, choices=list(ALL_DECODERS)
    )
    parser.add_argument("--gamma0", type=float, default=RELAY["gamma0"])
    parser.add_argument(
        "--interval",
        type=float,
        nargs=2,
        metavar=("LOW", "HIGH"),
        default=RELAY["gamma_dist_interval"],
    )
    parser.add_argument("--num-sets", type=int, default=RELAY["num_sets"])
    parser.add_argument(
        "--max-errors",
        type=int,
        default=MAX_ERRORS,
        help=f"stop a basis after this many failures, 0 decodes all shots (default {MAX_ERRORS})",
    )
    args = parser.parse_args()
    max_errors = args.max_errors if args.max_errors > 0 else None
    RELAY.update(
        gamma0=args.gamma0,
        gamma_dist_interval=tuple(args.interval),
        num_sets=args.num_sets,
    )

    distance = 5 if args.test else 11
    shots = (
        {label: {p: 2000 for p in ERROR_RATES} for label in ALL_DECODERS}
        if args.test
        else SHOTS
    )

    done_points: set[tuple[str, float]] = set()
    file = writer = None
    if args.out:
        out = Path(args.out)
        done_points = read_done_points(out)
        if done_points:
            print(f"Resuming: {len(done_points)} points already in {out}")
        file = open(out, "a", newline="")
        writer = csv.DictWriter(file, fieldnames=CSV_FIELDS)
        if not done_points:
            writer.writeheader()

    print(
        f"Relay-BP: gamma0={RELAY['gamma0']}, interval={RELAY['gamma_dist_interval']}, "
        f"num_sets={RELAY['num_sets']}, stop_nconv={RELAY['stop_nconv']}, d={distance}"
    )
    for label in args.decoders:
        kind, xyz = ALL_DECODERS[label]
        for p in ERROR_RATES:
            if (label, p) in done_points:
                print(f"{label:15s} p={p:<6} already in {args.out}, skipping")
                continue
            num_shots = shots[label][p]
            start = time.time()
            per_basis = {}
            for circuit_name in CIRCUITS:
                if kind == "matching":
                    result = run_matching(
                        circuit_name, distance, p, num_shots, args.seed, max_errors
                    )
                else:
                    result = run_bp(
                        circuit_name,
                        kind,
                        xyz,
                        distance,
                        p,
                        num_shots,
                        args.seed,
                        max_errors,
                    )
                per_basis[circuit_name[-1]] = result
            shots_x, errs_x, nonconv_x, wrong_x = per_basis["X"]
            shots_z, errs_z, nonconv_z, wrong_z = per_basis["Z"]
            p_x, p_z = errs_x / shots_x, errs_z / shots_z
            p_shot = 1 - (1 - p_x) * (1 - p_z)
            p_shot_obs = 1 - (1 - wrong_x / shots_x) * (1 - wrong_z / shots_z)
            row = dict(
                decoder=label,
                xyz=xyz,
                p=p,
                d=distance,
                max_shots=num_shots,
                shots_X=shots_x,
                shots_Z=shots_z,
                ler_shot=p_shot,
                ler_per_cycle=per_cycle(p_shot, distance),
                ler_X=p_x,
                ler_Z=p_z,
                errs_X=errs_x,
                errs_Z=errs_z,
                nonconv_X=nonconv_x,
                nonconv_Z=nonconv_z,
                wrong_obs_X=wrong_x,
                wrong_obs_Z=wrong_z,
                ler_per_cycle_obs_only=per_cycle(p_shot_obs, distance),
                gamma0=RELAY["gamma0"] if kind == "relay" else "",
                low=RELAY["gamma_dist_interval"][0] if kind == "relay" else "",
                high=RELAY["gamma_dist_interval"][1] if kind == "relay" else "",
                num_sets=RELAY["num_sets"] if kind == "relay" else "",
                seconds=round(time.time() - start, 1),
            )
            if writer is not None:
                writer.writerow(row)
                file.flush()
            print(
                f"{label:15s} p={p:<6} d={distance} shots(X={shots_x}, Z={shots_z}) "
                f"LER/cycle={row['ler_per_cycle']:.3e} errs(X={errs_x}, Z={errs_z}) "
                f"unconverged(X={nonconv_x}, Z={nonconv_z}) "
                f"observables only LER/cycle={row['ler_per_cycle_obs_only']:.3e} "
                f"[{row['seconds']} s]"
            )

    if file is not None:
        file.close()
        print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
