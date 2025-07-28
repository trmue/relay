# (C) Copyright IBM 2025
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

import re
from pathlib import Path

import stim


testdata_path = Path(__file__).parent


def filter_file(file_name: Path, search_keys: dict[str, str]) -> bool:
    file_str = file_name.stem
    for key, value in search_keys.items():
        if value == "*":
            continue
        match_str = re.escape(f"{key}={value}")
        regex_str = f"{match_str}(?=(,|$))"
        if not re.search(regex_str, file_str):
            return False
    return True


def get_stored_circuits_paths(
    search_keys: dict[str, str], search_path: Path = testdata_path
) -> list[Path]:
    """Search for paths to valid matching stored test circuits."""
    search_path = search_path.resolve()

    files = [p for p in search_path.rglob("*") if p.is_file()]
    return [file for file in files if filter_file(file, search_keys)]


def _build_search_keys(
    circuit: str,
    observables: str,
    distance: int | str,
    rounds: int | str,
    error_rate: float,
    noise_model: str,
    **search_keys,
):
    return {
        "circuit": circuit,
        "observables": observables,
        "distance": distance,
        "rounds": rounds,
        "error_rate": (
            f"{error_rate:.6f}".rstrip("0")
            if isinstance(error_rate, float)
            else error_rate
        ),
        "noise_model": noise_model,
        **search_keys,
    }


def get_test_circuit(
    circuit: str,
    error_rate: float,
    distance: int | str = "*",
    rounds: int | str = "*",
    observables: str = "*",
    noise_model: str = "*",
    search_path: Path = testdata_path,
    **search_keys,
) -> stim.Circuit:
    """Get the matching test circuit.

    "*" value is treated as a wildcard.

    Raises an error if there are multiple.
    """
    search_keys = _build_search_keys(
        circuit, observables, distance, rounds, error_rate, noise_model, **search_keys
    )

    circuits = get_stored_circuits_paths(search_keys, search_path=search_path)

    if not circuits:
        raise ValueError(f"No matching circuit found for search keys: {search_keys}.")
    elif len(circuits) > 1:
        raise ValueError(
            f"More than one matching circuit found for search keys: {search_keys}. Try providing additional filters."
        )

    return stim.Circuit.from_file(circuits[0])


def get_all_test_circuits(
    circuit: str,
    error_rate: float,
    distance: int | str = "*",
    rounds: int | str = "*",
    observables: str = "*",
    noise_model: str = "*",
    search_path: Path = testdata_path,
    **search_keys,
) -> dict[str, stim.Circuit]:
    """Get all matching test circuits for provided filters.

    "*" value is treated as a wildcard.
    """
    search_keys = _build_search_keys(
        circuit, observables, distance, rounds, error_rate, noise_model, **search_keys
    )

    circuits = get_stored_circuits_paths(search_keys, search_path=search_path)

    circuit_dict = {}
    for circuit_ in circuits:
        circuit_dict[circuit_.stem] = stim.Circuit.from_file(circuit_)

    return circuit_dict
