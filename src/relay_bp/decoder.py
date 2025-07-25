# (C) Copyright IBM 2025
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
"""Decoder imports from rust bindings."""

from __future__ import annotations

__all__ = [
    "DecodeResult",
]

from _relay_bp import _decoder  # pylint: disable=E0611
from _relay_bp._decoder import DecodeResult
