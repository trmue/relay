# (C) Copyright IBM 2025
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
try:
    import stim
except ImportError:
    print(f'Could not import stim. Make you installed relay_bp with the "stim" option.')
    raise

from .sinter import (
    CheckMatrices,
    SinterDecoder_MemBP,
    SinterDecoder_MSLBP,
    SinterDecoder_RelayBP,
    sinter_decoders,
)
