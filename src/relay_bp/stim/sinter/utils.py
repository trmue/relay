# (C) Copyright IBM 2025
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

import pathlib
import sinter


def write_stats(stats: list[sinter.TaskStats], file_path: pathlib.Path | str):
    if isinstance(file_path, str):
        file_path = pathlib.Path(file_path)

    with open(file_path, "w") as f:
        f.write(sinter.CSV_HEADER)
        f.write("\n")
        for stat in stats:
            f.write(stat.to_csv_line())
            f.write("\n")
