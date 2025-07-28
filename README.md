<div align="center">

# Relay-BP decoder

A rust implementation of the [Relay-BP(arXiv:2506.01779)](https://arxiv.org/abs/2506.01779)
decoder for qLDPC codes with python bindings.

[![Licensed under the Apache 2.0 open-source license](https://img.shields.io/badge/License-Apache%202.0-3c60b1.svg?logo=opensourceinitiative\&logoColor=white\&style=flat-square)](https://github.com/trmue/relay/blob/main/LICENSE)
![Rust](https://img.shields.io/badge/Rust-000000?logo=rust&logoColor=white)
![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)

[Installation](#installation) &ndash;
[Usage](#usage) &ndash;
[Paper](https://arxiv.org/abs/2506.01779) &ndash;
[Help](#help) &ndash;
[Citation](#citation) &ndash;

</div>

**Relay-BP** is an enhanced belief propagation (BP) decoder designed to overcome the limitations of standard BP when decoding quantum low-density parity-check (qLDPC) codes. It introduces three key innovations:

- *Disordered Memory Strengths* – Breaks trapping sets by diversifying memory dynamics.
- *Ensembling* – Explores a broader space of valid corrections through multiple decoding attempts.
- *Relaying* – Shares ensemble posteriors to accelerate convergence toward nearby alternative solutions.

**🔬 Performance Highlights**

Relay-BP achieves order-of-magnitude improvements in decoding performance for bivariate bicycle codes, including the Gross and Two-Gross codes. It also performs strongly on the rotated surface code, all while maintaining high throughput and hardware compatibility—especially when compared to general-purpose decoders like BP-OSD.

**✅ Tested On**
- Bivariate Bicycle (BB) Codes
- Logical Operations on/between BB code modules ([arXiv:2506.03094](https://arxiv.org/abs/2506.03094))
- Rotated Surface Codes

**💬 Get Involved**
We’re actively exploring Relay-BP and its potential extensions. If you're interested in contributing, experimenting, or just curious, feel free to:
- Open an issue for discussion
- Reach out to the authors directly

## Installation
- [Install rust](https://www.rust-lang.org/tools/install)
- (Optional) Setup a Python virtual environment and activate it:
    - `pyenv install 3.12 --keep`
    - `pyenv virtualenv 3.12 relay_bp`
    - `pyenv activate relay_bp`
- `pip install ".[stim]"` ([stim](https://github.com/quantumlib/Stim) support is optional but helpful for testing.)

## Usage
The Relay-BP package supports decoding support for decoding by setting up
the decoding problem manually by specifying a check matrix, (optional) observable matrix, and, error priors.

Alternatively, optional support for Stim+Sinter is provided to enable easy integration with other Stim based efforts.

Below we give a quick example for both the Python and Sinter APIs. However, we recommend
seeing the getting started [notebook](./examples/GettingStarted.ipynb) for a more detailed
overview of how to use the decoder.

### Python

Below is a simple program that demonstrates how to run Relay for a repetition code with the Python API:

```python
import relay_bp
import numpy as np
from scipy.sparse import csr_matrix

check_matrix = csr_matrix(np.asarray([
    [1, 1, 0],
    [0, 1, 1],
])) # detectors x error variables

decoder = relay_bp.RelayDecoderF32(
    check_matrix,
    error_priors=np.array([0.003, 0.003, 0.003], dtype=np.float64), # Set the priors probability for each error
    gamma0=0.65, # Uniform memory weight for the first ensemble
    pre_iter=80, # Max BP iterations for the first ensemble
    num_sets=100, # Number of relay ensemble elements
    set_max_iter=60, # Max BP iterations per relay ensemble
    gamma_dist_interval=(-0.24, 0.66), # Set the uniform distribution range for disordered memory weight selection
    stop_nconv=5, # Number of relay solutions to find before stopping (the best will be selected)
)

detectors = np.array([1, 1], dtype=np.uint8)

decoding = decoder.decode(detectors)

print(f"For the detectors {detectors} the decoded errors were {decoding}")
```

Will yield output:
```bash
>>> For the detectors [1 1] the decoded errors were [0 1 0]
```

If you want to decode a batch of detectors you may use `decode_batch` or `par_decode_batch`
to decode the batch in parallel.

```python
detector_batch = np.array([[0, 1], [1, 0], [1, 1]], dtype=np.uint8)

decoding = decoder.decode_batch(detector_batch)
print(decoding)
```

Giving:
```
[[0 0 1]
 [1 0 0]
 [0 1 0]]
```


### Stim + Sinter
The Relay-BP package includes optional integration with [`stim`](https://github.com/quantumlib/Stim) and [`sinter`](https://github.com/quantumlib/Stim/tree/main/glue/sample)
for constructing and running decoding experiments. Install with `pip install ".[stim]"`.

We include reference circuits for a variety of codes in [`testdata/`](./src/relay_bp/stim/testdata/) along with utility method to load these.
Below is an example of decoding the gross code with Stim + Sinter.

```python
import sinter
import multiprocessing
from relay_bp.stim import sinter_decoders
from relay_bp.stim.testdata import get_test_circuit, filter_detectors_by_basis

num_workers = multiprocessing.cpu_count()

circuit = "bicycle_bivariate_144_12_12_memory_Z"
distance = 12
rounds = 12
shots = 10000
error_rate = 0.003
XYZ_decoding = False

decoders = sinter_decoders(
    gamma0=0.1,
    pre_iter=80,
    num_sets=60,
    set_max_iter=60,
    gamma_dist_interval=(-0.24, 0.66),
    stop_nconv=1,
)

circuit = get_test_circuit(circuit=circuit, distance=distance, rounds=rounds, error_rate=error_rate)
if not XYZ_decoding:
    circuit = filter_detectors_by_basis(circuit, "Z")

task = sinter.Task(
    circuit=circuit,
    decoder="relay-bp",
    collection_options=sinter.CollectionOptions(max_shots=shots)
)

# Collect the samples (takes a few minutes).
samples = sinter.collect(
        tasks=[task],
        num_workers=num_workers,
        custom_decoders=decoders,
   )

task_output = samples[0]
print(f"Sinter execution - shots: {task_output.shots}, errors: {task_output.errors}, logical error rate: {task_output.errors/task_output.shots}")
```

Which should yield:

```bash
Sinter execution - shots: 10000, errors: 13, logical error rate: 0.0013
```

### Performance
Relay-BP is a highly configurable algorithm and may be tuned to tradeoff decoding for runtime performance.
As we describe in the [paper](https://arxiv.org/abs/2506.01779) Relay-BP decoding performance is sensitive
to the selection of the selected disordered memory weight (gamma) distribution. Empirically, this distribution is different for every decoding
graph and must be tuned. It is provided to the implementation in the form of an input distribution parameter - `gamma_dist_interval`.
Relatively good defaults are set, although tuning can yield at least an order of magnitude improvement. Similarly, the uniform memory
parameter `gamma0` must also be tuned.

Relay-BP may run until a configurable number of solutions have been found and enables the algorithm to explore a larger solution space for better quality solutions - this is controlled by the parameter `stop_nconv`.
It should be kept in mind that a better `gamma_dist_interval` will result in the number of solutions requested by `stop_nconv` to be met after fewer BP iterations.
As a result, specifying good gammas is critical to achieving good runtime performance.
In the worst-case where no solutions are found the algorithm may run for `pre_iter + num_sets*set_max_iter` BP iterations before exiting.
This can yield very poor runtime performance.

Below we summarize the parameters impacting decoding and runtime performance:
- `alpha`: Enable a check to [variable message scaling factor](https://ieeexplore.ieee.org/document/6940497) with an iteration-dependent exponential ramp. Improves BP convergence. The default is normally fine.
- `alpha_iteration_scaling_factor`: Provide the alpha scaling constant. The default is normally fine.
- `gamma0`: Set the ordered memory parameter. A default around 0.1 is normally good, but should be tuned as in the code analysis [notebooks](./examples/).
- `pre_iter`: Set the maximum number of BP iterations for the first ensemble iteration in which `gamma0` will be used. Increasing this value may improve decoding performance at the expense of runtime.
- `num_sets`: The number of Relay-BP ensembles to run. More ensembles will help improve decoding performance. However, runtime performance is very sensitive to this parameter.
- `set_max_iter`: The number of iteration to run per-ensemble. Similar to `pre_iter`.
- `gamma_dist_interval`: The uniform distribution range to select uniformly random memory weights over. The performance of Relay-BP is highly sensitive to this value and should be tuned as in the code analysis [notebooks](./examples/).
- `explicit_gammas`: Instead of selecting gammas from a uniform distribution they may be specified for each error variable. This is a 2D array of `np.float64` of shape `(num_sets, num_errors)`.
- `stop_nconv`: How many Relay ensemble solutions to find before terminating. Setting this value greater than one can better explore the solution space to improve decoding performance at the cost of running more ensemble elements (up to the max of `num_sets`).

While we make no direct claims about the performance of this implementation, it performs relatively well due to its Rust implementation which is backed by an efficient sparse array message-passing data structure with a contiguous memory layout.

#### Sinter
When running with Sinter we have noticed a relatively significant performance difference between using the packages builtin parallelism with `parallel=True` and Sinter's multiprocessing approach
to task-scheduled parallelism. For large simulations it may be important to consider a different parallelism/batching strategy with Sinter but we have not explored this in detail.


### Additional comments/features
- Stim test circuits are available in [testdata](./src/relay_bp/stim/testdata/) and may be fetched using the helper functions `relay_bp.stim.testdata.circuits.get_test_circuit` and `relay_bp.stim.testdata.circuits.get_all_test_circuits`.
- The `ObservableDecoderRunner` supports batch decoding in parallel with a progress bar as `observable_decoder.decode_batch(syndromes, parallel=True, progress_bar=False)`.
- Detailed execution information may be reported with `decode_detailed` and `decode_detailed_batch` returning a `DecodeResult`.
- A variety of Relay-BP implementation data types are available such as `RelayDecoderF32/RelayDecoderF64/RelayDecoderI32/RelayDecoderI64`. These may be further limited in their messaging precision using the input parameters `data_scale_value` and `max_data_value` to effectively explore a range of non-native precisions such as `I5`.
    - Separately a fixed point implementation is available as `MinSumBPDecoderFixed` using the Rust [fixed-point](https://docs.rs/fixed/latest/fixed/) number crate.
- The core MinSum BP implementation of Relay-BP is available as `relay_bp.MinSumBPDecoder<Type>`.

## Development

### Testing
- Python: `pytest tests/`
- Rust: `cargo test`

#### Static checks

If you are developing the package we recommend you install pre-commit and other related development tools in addition to the above:
```bash
poetry install
pre-commit install
```

This will ensure you are only ever committing code that passes the pre-commit checks.
You may manually run these checks before committing with:

```bash
pre-commit run --all-files
```

### Benchmarking
There are benchmarks bundled with the library
To run the benchmark it is as simple as
- `cargo bench`

**Profiling with Instruments**
It is also quick and easy to benchmark with Instruments on OSX.
- First install the [Cargo Instruments](https://crates.io/crates/cargo-instruments) crate with `cargo install --git https://github.com/reilabs/cargo-instruments.git --branch support-tests` (using the specific branch to enable test support)
- Then profile a specific benchmark with `cargo instruments --template time --test relay_bp --profile release -- bp::relay::tests::decode_144_12_12 --exact`. This will automatically launch the Instruments GUI when complete.
    - See the documentation of [Cargo Instruments](https://crates.io/crates/cargo-instruments) for more info on usage.
    - Make sure to click the run tab to see the output.
- You will need to make sure that debug symbols are available. Add them to the `Cargo.toml` temporarily with
    - ```toml
        [profile.release]
        debug = 1
      ```
## Python wrapper

### Development with Maturin
The project's [pyo3](https://pyo3.rs/main/doc/pyo3/) bindings are built with [maturin](https://www.maturin.rs/).
A simple way to do iterative development is to use Maturin directly:
- `pip install maturin`
- `maturin develop`

This will automatically build the relevant Rust crates, bindings and install to your activated Python installation

## Help

Relay-BP is not an IBM product. However, if you have any questions/concerns or want to report a bug/feature
please create an [issue](https://github.com/trmue/relay/issues). If you want to contribute code back to the project
please make a [pull request](https://github.com/trmue/relay/pulls).

## Citation

If you are publishing a paper or writing about the Relay-BP decoder please cite:

```
@misc{mullerImprovedBeliefPropagation2025,
  title = {Improved Belief Propagation Is Sufficient for Real-Time Decoding of Quantum Memory},
  author = {M{\"u}ller, Tristan and Alexander, Thomas and Beverland, Michael E. and B{\"u}hler, Markus and Johnson, Blake R. and Maurer, Thilo and Vandeth, Drew},
  year = {2025},
  month = jun,
  number = {arXiv:2506.01779},
  eprint = {2506.01779},
  primaryclass = {quant-ph},
  publisher = {arXiv},
  doi = {10.48550/arXiv.2506.01779},
  urldate = {2025-06-06},
  archiveprefix = {arXiv},
  keywords = {Quantum Physics},
}
```


## Disclaimer
This software is not a supported IBM product.

(C) Copyright IBM 2025
