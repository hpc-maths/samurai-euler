# Samurai Euler Tutorial

This project demonstrates how to solve the Euler equations using the [Samurai](https://github.com/hpc-maths/samurai) library with adaptive mesh refinement (AMR).

## Prerequisites

You need a package manager like `conda` or `mamba` to install the dependencies.

## Installation

1.  **Create the Conda environment:**

    Use the provided `environment.yml` file to create the environment.

    ```bash
    mamba env create -f conda/environment.yml
    ```

    Or with conda:

    ```bash
    conda env create -f conda/environment.yml
    ```

2.  **Activate the environment:**

    ```bash
    mamba activate samurai-euler-env
    ```

## Building the Project

1.  **Create a build directory:**

    ```bash
    mkdir build
    cd build
    ```

2.  **Configure the project with CMake:**

    ```bash
    cmake .. -DCMAKE_BUILD_TYPE=Release
    ```

3.  **Build the executables:**

    ```bash
    make
    ```

## Running the Tests

### Euler 2D

To run the 2D Euler simulation:

```bash
./euler_2d
```

This will generate output files (e.g., HDF5/XDMF) in the `results` directory (or current directory depending on configuration), which can be visualized using ParaView.

### Other Executables

*   `euler_1d`: 1D Euler simulation.
*   `euler_user_pred_1d`: 1D Euler simulation with user-defined prediction.

## Command Line Options

The `euler_2d` executable accepts several command-line arguments to control the simulation, multiresolution parameters, and output.

### Simulation Parameters

| Option               | Description                                                     | Default                  |
| :------------------- | :-------------------------------------------------------------- | :----------------------- |
| `--cfl`              | The CFL number                                                   | `0.4`                    |
| `--Ti`               | Initial time                                                     | `0.0`                    |
| `--Tf`               | Final time                                                       | `0.25`                   |
| `--scheme`           | Finite volume scheme (`rusanov`, `hll`, `hllc`)                  | `hllc`                   |
| `--test-case`        | Test case to run                                                 | `double_mach_reflection` |
| `--gamma`            | Ratio of specific heats; overrides the value of the test case    | (test case)              |
| `--restart-file`     | Path to a file to restart the simulation from                    | (empty)                  |
| `--check-positivity` | Check positivity of density and pressure at each iteration       | off                      |

Run `./euler_2d --help` for the list of available test cases: it is read from
the registry, so it always matches what the binary actually supports. `euler_3d`
accepts the same options and exposes the cases that are defined in 3D.

### Multiresolution Parameters

| Option        | Description                          | Default |
| :------------ | :----------------------------------- | :------ |
| `--min-level` | Minimum level of the multiresolution | `8`     |
| `--max-level` | Maximum level of the multiresolution | `8`     |

### Output Parameters

| Option       | Description                        | Default                  |
| :----------- | :--------------------------------- | :----------------------- |
| `--path`     | Output directory path              | `results`                |
| `--filename` | Output file name prefix            | `<test-case>_<scheme>`   |
| `--nfiles`   | Number of output files to generate | `1`                      |

### Example Usage

Run the simulation with a specific final time and output 10 files:

```bash
./euler_2d --Tf 0.5 --nfiles 10
```

Run with adaptive mesh refinement (levels 5 to 10):

```bash
./euler_2d --min-level 5 --max-level 10
```

Run a case with a different gas, writing somewhere else:

```bash
./euler_2d --test-case sedov_blast --gamma 1.6666667 --path out --filename sedov_g53
```

## Tests

The suite drives the built binaries as subprocesses, so it checks what a user
actually runs, command line included.

```bash
ctest --test-dir build --output-on-failure       # the fast tests, about 30 s
ctest --test-dir build -L slow                   # the validation runs as well
```

It has three tiers, and they are not interchangeable.

`test_invariants.py` owns no reference file. It asserts properties that stay
true when the numerics legitimately change: a uniform flow stays uniform, a
closed box conserves mass and energy, density and pressure stay positive, the
Sedov blast keeps its rotational symmetry, a restart reproduces the run. A
better scheme cannot make these fail, and no amount of regenerating can make
them pass.

`test_regression.py` compares whole fields against references under
`tests/reference`. Every case there runs on a **uniform** mesh, on purpose: on
an adapted mesh a rounding difference of the order of 1e-16 near the
multiresolution threshold flips a refinement decision, the mesh changes, and the
comparison fails on another compiler without anything being wrong. Regenerate
the references with `pytest --generate-ref`, and say in the commit message why
they moved.

`test_validation.py` is marked slow and asserts on scalars rather than fields,
which is what makes it usable on an adapted mesh. It measures the convergence
order of the isentropic vortex against its exact solution, and checks that
adaptation reaches the same error as a uniform mesh with fewer cells.

## Adding a test case

A test case is a domain, an initial state, a set of boundary conditions and the
gas it is written for. Add a header in `euler/init/`, expose `register_me()`, and
list it in `register_all()` in `euler/init/cases.hpp`. Cases whose definition
does not depend on the dimension (`free_stream`, `sedov_blast`) are templated on
the field and serve `euler_1d`, `euler_2d` and `euler_3d`; the others register
for the dimensions they are written for. The boundary conditions the cases need — outflow, solid wall, an imposed
state — are in `euler/bc.hpp` and work in any dimension.

Give the case the gas it was written for through its `eos` field: monofluid
cases use `EOS::ideal_gas(gamma)`. `euler/eos.hpp` also defines a stiffened gas
for a future two-phase model; the solver is templated on the state law, so the
monofluid path does not pay for the coefficients it never uses.
