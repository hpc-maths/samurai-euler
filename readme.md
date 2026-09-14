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
*   `two_phase_1d`, `two_phase_2d`: the five-equation two-phase model, a
    different system of equations from the three above. See
    [Two-phase flow](#two-phase-flow-the-five-equation-model).

## Command Line Options

The `euler_2d` executable accepts several command-line arguments to control the simulation, multiresolution parameters, and output.

### Simulation Parameters

| Option               | Description                                                     | Default                  |
| :------------------- | :-------------------------------------------------------------- | :----------------------- |
| `--cfl`              | The CFL number                                                   | `0.4`                    |
| `--Ti`               | Initial time                                                     | `0.0`                    |
| `--Tf`               | Final time                                                       | `0.25`                   |
| `--scheme`           | Riemann solver (`rusanov`, `hll`, `hllc`)                        | `hllc`                   |
| `--order`            | Order in space: `1` on cell averages, `2` on a MUSCL reconstruction | `1`                   |
| `--slope-limiter`    | Slope limiter of the reconstruction (`minmod`, `vanleer`, `moncen`, `none`) | `moncen`      |
| `--time-integrator`  | Time stepping (`auto`, `euler`, `ssprk2`, `strang`)              | `auto`                   |
| `--test-case`        | Test case to run                                                 | `double_mach_reflection` |
| `--gamma`            | Ratio of specific heats; overrides the value of the test case    | (test case)              |
| `--restart-file`     | Path to a file to restart the simulation from                    | (empty)                  |
| `--check-positivity` | Check positivity of density and pressure at each iteration       | off                      |

Run `./euler_2d --help` for the list of available test cases: it is read from
the registry, so it always matches what the binary actually supports. `euler_3d`
accepts the same options and exposes the cases that are defined in 3D.

### Test case parameters

A case that comes in variants declares its own option, and `--help` lists them
under *Test case parameters*. There are two today, both belonging to `lax_liu`:

| Option                | Description                                             | Default |
| :-------------------- | :------------------------------------------------------ | :------ |
| `--riemann-config`    | Lax & Liu configuration, 1 to 19                         | `3`     |
| `--riemann-interface` | Where the quadrants meet, in each direction              | `0.8`   |

`lax_liu` is the four-quadrant Riemann problem in two dimensions and the eight
octant one in three. In 3D only configuration 3 exists, the one the article
extends, and `--riemann-config` there accepts nothing else.

The interface position is not part of the classification. Lax & Liu and
Kurganov & Tadmor put the four quadrants of the unit square at its centre; the
default here is 0.8, which is what the article uses for configuration 3 so that
the waves fill the domain by `t_f = 0.8` without reaching the boundary. It
suits configurations 3 and 4 and misleads for the other seventeen, whose
structure is born 0.2 from a corner and reaches it early: pass
`--riemann-interface 0.5` to compare those against their published figures.

The options of *every* case are declared, not only those of the selected one:
which case runs is itself decided by the parse. Two cases must therefore not ask
for the same option name, and naming the option after the case is what keeps
them apart.

### Order and time stepping

`--scheme` and `--order` are independent: the first says which Riemann solver
settles an interface, the second what states it is handed. At order 1 those are
the two cell averages; at order 2 they are values reconstructed on the face
from a limited slope, on a stencil of four cells.

`--time-integrator` says how the step is taken.

| Value    | What it does                                                                    |
| :------- | :------------------------------------------------------------------------------ |
| `euler`  | One explicit step. At order 2 the flux also advances the face values half a step, which makes it MUSCL-Hancock. |
| `ssprk2` | Two stages averaged (Heun).                                                      |
| `strang` | One directional sweep at a time, X(dt/2) Y(dt) X(dt/2) in two dimensions.         |
| `auto`   | Explicit Euler at order 1, Strang at order 2.                                     |

The three are not interchangeable at order 2 in more than one dimension. The
Hancock predictor advances a face value with the equations taken normal to that
face, and the transverse terms it leaves out are of the same order as the ones
it keeps, so unsplit it converges at 1.05 on the isentropic vortex. Under a
directional sweep there is no transverse direction and the same predictor is
exact, which is why Strang reaches 2.23 there, and SSP-RK2 2.13. In one
dimension Strang is the single Hancock step, to the bit.

`--slope-limiter none` is unlimited and oscillates at a shock. It is there to
measure an order on a smooth solution, where every limiter clips the extremum
and costs a fraction of an order that says nothing about the scheme.

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
| `--metrics-file` | Write the performance metrics of the run, as JSON, to this file | (none) |

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

Run the reference case of the article, configuration 3 of Lax & Liu, to its
final time on an adapted mesh:

```bash
./euler_2d --test-case lax_liu --riemann-config 3 --min-level 4 --max-level 10 --Tf 0.8 --order 2
```

Run configuration 5 as Kurganov & Tadmor publish it, quadrants meeting at the
centre:

```bash
./euler_2d --test-case lax_liu --riemann-config 5 --riemann-interface 0.5 --Tf 0.3
```

## Performance

The article this repository reproduces is first of all a performance paper, and
its tables are made of three numbers. Every run reports them:

```
performance
  cells            17524 -> 151480 of 262144 uniform
  sparsity index   6.68% -> 57.79%
  cell updates     262193561 over 2672 time steps
  time to solution 102.92 s, 28.4% of it adapting the mesh
  throughput       2.55 Mcu/s
```

| Metric | What it is |
| :----- | :--------- |
| sparsity index | cells of the adapted mesh over the cells of the uniform mesh at `--max-level`, in percent. 100% is a mesh that never coarsened. Given at the initial and at the final time, as the article gives it: a Riemann problem fills its mesh up as the waves spread, and one number taken at one end would flatter or damn it. |
| cell updates, Mcu/s | one cell advanced by one time step is one cell update; the throughput is millions of those per second over the whole run. Counted once per cell per step whatever the integrator does inside, so that the `2*dim - 1` sweeps of Strang do not read as more work done. |
| time to solution | the time loop. The mesh adaptation is part of it and is reported apart; writing files is not, `--nfiles` being a choice of whoever runs the solver. |

`--metrics-file <name>` writes the same numbers as JSON, which is what
`python/performance.py` builds a table out of:

```bash
python python/performance.py --levels 6 7 8 9 --uniform
```

One run per resolution, on the reference case of the article — configuration 3
of Lax & Liu, to `t_f = 0.8`, second order — gives, on one core:

```
 l_min  l_max   resolution    mr-eps    Mcu/s  time (s)    AMR           cells ti/tf    sparsity ti/tf
     3      6         64^2   default      2.2      0.39  25.1%           1720 / 3784     42.0% / 92.4%
     3      7        128^2   default      2.4      2.49  29.6%          3916 / 13522     23.9% / 82.5%
     3      8        256^2   default      2.5     15.41  28.1%          8416 / 45394     12.8% / 69.3%
     3      9        512^2   default      2.5    102.92  28.4%        17524 / 151480      6.7% / 57.8%
     9      9        512^2   default      4.7    147.64   0.0%       262144 / 262144   100.0% / 100.0%
```

The cell counts are reproducible to the cell; the times are wall clock on one
core and move by ten percent or so between runs, which is worth remembering
before reading anything into a small difference.

The last row is the uniform mesh at the same resolution, which is the reference
the article puts at the bottom of its own table. Reading the two bottom rows
together is the whole point of the exercise: **the adapted run does 2.7 times
fewer cell updates and is 1.4 times faster**, because it runs at a little more
than half the throughput of the uniform one. Roughly a third of what is lost is
the adaptation itself, at 28% of the time to solution; the rest is what an
adapted mesh costs per cell — level interfaces, prediction, intervals that are
shorter than a uniform row.

Comparing that with the table of the article takes some care, and the sparsity
column is where it goes wrong most easily:

- **Equivalent resolution is the only fair pairing.** The article varies the
  number of cells per octree leaf at a fixed equivalent resolution of 4096²;
  samurai carries one cell per leaf, so that axis does not exist here and the
  table above sweeps the resolution instead. A sparsity index quoted without the
  max-level it was measured at compares nothing: what the adaptation keeps is a
  neighbourhood of the discontinuities, which are curves in a plane, so their
  share of the mesh falls as the resolution rises — 92%, 83%, 69%, 58% over the
  four rows above.

- **The refinement criterion is not the same one**, and this is the real
  difference between the two codes rather than a defect of either. The article
  refines on a Löhner criterion, a normalised second difference thresholded at
  `r_refine = 0.4`; samurai refines on the details of the multiresolution
  thresholded at `--mr-eps`. The multiresolution comes with an error estimate
  that the gradient criterion has not, and it keeps more cells for it. The
  threshold is the knob that trades the two against each other, and `--mr-eps`
  sweeps it:

```bash
python python/performance.py --levels 9 --mr-eps 1e-4 1e-3 1e-2
```

```
 l_min  l_max   resolution    mr-eps    Mcu/s  time (s)    AMR           cells ti/tf    sparsity ti/tf
     3      9        512^2     1e-04      2.3    115.04  28.5%        17524 / 151480      6.7% / 57.8%
     3      9        512^2     1e-03      1.8     69.19  38.9%         17524 / 72616      6.7% / 27.7%
     3      9        512^2     1e-02      1.5     58.86  43.6%         17524 / 42139      6.7% / 16.1%
```

  Two decades of threshold take the final sparsity from 58% to 16%, which is the
  order of magnitude the article reports, and the time to solution from 115 s to
  59 s. The initial mesh does not move at all: the details of a piecewise
  constant state are of order one at the discontinuities and far above every
  threshold in this range, so the three runs start from the same cells and part
  company as the solution develops structure. **These rows say nothing about
  accuracy**, and a large enough threshold makes any mesh sparse and any
  solution wrong; the error of an adapted run against a uniform one is what
  `python/error_analysis.py` measures, on the cases that have an exact solution.

- **The AMR share is not measured over the same cadence.** The article runs its
  AMR cycle once every 10 time steps and this solver adapts at every one, which
  is most of the distance between the 28% above and the few percent it reports
  on CPU at its nominal block size.

- **Throughput is architecture, not method.** The numbers of the article are
  measured on 72 ARM cores or on a Hopper GPU, against one core here, and its
  solver works on blocks of 16² cells where this one works on intervals. The
  column worth comparing is the sparsity index; the Mcu/s column is worth
  comparing against *itself*, between two runs of this solver.

## Two-phase flow: the five-equation model

`two_phase_1d` and `two_phase_2d` solve a different system from the three
binaries above: two compressible fluids sharing the mesh, in pressure and
velocity equilibrium, with an interface between them.

```
d_t (alpha_i rho_i) + div(alpha_i rho_i u) = 0      i = 0, 1
d_t (rho u)         + div(rho u @ u + p I) = 0
d_t E               + div((E + p) u)       = 0
d_t alpha_0         + u . grad(alpha_0)    = 0
```

so `dim + 4` components per cell, against `dim + 2` for the monofluid solver.
Each phase is a stiffened gas, and the mixture behaves as one whose coefficients
depend on the volume fraction:

```
1 / (gamma_m - 1) = sum_i alpha_i / (gamma_i - 1)
gamma_m pi_m / (gamma_m - 1) = sum_i alpha_i gamma_i pi_i / (gamma_i - 1)
```

Those two sums are the whole equation of state, and the fact that both are
**linear in alpha** is what makes the model work: it is exactly what lets a
uniform pressure survive the averaging of two fluids in one cell.

### What is different in the scheme

The first four equations are conservation laws and go through the Riemann solver
the monofluid ones do, with the mixture sound speed and the two partial
densities carried through the contact. The fifth is not a conservation law, and
is discretized with the contact velocity `u*` the Riemann solver already
computes:

```
alpha_i^{n+1} = alpha_i - dt/dx [ (u* alpha*)_{i+1/2} - (u* alpha*)_{i-1/2}
                                  - alpha_i (u*_{i+1/2} - u*_{i-1/2}) ]
```

The term in `alpha_i` is the cell's own volume fraction, so the two cells an
interface separates receive different contributions from it. samurai calls that
a non-conservative flux and takes a pair of values per face, which is what the
two-phase scheme returns; the four conservation laws take the conservative pair
and are conserved to the last bit — the test suite holds them to 1e-12.

`--scheme`, `--order`, `--slope-limiter` and `--time-integrator` mean what they
mean for the monofluid solver, and the Hancock predictor is the same one with
two more equations.

### Test cases

| Case | What it is |
| :--- | :--------- |
| `water_air_shock_tube` | Section 6.1.1 of the article: water at 1e9 Pa against air at 1e5, tube [-2, 2], diaphragm at 0.7, `t_f = 9e-4`. In 2D the domain is [-2,2] x [-0.4,0.4], where level 6 is exactly the 320 x 64 equivalent resolution the article quotes. |
| `triple_point` | Section 6.1.2, as the article poses it: two gases, `gamma = 1.5` in regions 1 and 3 and 1.4 in region 2, [0,7] x [0,3], `t_f = 2.0`. `triple_point_single_gamma` in the monofluid solver is the same geometry with one gas and says at length that it is not this case. |
| `advected_interface` | A slab of water in air, everything at one atmosphere and moving at 100 m/s. The exact solution is the initial state translated, and a scheme that advects the volume fraction inconsistently with the masses produces a pressure spike out of nothing. |
| `sod_x_pure` | Sod's tube as a two-phase state that is one fluid everywhere. With `alpha = 1` the model is the Euler system, and the suite holds the two solvers to twelve digits of each other. |
| `shock_bubble` | Section 6.1.3: a Mach 1.22 shock crossing a 25 mm helium bubble in a 445 x 89 mm tube. The one case of the article validated against an experiment — see below. |

```bash
./two_phase_1d --test-case water_air_shock_tube --min-level 10 --max-level 10 --order 2
./two_phase_2d --test-case water_air_shock_tube --min-level 4 --max-level 6 --order 2
./two_phase_2d --test-case triple_point --min-level 4 --max-level 8 --Tf 2.0 --order 2
```

A run writes the volume fraction, the mixture density and pressure, the velocity
**and the two partial densities**: `alpha * rho` is not the mass of either phase
in a mixed cell, and anything checking conservation needs the real ones.

### What it is held to

`python/exact_two_phase_riemann.py` solves the two-material stiffened-gas
Riemann problem exactly, which is what the shock tube is measured against rather
than a figure read by eye: `p* = 4.796906e5 Pa`, `u* = 491.97 m/s`,
`rho*_water = 800.33`, `rho*_air = 2.758`. At level 10 and second order the
solver reaches an L1 error of 1.5e-3 on the density, puts the contact within one
cell of `x = 1.1428` and the interface on about nine cells.

The interface condition is a separate test and a stricter one: on
`advected_interface` the pressure stays uniform to 1e-11 relative and the
velocity to 1e-14, which is round-off and not a physical smallness.

### Interface sharpening: `--thinc`

The diffuse interface of the five-equation model spreads over about ten cells and
keeps spreading. THINC — Tangent of Hyperbola for INterface Capturing — holds it
on two or three by reconstructing the volume fraction inside a mixed cell as a
hyperbolic tangent instead of a limited straight line:

```
alpha_i(X) = 1/2 [ 1 + tanh( beta (sigma X + x_c) ) ],   X in [0, 1]
```

`sigma` says which way the interface faces, `beta` how steep it is
(`--thinc-beta`, 1.6 by default) and `x_c` where it sits inside the cell. Only
`x_c` is unknown, and it is fixed by requiring the profile to average to the cell
average it came from — in closed form, so conservation is exact rather than
approached.

It replaces the reconstruction of the **volume fraction alone**. Pressure,
velocity and the two partial densities keep their MUSCL slopes, and the total
energy is rebuilt from the sharpened `alpha`; the mixture law being linear in
`alpha`, a uniform pressure survives it exactly, which the interface test still
measures at 1e-11.

The steepness is weighted by the interface normal, `beta_d = beta |n_d| + 0.001`,
so a face the interface runs parallel to is not sharpened at all — that is what
stops the scheme from carving steps into an oblique interface. The normal comes
from the gradient of `psi = alpha^m / (alpha^m + (1-alpha)^m)` with `m = 0.1`, and
since a flux stencil is a line and cannot see across itself, it is computed once
per time step over the whole mesh and read from a field.

Measured on the water-air tube at level 10, second order:

| | interface | contact | L1 on alpha | L1 on rho |
| :--- | :---: | :---: | :---: | :---: |
| diffuse | 9 cells | 1.1426 | 1.8e-3 | 1.53e-3 |
| `--thinc` | **3 cells** | 1.1426 | 6.9e-4 | 1.54e-3 |

The contact does not move, the density and pressure errors do not change, and
the volume fraction is two and a half times more accurate. On the triple point
the mixed cells drop from 1912 to 1147 at the same time. A run with no interface
at all is untouched to the bit, which the suite checks on Sod's tube: the
sharpening keys on the volume fraction, not on steepness, so it does not reach
for a shock.

`--thinc` needs `--order 2`: there is no reconstruction to replace at first
order.

### Against an experiment: the shock-bubble interaction

`shock_bubble` is section 6.1.3, and the only case in the article held against a
laboratory rather than against another code. A Mach 1.22 shock runs down the tube
into a helium bubble; the bubble carries sound at three times the speed of the
air around it, so the refracted wave outruns the incident shock, the bubble caves
in on its upstream side and drives a jet through itself. Five fronts come out of
it, and Haas and Sturtevant measured their speeds in 1987.

```bash
python python/shock_bubble_waves.py --level 8
```

runs the case and measures the five speeds the way the article does: a snapshot
every ten microseconds, each front read along the axis of the tube, a straight
line fitted through its positions. The first three are one measurement rather
than three — the leading pressure front on the centreline *is* the incident shock
while it is right of the bubble, the refracted wave while it is inside it, and
the transmitted wave once it is out — so the segments are separated at the edges
of the bubble and not by eye.

Level 10 is the 5120 x 1024 the article runs; level 8 takes three minutes and
level 9 a quarter of an hour:

| m/s | level 8 | level 9 | article | Haas & Sturtevant |
| :--- | ---: | ---: | ---: | ---: |
| shock | 417.2 | 422.4 ± 1.7 | 423.2 ± 0.6 | 410 ± 41 |
| refracted | 946.6 | 951.6 ± 2.7 | 953 ± 7 | 900 ± 90 |
| transmitted | 382.3 | 381.6 ± 0.2 | 381.2 ± 0.7 | 393 ± 39 |
| downstream | 141.5 | 135.4 ± 1.8 | 141.5 ± 1.9 | 145 ± 15 |
| jet | 221.1 | 225.5 ± 1.4 | 222.9 ± 2.2 | 230 ± 23 |

Every one is inside the experiment's 10% margin at both resolutions, and at level
9 three of the five agree with the article to two parts in a thousand. The shock
reaches the bubble at 58.8 µs against the "about 58" the article quotes — which
is the geometry and the initial state in a single number.

Two things are worth knowing before reading those figures. The initial states are
the article's equation (8), and taken literally they are not a shock: the
pressure ratio is exactly Mach 1.22 but the density behind it is 1.6571 where
Rankine-Hugoniot asks for 1.6295, which would make the front travel at 401 m/s.
The solver settles that in the first microseconds — the discontinuity resolves
into the shock the pressure jump calls for — and 417 is what comes out, which is
also why the article's own 58 µs and its 423 m/s agree with each other and not
with 401. And the speeds are wave speeds, not fine structure: they are already
inside the experimental margin at a sixteenth of the article's resolution, where
the schlieren pictures of its fig. 14 would not be.

### What is not there yet

- **The positivity-preserving multiresolution prediction** of the monofluid
  solver, which keys on the Euler layout. An adapted two-phase run has the
  default prediction plus the admissibility floor, which clamps the volume
  fraction to [0, 1], the partial densities to non-negative and the pressure
  above the vacuum of the mixture.
- **The multiple-bubble case** of section 6.1.4, which is the one above at an
  equivalent 32768² over 3.3e5 time steps: the model is there, the machine is
  not.

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
closed box conserves mass and energy and a periodic one conserves momentum as
well, density and pressure stay positive, the Sedov blast keeps its rotational
symmetry, a restart reproduces the run. A better scheme cannot make these fail,
and no amount of regenerating can make them pass.

`test_regression.py` compares whole fields against references under
`tests/reference`. One test there compares no field against a reference but the
references against each other: three identical files for the three Riemann
solvers mean the entry exercises none of them, which happens when a run is too
short to leave the initial state or too coarse to resolve it. Every case there runs on a **uniform** mesh, on purpose: on
an adapted mesh a rounding difference of the order of 1e-16 near the
multiresolution threshold flips a refinement decision, the mesh changes, and the
comparison fails on another compiler without anything being wrong. Regenerate
the references with `pytest --generate-ref`, and say in the commit message why
they moved.

`test_two_phase.py` covers the five-equation model on all three tiers at once,
the model being new enough that splitting it across the three files would scatter
it: the interface condition and conservation as invariants, three reference files
for the three Riemann solvers, and the water-air tube against its exact solution
as a slow test.

Its field reference is Sod's tube and not the water-air one, for a reason worth
knowing before adding a case of your own. A rarefaction running into a liquid at
a gigapascal leaves, ahead of its analytic head, a foot where the density is
1000 minus something tiny — a number built entirely by cancellation. Two hundred
time steps later one bit of difference in a sum has grown into three parts in ten
thousand there, and a run on another machine no longer matches: compiling the
same source with `-ffp-contract=off` reproduces the CI runner's numbers to the
digit. A uniform mesh is necessary for a field comparison and not sufficient —
the case also has to be one whose answer is not built by cancellation — and the
water-air tube is held instead to its star state, its contact and its conserved
masses, which are.

`test_validation.py` is marked slow and asserts on scalars rather than fields,
which is what makes it usable on an adapted mesh. It measures the convergence
order of the isentropic vortex against its exact solution, checks that
adaptation reaches the same error as a uniform mesh with fewer cells, holds the
two shock tubes to the exact Riemann solution of `python/exact_riemann.py` — L1
errors and the position of each wave — and holds the six Lax & Liu
configurations that are symmetric about the diagonal to that symmetry, which a
single mistyped digit in one quadrant breaks.

## Adding a test case

A test case is a domain, an initial state, a set of boundary conditions and the
gas it is written for. Add a header in `euler/init/`, expose `definition<Field>()`
in a namespace of its own, close the file with

```cpp
REGISTER_TEST_CASE(my_case, test_case::my_case, 2, 3)
```

and add one `#include` to `euler/init/cases.hpp`. There is no list to keep in
sync: the trailing numbers are the dimensions the case is written for, and a
case whose definition does not depend on the dimension (`free_stream`,
`sedov_blast`, `sod_x`) passes all three and serves `euler_1d`, `euler_2d` and
`euler_3d`. The boundary conditions the cases need — outflow, solid wall, an
imposed state — are in `euler/bc.hpp` and work in any dimension; a case with no
boundary at all sets `periodic` instead, as `blast_periodic` does.

A case that takes a parameter of its own gives the registry an `options`
function, which declares the command line option and keeps the variable the
parser writes into. `lax_liu` is the worked example: the parameter is read when
a cell is initialised, which is after the parse, and the values the option
accepts depend on the dimension the case was registered for.

Give the case the gas it was written for through its `eos` field: monofluid
cases use `EOS::ideal_gas(gamma)`. `euler/eos.hpp` also defines a stiffened gas
for a future two-phase model; the solver is templated on the state law, so the
monofluid path does not pay for the coefficients it never uses.
