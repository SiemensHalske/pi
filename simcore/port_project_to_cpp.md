# 50-Step Roadmap — Total Rewrite of PowderPhysicsEngine to Pure C++20/23

**Target end-state:** one standalone native executable, statically linked where practical, no Python runtime, no pybind11, no hybrid bridge.

---

## Phase 1 — Toolchain, Build Graph, and Core Runtime Skeleton (Steps 1–7)

1. Create a monorepo CMake superproject with targets: `engine_core`, `engine_sim`, `engine_render`, `engine_script`, `engine_app`, `tests`, `benchmarks`.
2. Enforce C++23 (fallback C++20) and hard-fail on unsupported compilers; enable strict warnings (`-Wall -Wextra -Wpedantic -Wconversion`) and treat warnings as errors in CI.
3. Add build profiles `Debug`, `RelWithDebInfo`, `ReleaseLTO`, `ASanUBSan`; wire profile-specific flags (`-march=native` optional, deterministic baseline profile mandatory).
4. Integrate OpenMP via `find_package(OpenMP COMPONENTS CXX)` and link through `OpenMP::OpenMP_CXX`; keep a non-OpenMP serial backend for correctness diffing.
5. Add SIMD feature probing (`AVX2`, `AVX-512F`, `AVX-512DQ`, `AVX-512BW`, `FMA`) at compile-time and runtime; implement dispatch table for kernel variants.
6. Establish deterministic simulation bootstrap: fixed seed RNG, fixed tick rate, fixed substep order, deterministic floating-point mode policy documented.
7. Build an executable shell with CLI config loading (`--profile`, `--headless`, `--benchmark`, `--threads`) and startup telemetry banner (CPU features, thread count, dt, grid size).

---

## Phase 2 — Data-Oriented Memory System and Math/Utility Core (Steps 8–14)

8. Implement custom allocators: frame arena, persistent arena, and pool allocator; all grid arrays must be 64-byte aligned and padded for SIMD tails.
9. Represent every field as flat 1D SoA buffers (`index = y * pitch + x`), with `pitch` rounded up to SIMD width and cache-line boundaries.
10. Define typed field wrappers (`Field2D<float>`, `Field2D<uint8_t>`, `FaceFieldU`, `FaceFieldV`) that expose raw pointers for kernels and bounds metadata.
11. Introduce ghost-cell layers (at least 1, preferably 2) around every PDE field to remove branchy boundary checks from inner loops.
12. Add low-level math primitives: fused finite differences, bilinear samplers, slope limiters, safe reciprocal, and branch-minimized clamp functions.
13. Implement a kernel registry with function pointers/lambdas for backend selection (`scalar`, `omp`, `avx2`, `avx512`) and per-kernel microbenchmark hooks.
14. Add NUMA/thread-affinity controls and first-touch initialization path to reduce remote memory access on multi-socket systems.

---

## Phase 3 — World State, Material DB, and Main Substep Pipeline (Steps 15–20)

15. Define canonical `WorldState` as contiguous SoA blocks: pressure, temperature, enthalpy, density, velocity (MAC), phase fractions, species, stress tensors, debris lists.
16. Convert material definitions to compiled C++ tables (constexpr where possible): EOS params, viscosity, thermal conductivity, heat capacity, yield limits, Arrhenius constants.
17. Implement immutable `SimConfig` + hot-reloadable runtime config overlay with versioning and strict schema validation (JSON or TOML parser in C++ only).
18. Build a hard-ordered substep scheduler (`forces -> advection -> diffusion -> projection -> thermodynamics -> chemistry -> structure -> particles -> boundaries`).
19. Add double/triple-buffer strategy per field group to eliminate write-after-read hazards; prohibit in-place updates unless mathematically valid.
20. Implement checkpoint serializer/deserializer (binary, endian-safe) to support regression replay and cross-version migration tests.

---

## Phase 4 — Cellular/Granular Layer and DEM Coupling Base (Steps 21–26)

21. Rebuild CA occupancy/material id grid as bit-packed + byte-coded SoA layers for fast neighborhood reads and branch-light rule application.
22. Implement granular transport rules (sand/powder/slurry) as tiled kernels over occupancy + local velocity, using deterministic conflict resolution.
23. Add DEM particle store (SoA arrays for position, velocity, angular velocity, radius, mass, inertia, material id) with free-list recycling.
24. Implement broadphase spatial hash on fixed grid buckets, then narrow-phase contact with penalty spring-dashpot and Coulomb friction.
25. Add two-way particle-grid coupling interface: P2G momentum deposition and G2P velocity sampling with conservative mass/momentum accounting.
26. Introduce OpenMP-safe coupling strategy: thread-local accumulators for P2G, followed by deterministic reduction pass (avoid atomics in hot loops where possible).

---

## Phase 5 — Fluid PDE Stack, Multigrid, and AVX-512/OpenMP Optimization (Steps 27–34)

27. Implement MAC grid topology: pressure/scalars at cell centers, `u` at x-faces, `v` at y-faces; maintain explicit divergence and gradient operators on staggered locations.
28. Add Semi-Lagrangian advection for velocity/scalars, then BFECC/MacCormack correction with limiter to prevent overshoot near discontinuities.
29. Implement force accumulation (gravity, buoyancy, user impulses, drag) as separate kernels writing to intermediate face-velocity fields.
30. Build Poisson projection path: compute divergence RHS, solve `∇²p=b`, subtract pressure gradient from face velocities, then enforce boundary constraints.
31. Implement baseline Jacobi and Red-Black Gauss-Seidel solvers for reference; add residual norm tracking and adaptive iteration stop criteria.
32. Implement geometric multigrid V-cycle (restriction/prolongation/smoother choices documented) and use it as default pressure solver.
33. Write AVX-512 kernels for stencil operations (`laplacian`, `jacobi`, `residual`, `axpy`) using aligned loads/stores, masked tails, and prefetch hints only where measured beneficial.
34. Parallelize PDE sweeps with OpenMP (`collapse(2)` or tiled loops), explicit barriers only between dependency stages, and false-sharing avoidance via tile-private scratch.

---

## Phase 6 — Thermodynamics, Phase Change, Structural Mechanics, and Chemistry (Steps 35–42)

35. Implement heat equation with implicit or semi-implicit discretization (ADI or Crank-Nicolson) using material-dependent conductivity and heat capacity.
36. Add enthalpy-porosity phase model: latent heat handling, melt fraction field, and viscosity ramp in mushy zone for solidification/melting stability.
37. Add vaporization/condensation and Leidenfrost regime switch with temperature-dependent film coefficient model and mass/energy conservation checks.
38. Implement elastoplastic structural solver on solid cells: strain-rate update, Cauchy stress tensor evolution, von-Mises yield projection, damage accumulation.
39. Add hyperelastic option (Neo-Hookean) for large deformation materials and thermomechanical coupling via temperature-dependent constitutive parameters.
40. Implement stiff chemistry integrator per cell: operator-split transport/reaction, implicit Newton iterations for Arrhenius source terms, bounded species updates.
41. Add EDC-style turbulence-chemistry limiter and stoichiometric mixing constraints across species fields (`O2`, fuel vapor, products, soot precursors).
42. Integrate pyrolysis + soot + electrolyte submodels into unified reaction source API with per-material reaction pathways and tabulated parameters.

---

## Phase 7 — Rendering Runtime (SDL2/GLFW), Scripting (Lua), Validation, and Release (Steps 43–50)

43. Choose runtime shell (`GLFW` or `SDL2`) and implement platform layer: window, input, timing, swapchain/context init, headless mode compatibility.
44. Build renderer abstraction with two backends: immediate OpenGL debug renderer and optional Vulkan path; simulation buffers remain backend-agnostic.
45. Implement GPU upload strategy from SoA fields to textures/SSBOs using staging buffers and frame pacing to avoid sim-thread stalls.
46. Recreate in-game console natively in C++: command registry, typed arguments, reflection-free dispatch, and command history persistence.
47. Embed Lua (sol2 or equivalent) with strict sandbox: controlled API surface, per-script memory caps, execution time budget, and deterministic tick hooks.
48. Expose script API points by stage (`pre_step`, `post_advection`, `post_projection`, `pre_render`, `post_render`) with immutable/mutable access contracts.
49. Add validation matrix: unit tests for kernels, solver convergence tests, conservation checks, replay determinism tests, and perf regression benchmarks.
50. Produce release artifact: static/native packaged executable, reproducible build script, CI pipeline (build+test+bench), and migration-complete declaration removing Python codepaths.

---

## Non-Negotiable Engineering Constraints

- No Python interpreter, no AST bridge, no pybind11 bridge, no mixed runtime.
- Inner loops must be vectorization-friendly: contiguous memory, predictable strides, minimal branches, explicit boundary strip handling.
- Thread safety in coupling stages must be designed first, optimized second; determinism mode must remain available.
- Every physics subsystem must provide conserved-quantity diagnostics (mass, momentum, energy) and residual/error metrics.
