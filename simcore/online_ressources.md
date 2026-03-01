# Online Resources Used

All timestamps are in UTC (ISO 8601).

1. **NVIDIA GPU Gems — Chapter 38 (Fast Fluid Dynamics Simulation on the GPU)**  
 URL: <https://developer.nvidia.com/gpugems/gpugems/part-vi-beyond-triangles/chapter-38-fast-fluid-dynamics-simulation-gpu>  
 Accessed: 2026-03-01T16:00:35Z  
 Notes: Operator splitting, projection method, Poisson pressure solve, Jacobi baseline, boundary-condition handling.

2. **OpenMP Specifications (OpenMP ARB)**  
 URL: <https://www.openmp.org/specifications/>  
 Accessed: 2026-03-01T16:00:35Z  
 Notes: Current standard status (OpenMP 6.0), references for synchronization, reductions, and threading model choices.

3. **sol2 Documentation (sol 3.2.3)**  
 URL: <https://sol2.readthedocs.io/en/latest/>  
 Accessed: 2026-03-01T16:00:35Z  
 Notes: Lua embedding model, safety toggles, usertypes/functions API considerations.

4. **Intel® Intrinsics Guide**  
 URL: <https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html>  
 Accessed: 2026-03-01T16:00:35Z  
 Notes: AVX-512 intrinsic families, latency/throughput references, instruction-level implementation details.

5. **CMake FindOpenMP Module Documentation**  
 URL: <https://cmake.org/cmake/help/latest/module/FindOpenMP.html>  
 Accessed: 2026-03-01T16:00:35Z  
 Notes: `find_package(OpenMP ...)`, imported targets `OpenMP::OpenMP_<lang>`, per-language detection and flags.

6. **GLFW Documentation**  
 URL: <https://www.glfw.org/documentation.html>  
 Accessed: 2026-03-01T16:00:35Z  
 Notes: Window/context setup baseline and API references for cross-platform runtime shell.

7. **cppreference — `std::aligned_alloc`**  
 URL: <https://en.cppreference.com/w/cpp/memory/c/aligned_alloc>  
 Accessed: 2026-03-01T16:14:21Z  
 Notes: Alignment/size constraints for aligned allocation and deallocation semantics.

8. **Linux man page — `sched_setaffinity(2)`**  
 URL: <https://man7.org/linux/man-pages/man2/sched_setaffinity.2.html>  
 Accessed: 2026-03-01T16:14:21Z  
 Notes: CPU affinity API contract, error handling, and per-thread affinity behavior.

9. **Linux man page — `numa(3)`**  
 URL: <https://man7.org/linux/man-pages/man3/numa.3.html>  
 Accessed: 2026-03-01T16:14:21Z  
 Notes: libnuma runtime API, first-touch policy behavior, and CPU/node binding calls.

10. **Linux man page — `numactl(8)`**  
 URL: <https://man7.org/linux/man-pages/man8/numactl.8.html>  
 Accessed: 2026-03-01T16:14:21Z  
 Notes: NUMA policy operational semantics (`--membind`, `--interleave`, `--localalloc`) used for runtime placement strategy.

11. **TOML Specification v1.0.0**  
 URL: <https://toml.io/en/v1.0.0>  
 Accessed: 2026-03-01T16:19:25Z  
 Notes: Key/value syntax, table scoping, duplicate key invalidation rules for strict overlay parsing.

12. **cppreference — `std::endian`**  
 URL: <https://en.cppreference.com/w/cpp/types/endian>  
 Accessed: 2026-03-01T16:19:25Z  
 Notes: Runtime-independent endian detection for portable binary checkpoint format.

13. **cppreference — `std::byteswap`**  
 URL: <https://en.cppreference.com/w/cpp/numeric/byteswap>  
 Accessed: 2026-03-01T16:19:25Z  
 Notes: C++23 byte order conversion for endian-safe integer serialization.
14. **LIGGGHTS-INL Documentation — gran model hooke**  
 URL: <https://opendemjapan.parallel.jp/LIGGGHTS-INL-DOC/gran_model_hooke.html>  
 Accessed: 2026-03-01T16:26:48Z  
 Notes: Reference contact-force decomposition (spring-dashpot normal/tangential terms), frictional yield truncation, and practical model constraints for DEM contact behavior.

15. **Interactive Computer Graphics — Neighborhood Search (Spatial Hashing)**  
 URL: <https://interactivecomputergraphics.github.io/physics-simulation/examples/neighborhood_search.html>  
 Accessed: 2026-03-01T16:26:48Z  
 Notes: Spatial grid hashing recipe for bounded-radius neighbor queries using cell size equal to support radius.

16. **OpenMP ARB — OpenMP 6.0 Specification Portal**  
 URL: <https://www.openmp.org/specifications/>  
 Accessed: 2026-03-01T16:26:48Z  
 Notes: Updated authoritative reference for OpenMP 6.0/6.0.1 clauses and synchronization/runtime semantics.

17. **Jos Stam — Real-Time Fluid Dynamics for Games (GDC 2003 PDF)**  
 URL: <https://www.dgp.toronto.edu/public_user/stam/reality/Research/pdf/GDC03.pdf>  
 Accessed: 2026-03-01T16:36:29Z  
 Notes: Semi-Lagrangian advection and projection workflow reference for stable fluids implementation.

18. **CS 418 Text — Fluids on a Grid (Staggered/MAC Overview)**  
 URL: <https://cs418.cs.illinois.edu/website/text/grid-fluids.html>  
 Accessed: 2026-03-01T16:36:29Z  
 Notes: Staggered-grid discretization, divergence/pressure formulation, and advection design constraints.

19. **OpenMP API 6.0 Specification (PDF)**  
 URL: <https://www.openmp.org/wp-content/uploads/OpenMP-API-Specification-6-0.pdf>  
 Accessed: 2026-03-01T16:36:29Z  
 Notes: Clause-level guidance for parallel loop scheduling and reduction semantics used in PDE sweeps.

20. **Crank–Nicolson Method (Wikipedia)**  
 URL: <https://en.wikipedia.org/wiki/Crank%E2%80%93Nicolson_method>  
 Accessed: 2026-03-01T16:46:28Z  
 Notes: Semi-implicit diffusion discretization form used for Phase 6 heat equation solve and iterative linear update structure.

21. **Leidenfrost Effect (Wikipedia)**  
 URL: <https://en.wikipedia.org/wiki/Leidenfrost_effect>  
 Accessed: 2026-03-01T16:46:28Z  
 Notes: Film-boiling regime switch rationale and reduced heat-transfer coefficient behavior for high-temperature vapor-film conditions.

22. **Arrhenius Equation (Wikipedia)**  
 URL: <https://en.wikipedia.org/wiki/Arrhenius_equation>  
 Accessed: 2026-03-01T16:46:28Z  
 Notes: Temperature-dependent reaction-rate constant form used in stiff per-cell chemistry source integration.

23. **Von Mises Yield Criterion (Wikipedia)**  
 URL: <https://en.wikipedia.org/wiki/Von_Mises_yield_criterion>  
 Accessed: 2026-03-01T16:46:28Z  
 Notes: Equivalent stress projection target for elastoplastic constitutive update in structural cells.

24. **Neo-Hookean Solid (Wikipedia)**  
 URL: <https://en.wikipedia.org/wiki/Neo-Hookean_solid>  
 Accessed: 2026-03-01T16:46:28Z  
 Notes: Compressible Neo-Hookean stress model used for thermomechanical hyperelastic option under finite deformation gradient approximation.

25. **GLFW Getting Started (Quick Guide)**  
 URL: <https://www.glfw.org/docs/latest/quick_guide.html>  
 Accessed: 2026-03-01T16:59:01Z  
 Notes: Canonical init/create-context/poll/swap event-loop sequence used to shape runtime shell API boundaries.

26. **GLFW Window Guide**  
 URL: <https://www.glfw.org/docs/latest/window_guide.html>  
 Accessed: 2026-03-01T16:59:01Z  
 Notes: Window/framebuffer/swap-control semantics used for headless/windowed shell split and platform-layer responsibilities.

27. **Lua 5.4 Reference Manual (C API, Environments, Protected Calls)**  
 URL: <https://www.lua.org/manual/5.4/manual.html>  
 Accessed: 2026-03-01T16:59:01Z  
 Notes: Host embedding contract (`lua_pcall`, `_ENV`, registry, error/status handling) used to define sandbox access and hook execution constraints.

28. **SDL3 API Category Index**  
 URL: <https://wiki.libsdl.org/SDL3/CategoryAPI>  
 Accessed: 2026-03-01T16:59:01Z  
 Notes: Surveyed as alternate runtime shell path against GLFW for window/input/timing abstraction scope.
