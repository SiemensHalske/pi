# powder_cpp

Phase 1 scaffold for a full native C++ PowderPhysicsEngine rewrite.

## Build

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=RelWithDebInfo
cmake --build build -j
```

## Run

```bash
./build/engine_app_main --profile deterministic --headless --threads 8
```

## Build Profiles

- `Debug`
- `RelWithDebInfo`
- `Release`
- `ReleaseLTO`
- `ASanUBSan`

## Deterministic Runtime Policy

- Fixed seed is used for bootstrap initialization.
- Tick rate defaults to 60 Hz and fixed `dt = 1/60`.
- Physics substep ordering is fixed and explicit.
- Floating-point policy: IEEE-754 mode, no fast-math assumptions in deterministic profile.
