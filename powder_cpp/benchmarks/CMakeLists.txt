add_executable(bench_phase1
  bench_stub.cpp
)

target_link_libraries(bench_phase1 PRIVATE engine_core powder_build_flags)
