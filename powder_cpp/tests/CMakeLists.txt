add_executable(test_phase1_smoke
  smoke.cpp
)

target_link_libraries(test_phase1_smoke PRIVATE engine_core powder_build_flags)

add_test(NAME phase1_smoke COMMAND test_phase1_smoke)

add_executable(test_phase2_core
  phase2_core.cpp
)

target_link_libraries(test_phase2_core PRIVATE engine_core powder_build_flags)

add_test(NAME phase2_core COMMAND test_phase2_core)

add_executable(test_phase3_core
  phase3_core.cpp
)

target_link_libraries(test_phase3_core PRIVATE engine_sim engine_core powder_build_flags)

add_test(NAME phase3_core COMMAND test_phase3_core)

add_executable(test_phase4_core
  phase4_core.cpp
)

target_link_libraries(test_phase4_core PRIVATE engine_sim engine_core powder_build_flags)

add_test(NAME phase4_core COMMAND test_phase4_core)

add_executable(test_phase5_core
  phase5_core.cpp
)

target_link_libraries(test_phase5_core PRIVATE engine_sim engine_core powder_build_flags)

add_test(NAME phase5_core COMMAND test_phase5_core)

add_executable(test_phase6_core
  phase6_core.cpp
)

target_link_libraries(test_phase6_core PRIVATE engine_sim engine_core powder_build_flags)

add_test(NAME phase6_core COMMAND test_phase6_core)
