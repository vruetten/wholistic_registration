# The run-loader tests read two real run directories on /nrs. They are
# registered only when both exist.
#
# Skipping inside the test binary would be worse: a suite that reports "passed"
# on a machine with no data has converted "unverified" into "verified" without
# doing any work. Not registering the test, and saying so at configure time,
# leaves the absence visible.
set(WRDASH_RUN_ROOT
    "/nrs/ahrens/Virginia_nrs/wVT/mesoscope/260517_ubbr_mkate_phox2b/registration_out"
    CACHE PATH "Directory holding the QC run directories the loader tests read")

set(WRDASH_RUN_V4   "${WRDASH_RUN_ROOT}/f260517_0625_qc_v4")
set(WRDASH_RUN_100T "${WRDASH_RUN_ROOT}/f260517_0625_qc_4slice_100t_movie")

if(EXISTS "${WRDASH_RUN_V4}" AND EXISTS "${WRDASH_RUN_100T}")
  add_executable(wrdash_tests
    test_main.cpp
    test_run.cpp
  )
  target_link_libraries(wrdash_tests PRIVATE wrdash_run doctest::doctest)
  target_compile_definitions(wrdash_tests PRIVATE
    WRDASH_RUN_V4="${WRDASH_RUN_V4}"
    WRDASH_RUN_100T="${WRDASH_RUN_100T}"
  )
  add_test(NAME wrdash_tests COMMAND wrdash_tests)
  message(STATUS "wrdash_tests: enabled against ${WRDASH_RUN_ROOT}")
else()
  message(STATUS
    "wrdash_tests: DISABLED — run directories not found under ${WRDASH_RUN_ROOT}. "
    "The run loader is therefore unverified in this build.")
endif()
