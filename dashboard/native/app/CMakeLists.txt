# wrdash: the registration-QC layer. Everything that knows what a frame, a
# plane, a displacement field or a reference volume is lives here; dashcore
# below it knows none of those words, and core/tests/check_no_domain_leak.sh
# fails the build if one reaches it.
add_library(wrdash_run STATIC
  project.cpp
  run.cpp
  tiff_volume.cpp
  csv_table.cpp
)
target_include_directories(wrdash_run PUBLIC ${CMAKE_SOURCE_DIR})
target_link_libraries(wrdash_run PUBLIC
  dashcore
  nlohmann_json::nlohmann_json
  ${TIFF_LIB}
)

add_subdirectory(tests)
