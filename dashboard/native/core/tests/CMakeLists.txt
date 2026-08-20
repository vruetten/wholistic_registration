# The seam this whole build proves: dashcore_tests links only against the
# `dashcore` target (plus doctest) and adds no include directory of its own
# beyond this tests/ folder for its own headers. app/ exists and must never
# be added here.
add_executable(dashcore_tests
  test_main.cpp
  test_selection.cpp
  test_npy.cpp
  test_table_panel.cpp
  test_theme.cpp
  test_heatmap_panel.cpp
  test_image_canvas.cpp
  test_panel.cpp
  test_panel_seam.cpp
  test_gl_texture.cpp
  test_percentile.cpp
  test_plot_autoscale.cpp
)
target_include_directories(dashcore_tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(dashcore_tests PRIVATE dashcore doctest::doctest)

add_test(NAME dashcore_tests COMMAND dashcore_tests)

# ---- SelectionSet::ids/version must be private (adversarial review HIGH 1) --
# fixtures/direct_field_access.cpp directly mutates both fields; it must FAIL
# to compile. If it ever compiles, the accessor-only API has regressed to
# "enforced by convention," which is the exact defect this guards against.
try_compile(
  DASHCORE_SELECTION_FIELDS_PRIVATE_COMPILES
  ${CMAKE_CURRENT_BINARY_DIR}/try_compile_selection_fields
  SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/fixtures/direct_field_access.cpp
  CXX_STANDARD 20
  CXX_STANDARD_REQUIRED ON
  CMAKE_FLAGS "-DINCLUDE_DIRECTORIES=${CMAKE_CURRENT_SOURCE_DIR}/../.."
)
if(DASHCORE_SELECTION_FIELDS_PRIVATE_COMPILES)
  message(FATAL_ERROR
    "core/tests/fixtures/direct_field_access.cpp compiled successfully: "
    "SelectionSet::ids/version are public, not private as HIGH 1 requires.")
endif()
add_test(NAME selection_fields_are_private
         COMMAND ${CMAKE_COMMAND} -E true)
