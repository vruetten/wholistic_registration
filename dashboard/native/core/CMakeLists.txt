# dashcore: a static library with no knowledge of any particular pipeline.
# Everything here is included as "core/<path>.hpp" from a public include root
# one level up (dashboard/), so an external consumer spells it the same way
# code inside this directory does.

add_library(dashcore STATIC
  colormap.cpp
  panel.cpp
  table_panel.cpp
  heatmap_panel.cpp
  image_canvas.cpp
  npy.cpp
  npz.cpp
  detail/npy_header.cpp
  theme.cpp
  ui/plot_autoscale.cpp
  ui/widgets.cpp
  util/gl_texture.cpp
  util/percentile.cpp
)

target_include_directories(dashcore PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/..)
# imgui/implot types (ImTextureID, ImPlotColormap, ...) appear in dashcore's
# own public headers and every panel .cpp calls into both, so both propagate
# as PUBLIC. cnpy is an implementation detail of npy.cpp only — no dashcore
# header exposes a cnpy type — so it stays PRIVATE.
target_link_libraries(dashcore PUBLIC imgui implot)
target_link_libraries(dashcore PRIVATE cnpy)
# cnpy's target doesn't propagate its own include dir.
target_include_directories(dashcore PRIVATE ${cnpy_SOURCE_DIR})
# npz.cpp inflates zip members itself rather than going through cnpy's npz
# path, which reports neither the dtype kind nor the byte order and so cannot
# support the same guards as npy.cpp.
find_package(ZLIB REQUIRED)
target_link_libraries(dashcore PRIVATE ZLIB::ZLIB)
target_compile_options(dashcore PRIVATE -Wall -Wextra -Wpedantic)

# ---------- Enforce the seam in the build, not just by convention ----------
# No file under core/ may reference a domain concept (a pipeline "run", a
# "cell", a "voxel", "NMF" itself). This scans the whole core/ tree,
# including tests/, and fails the build the moment one leaks in.
add_custom_target(dashcore_no_domain_leak ALL
  COMMAND bash ${CMAKE_CURRENT_SOURCE_DIR}/tests/check_no_domain_leak.sh ${CMAKE_CURRENT_SOURCE_DIR}
  COMMENT "dashcore: verifying core/ has no cell/run/voxel/NMF vocabulary"
  VERBATIM
)
add_dependencies(dashcore dashcore_no_domain_leak)

add_test(
  NAME no_domain_leak
  COMMAND bash ${CMAKE_CURRENT_SOURCE_DIR}/tests/check_no_domain_leak.sh ${CMAKE_CURRENT_SOURCE_DIR}
)

add_subdirectory(tests)
