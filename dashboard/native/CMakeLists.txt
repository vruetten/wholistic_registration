cmake_minimum_required(VERSION 3.20)
project(wr_dashboard LANGUAGES CXX C)

# cnpy still ships `cmake_minimum_required(VERSION 2.6)`, refused by CMake 4.
set(CMAKE_POLICY_VERSION_MINIMUM 3.5)

# This top level vendors third-party libraries, then adds core/ (dashcore: a
# static library with no knowledge of any particular pipeline) and app/
# (wrdash: the registration-QC panels and the executable).

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

enable_testing()

# ---------- Deps from the conda env ----------------------------------------
# Relies on the `all` conda env for OpenGL, GLFW, X11 and libtiff. Skipping
# find_package(OpenGL) because conda's libglvnd installs .so.0 without a
# plain .so symlink, which trips CMake's stock probe.
if(DEFINED ENV{CONDA_PREFIX})
  set(CONDA_ROOT $ENV{CONDA_PREFIX})
else()
  message(FATAL_ERROR "CONDA_PREFIX not set. Activate the `all` env first: `conda activate all`.")
endif()

set(CONDA_INCLUDE ${CONDA_ROOT}/include)
set(CONDA_LIB     ${CONDA_ROOT}/lib)
# GL/gl.h lives in the conda-gcc sysroot, but adding that whole sysroot
# shadows glibc/libstdc++ from the system gcc we compile with and breaks
# <cmath>. Instead symlink just the GL/ and KHR/ subdirs into a local shim
# (see build_include/README.md).
set(GL_HEADER_SHIM ${CMAKE_SOURCE_DIR}/build_include)

# Conda ships libGL.so.1 without a plain libGL.so symlink, so find_library(GL)
# fails; point at the versioned file directly.
find_library(GL_LIB
  NAMES GL libGL.so.1
  PATHS ${CONDA_LIB} ${CONDA_ROOT}/x86_64-conda-linux-gnu/sysroot/usr/lib64
  NO_DEFAULT_PATH REQUIRED
)
find_library(GLFW_LIB glfw PATHS ${CONDA_LIB} NO_DEFAULT_PATH REQUIRED)
# The projected / refspace / raw-moving volumes are TIFF, so unlike the NMF
# dashboard this one needs libtiff. It is consumed by app/ only; dashcore
# stays format-agnostic beyond .npy.
find_library(TIFF_LIB tiff PATHS ${CONDA_LIB} NO_DEFAULT_PATH REQUIRED)
find_library(DL_LIB   dl)
find_library(RT_LIB   rt)
find_package(Threads REQUIRED)

# ---------- Vendored via FetchContent --------------------------------------
# Pin tags so a fresh clone builds bit-identically; bump deliberately. Same
# pins as NMFDemo/dashboard, so the vendored dashcore compiles against the
# versions it was written for.
include(FetchContent)

FetchContent_Declare(
  imgui
  GIT_REPOSITORY https://github.com/ocornut/imgui.git
  GIT_TAG        v1.91.5-docking       # docking branch tag; DockBuilder is required
  GIT_SHALLOW    TRUE
)
FetchContent_MakeAvailable(imgui)

FetchContent_Declare(
  implot
  GIT_REPOSITORY https://github.com/epezent/implot.git
  GIT_TAG        v0.16
  GIT_SHALLOW    TRUE
)
FetchContent_MakeAvailable(implot)

FetchContent_Declare(
  cnpy
  GIT_REPOSITORY https://github.com/rogersce/cnpy.git
  GIT_TAG        master
  GIT_SHALLOW    TRUE
)
FetchContent_MakeAvailable(cnpy)

FetchContent_Declare(
  tomlplusplus
  GIT_REPOSITORY https://github.com/marzer/tomlplusplus.git
  GIT_TAG        v3.4.0
  GIT_SHALLOW    TRUE
)
FetchContent_MakeAvailable(tomlplusplus)

FetchContent_Declare(
  nlohmann_json
  GIT_REPOSITORY https://github.com/nlohmann/json.git
  GIT_TAG        v3.11.3
  GIT_SHALLOW    TRUE
)
set(JSON_BuildTests OFF CACHE INTERNAL "")
FetchContent_MakeAvailable(nlohmann_json)

# stb: header-only, single-file image codecs. Used for PNG screenshot capture
# (stb_image_write.h) and, test-only, PNG round-trip verification
# (stb_image.h). The repo carries no version tags, so pinned to a commit.
FetchContent_Declare(
  stb
  GIT_REPOSITORY https://github.com/nothings/stb.git
  GIT_TAG        2c980bb59875b0d32144a71867fbdebb2f77cd20
  GIT_SHALLOW    FALSE
)
FetchContent_MakeAvailable(stb)
add_library(stb INTERFACE)
target_include_directories(stb INTERFACE ${stb_SOURCE_DIR})

# Test-only: doctest is single-header and only needed by the tests/ trees.
FetchContent_Declare(
  doctest
  GIT_REPOSITORY https://github.com/doctest/doctest.git
  GIT_TAG        v2.4.11
  GIT_SHALLOW    TRUE
)
FetchContent_MakeAvailable(doctest)

# ---------- ImGui + ImPlot as library targets ------------------------------
add_library(imgui STATIC
  ${imgui_SOURCE_DIR}/imgui.cpp
  ${imgui_SOURCE_DIR}/imgui_draw.cpp
  ${imgui_SOURCE_DIR}/imgui_tables.cpp
  ${imgui_SOURCE_DIR}/imgui_widgets.cpp
  ${imgui_SOURCE_DIR}/imgui_demo.cpp
  ${imgui_SOURCE_DIR}/backends/imgui_impl_glfw.cpp
  ${imgui_SOURCE_DIR}/backends/imgui_impl_opengl3.cpp
)
target_include_directories(imgui PUBLIC
  ${imgui_SOURCE_DIR}
  ${imgui_SOURCE_DIR}/backends
  ${CONDA_INCLUDE}
  ${GL_HEADER_SHIM}
)
target_link_libraries(imgui PUBLIC ${GLFW_LIB} ${GL_LIB} Threads::Threads)

add_library(implot STATIC
  ${implot_SOURCE_DIR}/implot.cpp
  ${implot_SOURCE_DIR}/implot_items.cpp
  ${implot_SOURCE_DIR}/implot_demo.cpp
)
target_include_directories(implot PUBLIC ${implot_SOURCE_DIR})
target_link_libraries(implot PUBLIC imgui)

# ---------- dashcore, then wrdash -------------------------------------------
add_subdirectory(core)
if(EXISTS ${CMAKE_SOURCE_DIR}/app/CMakeLists.txt)
  add_subdirectory(app)
endif()
