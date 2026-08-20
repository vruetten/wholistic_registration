# GL header shim

`GL/gl.h` and `KHR/khrplatform.h` live in the conda-gcc sysroot, but putting
that whole sysroot on the include path shadows glibc and libstdc++ from the
system gcc that compiles this project, which breaks `<cmath>` and everything
downstream of it. Symlinking just those two subdirectories gives the GL headers
without the shadowing.

The symlinks are machine-specific and are not checked in. Recreate them once
per machine:

```bash
conda activate all
cd dashboard/native
mkdir -p build_include/GL build_include/KHR
ln -sf $CONDA_PREFIX/x86_64-conda-linux-gnu/sysroot/usr/include/GL/{gl,glext,glcorearb,glxext}.h build_include/GL/
ln -sf $CONDA_PREFIX/x86_64-conda-linux-gnu/sysroot/usr/include/KHR/khrplatform.h build_include/KHR/
```
