// merge_6ch_f260517.ijm
//
// Select the parent folder (e.g. f260517_0625/).
// Shows a 6-channel hyperstack:
//   C1 = raw_mem    C2 = proj_mem    C3 = ref_mem
//   C4 = raw_sparse C5 = proj_sparse C6 = ref_sparse

setBatchMode(true);

// ---- 1. Select folder ----
rootDir = getDirectory("Select parent folder (e.g. f260517_0625/)");
if (rootDir == "") exit("No folder.");

// ---- 2. Collect tif files from each sub-folder ----
function getTifs(subfolder) {
    list = getFileList(subfolder);
    out = newArray();
    for (i = 0; i < list.length; i++)
        if (endsWith(list[i], ".tif"))
            out = Array.concat(out, list[i]);
    return Array.sort(out);
}

rawMemFiles    = getTifs(rootDir + "raw_moving_mem/");
projMemFiles   = getTifs(rootDir + "projected_mem/");
refMemFiles    = getTifs(rootDir + "ref_surface_mem/");
rawSparseFiles = getTifs(rootDir + "raw_moving_sparseCell/");
projSparseFiles = getTifs(rootDir + "projected_sparseCell/");
refSparseFiles  = getTifs(rootDir + "ref_surface_sparseCell/");

nFrames = rawMemFiles.length;
if (nFrames == 0) exit("No TIFFs in raw_moving_mem/");

// ---- 3. Dimensions ----
open(rootDir + "raw_moving_mem/" + rawMemFiles[0]);
getDimensions(W, H, nC, nZ, nT);
close();
print("T=" + nFrames + "  Z=" + nZ + "  XY=" + W + "x" + H);

// ---- 4. Create blank 6-channel hyperstack ----
newImage("Combined", "32-bit black", W, H, 6 * nZ * nFrames);
run("Stack to Hyperstack...",
    "order=xyczt(default) channels=6 slices=" + nZ +
    " frames=" + nFrames + " display=Composite");
selectWindow("Combined");

// ---- 5. Fill each frame ----
for (f = 0; f < nFrames; f++) {

    // Build list of [dir, fileIndex] for this frame
    // Time-series: use frame f.  Ref: always index 0.
    dirs  = newArray(rootDir+"raw_moving_mem/", rootDir+"projected_mem/", rootDir+"ref_surface_mem/",
                     rootDir+"raw_moving_sparseCell/", rootDir+"projected_sparseCell/", rootDir+"ref_surface_sparseCell/");
    files = newArray(rawMemFiles[f], projMemFiles[f], refMemFiles[0],
                     rawSparseFiles[f], projSparseFiles[f], refSparseFiles[0]);

    for (c = 0; c < 6; c++) {
        open(dirs[c] + files[c]);
        src = getTitle();

        for (z = 1; z <= nZ; z++) {
            selectWindow(src);
            setSlice(z);
            run("Copy");

            selectWindow("Combined");
            Stack.setPosition(c + 1, z, f + 1);
            run("Paste");
        }
        selectWindow(src);
        close();
    }

    if (f % 50 == 0)
        showProgress(f + 1, nFrames);
}

// ---- 6. Display ----
for (c = 1; c <= 6; c++) {
    Stack.setChannel(c);
    run("Enhance Contrast", "saturated=0.35");
}
Stack.setPosition(1, nZ / 2, nFrames / 2);

setBatchMode(false);
print("Done.  " + nFrames + " frames, " + nZ + " Z, 6 channels.");
