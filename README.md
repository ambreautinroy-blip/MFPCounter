# ductcount

Count mammary ducts in H&E microscope fields of tissue sections, with a web app to browse every
section, count by clicking, check a deep-learning detector and retrain it on your corrections.

* Overlapping microscope fields of a cut are **stitched automatically**; a duct seen in several
  fields is counted **once** (in the first field that shows it).
* Serial cuts of a section are **aligned**, so marks of the other cuts can be shown while counting.
* A **U-Net duct-centre detector** proposes counts; corrections made in the app retrain it.
* Ducts can be tagged **normal / atypical**; the model learns the class once examples exist.

## Install

Requires [uv](https://docs.astral.sh/uv/) and Python ≥ 3.12 (macOS with Apple silicon uses the
GPU through MPS; CUDA or CPU also work).

```bash
git clone <this repository> && cd ductcount
uv sync
```

## Use

```bash
uv run ductcount setup /path/to/images        # scan, stitch, align (~5 s per field the first time)
uv run ductcount serve                        # http://localhost:8765
```

Everything generated goes to `./data` (set `DUCT_DATA=/somewhere/else` to change it): image cache,
stitching, mosaics, annotations, predictions, models and job logs. The images are never modified.
Re-running `setup` keeps the annotations already made in the app.

### Images folder

File names must contain `<mouse>.<section>.<cut> Field_<n>` and end in `.jpg`, `.tif` or `.png`,
in any sub-folder, e.g.

```
images/T47D MFP 1/T47D MFP 1 Section 3/T47D MFP 1.3.2 Field_0004.jpg
```

= mouse 1, section 3, serial cut 2, field 4. Consecutive fields of a cut are expected to overlap
(they are stitched with SIFT feature matching). The field-of-view model assumes 4032×3040 images
with a circular field of view (a warning is printed otherwise). `setup` reports how many images,
cuts and mice it found and lists files whose names it could not parse.

**Existing manual counts** (optional): ImageJ multipoint exports named `<image name>_coords.csv`
(columns `Image, Index, X, Y`) anywhere under the folder are imported as marks; those fields are
treated as fully annotated and are used for training. Fields without such a file start empty and
unreviewed.

### Models

Trained weights are not part of the repository. Without a model the app works for manual counting;
mark ducts in a few fields, tick **Field reviewed**, and click **Retrain** to train a first model
(from ImageNet weights; one model per held-out mouse when ≥ 2 mice are annotated, ~15 min each on
an M1). To reuse a trained model:

```bash
uv run ductcount export-models shared_models/          # on the machine that has it
uv run ductcount setup /path/to/images --models shared_models/
```

## Web app

* **Summary**: every cut and section with manual and model counts (section = mean of its cuts).
* **Overview**: the stitched cut with all marks; click a location to open the field that owns it.
  **Edit alignment** lets you drag / rotate (`Q`/`E`, Shift = 5°) a misplaced field, then rebuilds
  the mosaic and the counts of that cut.
* **Field**: click = one duct (`1` normal, `2` atypical); click a mark to select it, `Del` or
  right-click to remove it; drag to pan, wheel to zoom.
  * `←`/`→` fields, `↑`/`↓` cuts (lands on the matching field of the next cut when aligned).
  * Grey dashed rings: duct already counted in another field of the cut — don't count it again.
  * Teal dotted rings: marks of the other cuts of the section (shown when the alignment is reliable).
  * Yellow rings: model detections. `T` training target, `H` model heatmap.
* **Field reviewed**: the field's marks are complete and can be used for training.
* **Retrain**: fine-tunes the current model(s) on all reviewed fields (from scratch if there is
  none) and re-predicts everything. With cross-validated models, each mouse is predicted by a model
  trained without that mouse, so corrections on a mouse improve the predictions of the *other* mice.
* Atypical: a field with at least one atypical mark is taken as fully classified — when tagging,
  classify every duct of that field.
* Export: CSV per cut and per field.

## Command line

```
uv run ductcount setup IMAGES [--models DIR]
uv run ductcount serve [--port 8765]
uv run ductcount infer                  # predict all cuts with the current model
uv run ductcount train [--epochs N]     # same as the Retrain button
uv run ductcount export-models OUT_DIR
```

## How it works

1. **Stitching** (`stitch.py`, `build_cache.py`): SIFT + RANSAC similarity between all field pairs of a
   cut, spanning tree on the strongest links → every field in a common mosaic frame.
2. **Detection** (`unet.py`): U-Net (ResNet-34 encoder, ImageNet weights) predicting a Gaussian heatmap
   at each duct centre on ¼-resolution fields, with per-image Reinhard colour normalisation and stain
   augmentation; local maxima = ducts. Training targets are the manual marks, propagated to every
   overlapping field that shows the same duct.
3. **Counting** (`jobs.py`): detections mapped to the mosaic, merged within 60 px, each duct assigned
   to the first field whose field of view contains it.
4. **Cut-to-cut alignment** (`project.py`, `xcut_refine.py`, `xcut_check.py`): rotation search + FFT
   correlation on stain saturation of the mosaics, ECC refinement, then per-field local ECC; a pair of
   cuts is flagged reliable when the ducts of one land on the ducts of the other.
5. **Evaluation** (`calibrate.py`, `run_unet.py`, `run_classical.py`, research scripts): leave-one-mouse-out
   cross-validation; the peak threshold for a mouse is chosen on the out-of-fold predictions of the
   other mice. `classical.py` is a colour-deconvolution + blob-detection baseline.
