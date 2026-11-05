# Tianjin longitudinal DR preprocessing

Complete implementation of the six requested preprocessing stages. This is a research pipeline, not a clinically validated label generator. The raw images are not included. Use Python 3.10–3.12 and Git; an isolated environment is recommended.

## Verified source details

The downloaded corrected_manifest.csv had 1,115 rows and nine columns, not the 1,148 pairs in the dataset-card summary. The workbook has Baseline (624 rows) and `2 year follow-up` (593 rows) sheets; 574 IDs occur in both. The loader validates the actual column prefixes before using the workbook positions. It never guesses pairs from filename order. It records hashes of files and fixes patient splits before processing.

The manifest identifies corresponding images but NOT OS/OD. `baseline_index` is not an eye-side label. If either visit has asymmetric OS/OD grades, finalization requires a manually verified laterality entry. For identical grades in both eyes at both visits, laterality is not needed to select the grade. No automatic method here proves same-eye identity: geometric matching is a screening check, and review is required before final labels.

## Install

```bash
python -m venv .venv
# Windows PowerShell: .venv\Scripts\Activate.ps1
# Linux/macOS: source .venv/bin/activate
python -m pip install -r requirements.txt
```

For segmentation, first install an appropriate matched torch/torchvision build for your CPU or GPU using https://pytorch.org/get-started/locally/ . Then:

```bash
python -m pip install -r requirements-models.txt
```

The segmentation toolkits download real pretrained weights. The first inference requires internet. There is no fake-mask or threshold-based fallback. Model errors produce error.txt and no completed result. CPU inference is supported but can be slow. CUDA is optional. The vessel model is a FIVES-trained ResNet34 U-Net; its architecture/weights and BGR input convention follow the author's current source. The lesion toolkit uses RGB and produces five softmax channels: BG, CTW, EX, HE, MA. CTW is the toolkit's spelling for cotton-wool spots. Hard lesion masks use argmax. The script fixes model revisions on first use and reuses them on later invocations.

## Execute

Run these commands from the folder containing pipeline.py.

```bash
python pipeline.py download --data data
python pipeline.py audit --data data --out outputs
python pipeline.py process --data data --out outputs --split train --limit 10 --registration-only
```

If you already downloaded the dataset, skip download and set `--data` to the directory that contains BOTH image directories, corrected_manifest.csv and Organized_Data of Patients.xlsx. Download pins the current repository SHA and saves it. Audit runs on the full dataset so patient splitting is meaningful; use --limit only for processing.

Inspect `outputs/registration/KEY/review.jpg` before proceeding. The first four panels are baseline, resized unaligned follow-up (for viewing only), aligned follow-up, and a blended grid overlay. Reject grossly wrong eye pairs, mismatched vessels, inconsistent optic disc positions, or substantial uncorrected distortion. SIFT/RANSAC residuals do not substitute for independently marked landmark error.

```bash
python pipeline.py process --data data --out outputs --split train --limit 10 --device cpu --grid 8
```

Use `--device cuda` only if the installed torch build and GPU support it. Inspect `outputs/processed/KEY/review.jpg`: its last two panels are lesion overlays, with CTW yellow, EX green, HE blue/orange, MA magenta in the saved BGR image. Also inspect vessels.jpg and common_fov.png. These are QC outputs, not published clinical annotations.

When the pilot is satisfactory:

```bash
python pipeline.py process --data data --out outputs --split train --device cpu --grid 8
```

Edit outputs/review.csv after viewing each pair. Do not automatically fill all rows with yes. Columns:

- laterality: OS (left eye) or OD (right eye), only when you can verify it. Anatomical orientation should be checked against actual acquisition conventions; filename order is not enough.
- same_eye: yes/no.
- registration_ok: yes/no.
- segmentation_ok: yes/no.
- reviewer: the reviewer’s name or initials.
- notes: reasons for rejection, uncertainty or artifacts.

If uncertain, leave pending until reviewed. A no excludes the pair. The first finalize command requires a decision for all training pairs and successful processing for all accepted training pairs. It also requires at least ten stable training eyes with nonzero change variability for every lesion. It fails rather than inventing scales when these conditions are unmet.

```bash
python pipeline.py finalize --out outputs
```

This freezes the label parameters and clinical imputation values from training data. Then process and review validation/test images with the same settings. Do not tune processing parameters on test results.

```bash
python pipeline.py process --data data --out outputs --split val --device cpu --grid 8
python pipeline.py process --data data --out outputs --split test --device cpu --grid 8
# Complete their review.csv entries, then:
python pipeline.py finalize --out outputs
```

A later finalize reuses the frozen training parameters. To change your training cohort, grid or definitions, create a NEW output folder and regenerate all stages; do not mix versions. Completed pair results are resumable; failed pairs are retried. Configuration and source hashes prevent stale reuse. `error.txt` records each failure; failures never receive a normal/stable label.

## What exactly is measured

1. Resize each visit to width 1024 with aspect ratio preserved; no square distortion.
2. Baseline grid bounds and retained regions derive ONLY from baseline FOV. Retain cells with >=50% baseline retinal coverage. No left/right mirroring.
3. SIFT on CLAHE green channel, affine RANSAC, optional guarded ECC refinement. The stored affine maps FOLLOW-UP coordinates to BASELINE coordinates. Require >=80% of baseline retinal area covered, plausible affine, spatially distributed matches and <=20 px p90 inlier residual. These are configurable research choices in the code, not validated clinical cutoffs.
4. Segment each visit independently BEFORE warping. Warp follow-up discrete lesion labels with nearest-neighbor interpolation. This avoids segmenting interpolated follow-up RGB images.
5. Baseline features use the entire baseline-visible portion of each cell. Outcome differences use the SAME common pixels in both visits. Cells with <80% of baseline-visible cell covered at follow-up get missing targets. Common coverage is never used to redefine baseline features or node existence.
6. For lesion k, delta[k] = follow-up lesion area fraction minus baseline area fraction within the common mask. A stable eye means its same-eye overall DR grade is unchanged (1–5) across visits. This is a calibration proxy: unchanged overall grade does NOT imply no local change.
7. On reviewed stable TRAINING eyes only, calculate sd[k] of signed regional changes, then S = sum(max(delta[k],0)/sd[k]). The threshold is the 95th percentile of S in those stable training regions. Worsening = S > threshold. Standard deviations and quantile pool valid regions; eye/patient weighting is not applied in this preprocessing definition. These choices must be reported in the methods. A zero SD triggers an error requiring investigation.
8. Exclude eye grades 6/7 or missing at either visit. Baseline glucose is kept in original source units; this workbook does not establish units in its GLU header, so no unit conversion is guessed. Clinical imputation is fit on unique training patients only. Features are left in raw units, except binary missing indicators. Any later scaling must be fit on the training set only.

The 2-year interval is a dataset-provided designation; this workbook does not provide individual visit timestamps. Do not invent measured patient-specific intervals. Treatment between visits is not fully controlled by this pipeline. Camera/resolution changes, registration and segmentation error can all resemble progression. Inspect these and report limitations.

## Output files

- audit.csv: all source pairs and exclusion reasons.
- duplicates.csv: exact file-hash duplicates (not a perceptual near-duplicate detector).
- pairs.csv: usable pairs, metadata, hashes and fixed split.
- review.csv: human review decisions.
- processed/KEY/: transform/QC, masks, region table, overlays and completion/error status.
- final/baseline_features.csv: ONLY baseline model inputs and keys.
- final/targets.csv: derived worsening labels and validity; missing targets remain missing.
- final/regional_outcomes.csv: diagnostic table including follow-up-derived quantities. **Do not feed this whole table into a model.**
- final/label_parameters.json and clinical_imputation.json: frozen training parameters.

Join features/targets on key, patient_id, split, region. Training/evaluation uses rows with nonmissing targets; preserve the baseline grid for future deployment. Both eyes always stay in the same patient split. Grade-based stratification uses the source patient progression field; it is never an input predictor.

## Tests and validation scope

```bash
python -m pytest test_pipeline.py -q
```

Tests cover patient-level split reproducibility, follow-up coverage independence of baseline features, missing-label propagation, training-only threshold fitting, zero-variance failure, ambiguous eye grades, path traversal and known affine direction. A real pair was also run through registration. These checks do not establish segmentation accuracy on this dataset or clinical validity of derived labels. See VALIDATION.md for the actual execution record.

## Sources

- Dataset: https://huggingface.co/datasets/usama10/retinal-dr-longitudinal
- Dataset metadata implementation inspected: https://github.com/Usama1002/retinal-diffusion/blob/master/autoresearch/prepare.py (its original filename-order pairing and global normalization are NOT used here).
- Lesion toolkit: https://github.com/ClementPla/fundus-lesions-toolkit, commit c1cc42416285ee486ad04c298d6d418c57e7e147.
- Vessel reference: https://github.com/gabriel-lepetitaimon/fundus-vessels-toolkit/blob/f725f502aa8b086c324c57d7206301e001a65b73/src/fundus_vessels_toolkit/segment/segmentation.py
- Vessel weights: https://huggingface.co/gabriel-lepetitaimon/FundusVessel

Follow source dataset/model use and citation terms. This package does not redistribute source photographs or the clinical workbook.
