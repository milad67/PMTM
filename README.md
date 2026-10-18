# PMTM

**Progressive multi-scale tumor-state memory for incomplete multimodal brain tumor segmentation**

This repository accompanies the PMTM manuscript.

## Release status

- **BraTS2020 and BraTS2023 GLI split identifiers:** available as the [original supplied files](splits) and in [subject_splits.csv](subject_splits.csv).
- **BraTS2018 split identifiers:** not yet uploaded. The study uses its own 205/23/57 training/validation/test partition; this file does not substitute a published split for it.
- **PMTM source code:** not yet uploaded.
- **Measured per-subject evaluation results:** not yet uploaded. [per_subject_results_TEMPLATE.csv](per_subject_results_TEMPLATE.csv) is a header-only format template, not evaluation data.

## Subject splits

| Dataset | Train | Validation | Test |
|---|---:|---:|---:|
| BraTS2020 | 219 | 50 | 100 |
| BraTS2023 GLI | 875 | 125 | 251 |

`subject_splits.csv` contains one row per subject with columns `dataset`, `split`, and `subject_id`. Identifiers and their order are retained from the upstream lists. BraTS2020 identifiers retain the upstream `HG_` or `LG_` prefix. Train, validation, and test identifiers have no overlap within either dataset, including after removal of that prefix.

The BraTS2020 lists originate from the [official M2FTrans repository](https://github.com/Jun-Jie-Shi/M2FTrans/tree/main/BraTS/BRATS2020_Training_none_npy). The BraTS2023 GLI lists originate from the [public IM-Fuse datalists in MiMoSe](https://github.com/AImageLab-zip/MiMoSe/tree/main/IMFuse/datalist). For the BraTS2023 validation/test CSV files, only the `case` column is used to extract subject membership; the supplied `mask` column is not used to infer PMTM evaluation settings or scores.

See [split_provenance.json](split_provenance.json) for the source paths, pinned source commits, blob hashes, transformations, and validation checks. All six subject-list files supplied by the manuscript authors were checked byte for byte against the pinned public source files and match exactly. Their subject membership and ordering also match the consolidated CSV. The originals are provided in [splits/2020](splits/2020) and [splits/2023](splits/2023), including the original BraTS2023 CSV mask columns.

## Per-subject results template

The template contains column names only. It has no subject rows or Dice scores. Populate it only from actual evaluation outputs:

- One row per dataset, subject, method, and evaluated modality subset.
- Use `0` or `1` for each modality-availability column. At least one modality must be available.
- Report the measured WT, TC, and ET Dice scores in percent (0 to 100); do not enter dataset averages as individual-subject values.
- Retain the actual evaluation conventions, including handling of empty target regions, when exporting scores. The template does not define or change those conventions.

The template must not be cited as released evaluation results. Dataset-level table averages cannot reconstruct individual-subject measurements.

## MRI data

The original MRI data are not redistributed in this repository. Access to the BraTS datasets remains subject to the access procedures and terms of the respective data providers. The subject identifiers above are dataset identifiers, not patient names.
