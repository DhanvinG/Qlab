# Validation record

Six automated tests passed. Actual source manifest and workbook schemas were inspected. One actual baseline/follow-up pair passed registration screening (92.5% overlap; 1.45 px median inlier reprojection residual, not independently measured landmark error).

Torch 2.5.1 CPU, torchvision 0.20.1 CPU and segmentation toolkit imports were checked. Full pretrained segmentation inference was NOT completed: model retrieval encountered a missing SOCKS proxy dependency in this execution environment. httpx[socks] is included in the installation requirements. This addition has not been verified by a completed model download/inference run.

The full cohort has NOT been processed or reviewed. Accuracy of segmentation, same-eye pairing, and clinical validity of derived labels are not established. Review outputs before accepting samples. The label calibration rule is an explicit research choice described in README.md; it is not a validated clinical definition.
