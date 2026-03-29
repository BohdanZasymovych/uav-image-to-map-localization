# UAV Localization — Aerial-to-Satellite Image Registration

Vision-based UAV localization via feature matching, RANSAC, and affine/projective
transformation estimation. Linear algebra course project.

## Setup

```bash
pip install -r requirements.txt
```

## Run the UI

```bash
streamlit run app/ui.py
```


## Package structure

```
localization/
  features/          FeatureExtractor ABC + SIFT, ORB, SURF
  transforms/        TransformationModel ABC + Affine, Projective, Similarity
  ransac.py          model-agnostic RANSAC
  pipeline.py        LocalizationPipeline orchestrator
  georeferencing.py  pixel -> lat/lon conversion
  result.py          shared dataclasses (MatchResult, LocalizationResult)
evaluation/
  base.py            DatasetGenerator ABC + SyntheticFrame dataclass
  dataset.py         SyntheticDatasetGenerator
  metrics.py         Evaluator + EvaluationReport
  visualizer.py      matplotlib plots
app/
  ui.py              Streamlit application
```

## Work split

<!-- | File(s)                              | Owner    |
|--------------------------------------|----------|
| transforms/base, affine, projective, similarity, georeferencing | Person A |
| features/base, sift, orb, surf, ransac, pipeline               | Person B |
| evaluation/base, dataset, metrics, visualizer, app/ui           | Person C | -->

| File(s)                              | Owner    |
|--------------------------------------|----------|
| `evaluation/`, `ransac` | Person A |
| `features/`, `georeferencing` | Person B |
| `transforms/`, `pipeline` | Person C |

