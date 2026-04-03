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
| `evaluation/`, `ransac` | Solomiia |
| `features/`, `georeferencing` | Marta |
| `transforms/`, `pipeline` | Bohdan |

## External Library For Evaluation (`evaluation/`)

For `evaluation/test.py`, we use the external dataset library **OrthoLoC**.

Install in your virtual environment:

```bash
pip install --no-deps git+https://github.com/deepscenario/OrthoLoC.git
pip install imcui==0.0.7 rasterio appdirs gputil opencv-python-headless py-cpuinfo tueplots
```

Run the evaluation demo script:

```bash
python -m evaluation.test
```
