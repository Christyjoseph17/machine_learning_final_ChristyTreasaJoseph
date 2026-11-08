# Social Media Use, Perceived Stress and Sleep in Students

### A leakage-controlled and explainable supervised learning study

**Author:** Christy Treasa Joseph
**Module:** EAM04DS Machine Learning
**Institution:** Emirates Aviation University

## Project overview

This project investigates whether recorded social media behaviours and demographic characteristics can predict perceived stress categories, sleep quality and sleep duration in a student dataset. It compares interpretable models with tree-based alternatives and examines model performance, uncertainty and explanations.

The study is predictive, not causal: its findings do not establish that social media use causes stress or changes sleep.

## Research objectives

- Predict stress-score categories, sleep quality and sleep duration from behavioural and demographic features.
- Compare interpretable linear and ordinal models with tree-based ensembles.
- Examine whether engineered exposure and interaction features improve development-set performance.
- Explain model predictions and explore performance across demographic and academic-level subgroups.
- Assess uncertainty, dataset provenance and limitations to generalisation.

## Dataset

The notebook expects a CSV file named `Social_media_impact_on_life.csv`. The saved notebook output shows **4,500 records and 16 columns** in the raw dataset.

The data includes social media usage, demographic characteristics and recorded stress and sleep outcomes. The notebook checks for the following columns:

`Student_ID`, `Age`, `Gender`, `Academic_Level`, `Primary_Platform`, `Daily_Usage_Hours`, `Weekend_Extra_Hours`, `Device_Type`, `Sleep_Duration_Hours`, `Sleep_Quality_Score`, `Late_Night_Usage`, `Social_Comparison_Frequency`, `Perceived_Stress_Score`, `Mental_Health_Index`, `Academic_Performance_GPA` and `Overall_Impact`.

The dataset is not included in the uploaded notebook. Its original source, collection methods, licensing and representativeness should be verified and documented before sharing or reuse. The stress bands are analytical categories, not clinical diagnoses.

## Methodology

1. **Data preparation:** Audit data integrity and duplicates, clean text, construct analytical targets and engineer candidate features while retaining the raw data as an audit reference.
2. **Leakage control:** Exclude identifiers and potentially outcome-derived summary variables from predictors. Do not use one study outcome as a predictor of another in the primary tasks. Fit learned preprocessing within training folds.
3. **Development and evaluation:** Use a shared 80/20 development/held-out split, stratified by stress category and missing stress status, with a random seed of 42. Model comparison and selection use the development partition, including nested cross-validation.
4. **Model comparison:** Compare logistic and ordinal regression with histogram gradient boosting for stress, ordinal regression with random forest for sleep quality, and Elastic Net feature sets for sleep duration.
5. **Interpretation and diagnostics:** Examine bootstrap uncertainty, calibration, subgroup performance, permutation importance, partial dependence and SHAP explanations.
6. **Reproducibility:** Export result tables and environment records, including an input-file SHA-256 fingerprint.

## Results reported in the notebook

The following are the notebook's reported results for the internal held-out partition. Confidence intervals are based on 1,000 bootstrap resamples of the held-out records and do not capture all uncertainty from model development and selection.

| Prediction task | Primary model | Reported held-out metric | 95% confidence interval |
|---|---|---:|---:|
| Stress category | Ordinal logit, base features | Macro-F1: 0.645 | 0.590–0.697 |
| Sleep quality | Ordinal logit, base features | Quadratic weighted kappa: 0.657 | 0.624–0.688 |
| Sleep duration | Elastic Net, base features | Mean absolute error: 0.642 hours | 0.609–0.673 hours |

In the notebook's interpretation, the primary models rely heavily on `Daily_Usage_Hours`. This describes the fitted models on this dataset and should not be interpreted as a causal effect or a population-wide conclusion.

**Evaluation caveat:** The notebook documents earlier exposure to held-out results during the study's development history. Rearranging the workflow does not undo that exposure, so the internal held-out estimates should be interpreted with caution. They are not external validation.

## Technologies

- Python and Google Colab
- NumPy, pandas and SciPy
- scikit-learn and statsmodels
- Matplotlib and seaborn
- SHAP and ipywidgets

## Getting started

### Run in Google Colab

1. Upload `ChristyTreasaJoseph_MachineLearning_Final (1).ipynb` to Google Colab and open it.
2. Obtain the dataset separately and make sure its filename is `Social_media_impact_on_life.csv`.
3. Run the notebook cells in order. The first code cell installs `shap` and `statsmodels` and checks for the CSV at `/content/Social_media_impact_on_life.csv`.
4. If the CSV is not already there, use the Colab upload prompt to select it.
5. Review the outputs, warnings and export messages. Model comparison and bootstrap evaluation may take time.

The notebook is written for Colab, including its file-upload mechanism and `/content` path. Running it in a local Jupyter environment requires adapting the data-loading cell and installing the necessary packages. Exact package versions are not pinned in the notebook; environment records describe the session when they are generated and do not prove which versions produced earlier saved outputs.

## Project files and outputs

- `ChristyTreasaJoseph_MachineLearning_Final (1).ipynb` — analysis, modelling, evaluation and interpretation notebook.
- `Social_media_impact_on_life.csv` — required input dataset, to be obtained separately.

The notebook also attempts to export additional model-selection, diagnostic, subgroup and interpretation tables. Check its export messages to confirm which files were successfully produced.

## Limitations

- Dataset provenance, sampling and measurement methods require independent verification.
- The results reflect one dataset and a single internal hold-out; they are not independent external validation.
- Prior exposure to held-out results is a limitation on the interpretation of test performance.
- Subgroup comparisons are exploratory, and small groups may not support reliable estimates.
- Feature explanations describe model behaviour, not causal relationships.
- Stress categories are not clinical diagnoses, and the models should not be used for clinical decision-making.
