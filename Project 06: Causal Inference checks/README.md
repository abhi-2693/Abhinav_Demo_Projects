# Project 06: Causal Inference Checks — Wage, IQ, and Education (R)

## Situation / Objective
Understanding drivers of wages often requires separating correlation from causal interpretation and checking for omitted-variable bias. The objective is to analyze wage data and evaluate how IQ and education relate to wages using regression.

## Task
- Explore wage and IQ/education relationships.
- Fit simple and multiple regression models.
- Test statistical significance of coefficients.
- Validate omitted-variable-bias (OVB) relationships between coefficients.

## Actions
- Loaded wage dataset and performed summary statistics and visualization (scatterplot with fitted line).
- Estimated:
  - Simple regression: `wage ~ IQ`
  - Multiple regression: `wage ~ IQ + educ`
- Conducted hypothesis tests using p-values and interpreted effect sizes.
- Performed an explicit OVB check by estimating `educ ~ IQ` and verifying the coefficient identity connecting simple vs. multiple regression estimates.

## Results / Summary
- Confirmed both IQ and education are statistically significant predictors of wages in the modeled sample.
- Demonstrated how adding education changes the IQ coefficient and validated the OVB relationship mathematically.

## Repository contents
- `Causal_inference_project.Rmd`
- `Causal_inference_project.html`
