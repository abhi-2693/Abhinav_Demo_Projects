# Project 16: Predictive Analytics (R) — Aravind Screening vs Surgery Modeling

## Situation / Objective
Aravind’s operational outcomes (total surgeries) can depend on outreach and screening strategy. The objective is to quantify how **paid** and **free** screening volumes relate to **total surgeries**, including interaction effects, and interpret results in actionable terms.

## Task
- Load historical screening and surgery data.
- Transform variables to support elasticity-style interpretation.
- Fit regression models with and without interaction.
- Select the better model and interpret coefficients.

## Actions
- Loaded the dataset from an Excel source and removed an instructed outlier year with zero values.
- Created log-transformed variables (e.g., log total surgeries, log paid screenings, log free screenings).
- Trained:
  - A baseline model without interaction.
  - An interaction model to capture diminishing/mutual overlap effects.
- Compared models using adjusted R-squared and interpreted elasticities and scenario effects (e.g., impact of a 15% change).

## Results / Summary
- Produced an interpretable regression model explaining most of the variance in log total surgeries.
- Identified that increasing paid/free screening is associated with higher surgeries, but interaction terms indicate diminishing returns when both are increased together.

## Repository contents
- `Aravind_screening_surgery_modeling.Rmd`
- `Aravind_screening_surgery_modeling.html`
