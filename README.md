# WNBA Playoff Qualification Prediction

## Overview
This project analyzes WNBA player performance and develops predictive models to forecast team playoff qualifications. By leveraging Machine Learning techniques, we aim to predict team success based on individual player statistics and overall team performance.

## Project Structure
The driver code of the project is in the `main.py` file. At the end of the file, one can choose between running the `main` function, which trains and tests several models before computing predictions for year 10, or the `predict_11th_year` function, which forecasts the outcome of the 11th season.

The core responsibilities of each module are:

- **`main.py`**: Handles feature engineering, data cleaning, processing, and preparation. This file serves as the entry point for executing model training and predictions.
- **`models.py`**: Implements various Machine Learning models, manages their training and evaluation, and is called from the main module.
- **CustomCrossValidator**: Implements a sliding window logic for cross-validation, as described in the annexes of the project presentation.
- **`analysis.py`**: Contains code for generating plots and visualizations used in the presentation.
- **`models/`**: Directory containing all pre-trained models for easier evaluation.
- **`data_reports/`**: Contains Ydata Profiling-generated reports detailing the source data provided by the teachers.
- **Root CSV Files**: Includes CSV files with test results for all models used in the project.

### Prediction Flow
1. Predict individual player performance.
2. Use player data to estimate overall team performance.
3. Predict whether a team qualifies for the playoffs based on its performance.

## Required Packages
To run this project, install the following dependencies:

- [joblib 1.3.2](https://joblib.readthedocs.io)
- [pandas 2.1.3](https://pandas.pydata.org/docs/)
- [numpy 1.23.5](https://numpy.org/doc/)
- [scikit-learn 1.3.1](https://scikit-learn.org/stable/)
- [os](https://docs.python.org/3/library/os.html)

The code was developed using **Python 3.10**.

## Usage
To run the project, execute the `main.py` file:
```bash
python main.py
```

To predict the outcome of the 11th season:
```bash
python main.py --predict_11th_year
```
