# Coreset Finding Using PDF Approximation

This project was completed as part of the "Tabular Data Science" course (89547-01) at Bar-Ilan University in 2024. We introduce a novel sampling technique designed to optimize model training processes, such as those used with XGBoost. Our approach aims to minimize the feature-wise distribution differences between the original training dataset and the sample. This is achieved through the use of a genetic algorithm, enhancing the efficiency and effectiveness of the training phase.

## Authors

- [@ghsumhubh](https://www.github.com/ghsumhubh)
- [@idansi98](https://www.github.com/idansi98)



## Supported Versions

Our project has been tested on the following software configurations:

### For Long Runs
- **Operating System**: Rocky Linux 9.3
- **Python**: 3.9.18
- **Packages**:
  - matplotlib: 3.8.1
  - numpy: 1.26.4
  - pandas: 2.1.2
  - scikit-learn: 1.3.2
  - scipy: 1.13.0
  - shap: 0.45.0
  - xgboost: 2.0.1

### For Notebooks and Short Runs
- **Operating System**: Windows 11
- **Python**: 3.11.4
- **Packages**:
  - matplotlib: 3.7.2
  - numpy: 1.24.3
  - pandas: 2.0.3
  - scikit-learn: 1.3.0
  - scipy: 1.11.4
  - seaborn: 0.13.2
  - shap: 0.45.0
  - xgboost: 2.0.0
## Running the Demo

To run the demo, follow these simple steps:

1. **Clone the Entire Project**: Ensure that you have cloned the entire project repository to your local machine or development environment.

2. **Open `demo.ipynb`**: Launch the `demo.ipynb` notebook. This notebook contains all the necessary code cells to execute the demo.

3. **Run All Cells**: Execute all cells in the notebook in sequence. This will run through the demo process as configured.

**Important Note**: It is recommended to use the software configurations mentioned previously as they are tailored to the notebook. Intermediary results are saved as pickle files, which may behave differently across operating systems due to serialization differences.

By following these instructions, you should be able to smoothly run the demo without any issues.
## Project Files

This section outlines the key files within our project and their specific purposes:

- **full_simulation.py**: Executes a full simulation using a specified dataset.

- **get_datasets_stats.ipynb**: Generates graphical representations for inclusion in reports.

- **get_online_datasets.ipynb**: Facilitates the downloading of datasets stored online.

- **post_simulation_analysis.py**: Conducts analysis following the completion of simulations.

- **process_datasets.ipynb**: Handles the preprocessing of datasets.

- **env.yml** a conda enviroment configuration file for the notebooks.

Each of these files is designed to perform distinct tasks within the workflow of our simulation and analysis process.
