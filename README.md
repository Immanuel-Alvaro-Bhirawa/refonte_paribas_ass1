# refonte_paribas_ass1

This project demonstrates a complete data science workflow using the France FCHI CAC 40 dataset. The workflow includes data cleaning, exploratory data analysis (EDA), model building, and visualization.

## Project Structure

```
project-name/
├── LICENSE
├── README.md
├── requirements.txt
├── data/
│   ├── processed/
│   │   ├── CAC 40 Historical Data.csv
|   |   └── CAC 40 History Data.xlsx
│   └── raw/
│       ├── CAC 40 Historical Data - Raw.csv
|       └── CAC 40 Historical Data.xslx
├── notebooks/
│   ├── 01_Data_Cleaning.ipynb
│   ├── 02_EDA.ipynb
│   ├── 03_Model_Building.ipynb
│   ├── 04_Visualisations.ipynb
│   ├── CAC 40 Historical Data.csv
│   └── CAC 40 Historical Data.xslx
├── results/
│   ├── FCHI_CAC_40_All_Plots.png
│   ├── FCHI_CAC_40_General_Plot.png
|   └── FCHI_CAC_40_Seaborn_Pairplot.png
└── scripts/
    ├── 01_Data_Cleaning_with_functions.py
    ├── 01_Data_Cleaning.py
    ├── 02_EDA_functions.py
    ├── 02_EDA.py
    ├── 03_Model_Building_functions.py
    ├── 03_Model_Building.py
    ├── 04_Visualisations_functions.py
    └── 04_Visualisations.py
```

## Setup Instructions

### Create Virtual Environment

1. **Create Virtual Environment**:
    ```bash
    python -m venv .venv
    ```

2. **Activate Virtual Environment**:
    - **Windows**:
        ```bash
        .venv\Scripts\activate
        ```
    - **macOS and Linux**:
        ```bash
        source .venv/bin/activate
        ```

3. **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

## Running Scripts and Notebooks

1. **Navigate to Project Directory**:
    ```bash
    cd path/to/project-name/
    ```

2. **Run Scripts**:
    - **Data Cleaning**:
        ```bash
        python scripts/01_data_cleaning.py
        ```
    - **EDA**:
        ```bash
        python scripts/02_EDA.py
        ```
    - **Model Building**:
        ```bash
        python scripts/03_model_building.py
        ```
    - **Visualization**:
        ```bash
        python scripts/04_visualisation.py
        ```

3. **Run Jupyter Notebooks**:
    ```bash
    jupyter notebook
    ```
    Open the desired notebook (e.g., `01_Data_Cleaning.ipynb`, `02_EDA.ipynb`, etc.).

## Execution Order

Run the notebooks in the following order:
1. `01_Data_Cleaning.ipynb`
2. `02_EDA.ipynb`
3. `03_Model_Building.ipynb`
4. `04_Visualisations.ipynb`

## Requirements

- Python 3.6+
- pandas
- seaborn
- matplotlib
- scikit-learn
