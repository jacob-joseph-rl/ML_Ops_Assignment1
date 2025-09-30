Boston Housing MLOps Assignment
Installation
1.	Clone the repository:
git clone <https://github.com/jacob-joseph-rl/ML_Ops_Assignment1.git>

2.	Create a virtual environment (recommended):
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

3.	Install dependencies:
pip install -r requirements.txt

The main dependencies are:
o	pandas
o	numpy

Running the Code
To load and preview the Boston Housing dataset:
python src/data/load_data.py

This will print the first few rows of the dataset to confirm successful loading.

Notes
•	The dataset is loaded manually from the StatLib archive due to deprecation in scikit-learn.
•	All code is modular and ready for further steps (model training, evaluation, packaging).
Troubleshooting
•	If you encounter issues with missing packages, ensure your virtual environment is activated and run pip install -r requirements.txt again.
•	For Python version compatibility, use Python 3.8 or newer.