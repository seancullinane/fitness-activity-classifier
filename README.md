# 🏃‍♂️ Fitness Activity Classifier (HAR)

Classifies human activities (walking, sitting, standing, laying, stairs) from wearable sensor features (UCI HAR).  
Tech: Python, scikit-learn, RandomForest.

## Quick start
```bash
python -m venv venv && .\venv\Scripts\Activate
pip install -r requirements.txt
python src/get_data.py
python src/train.py

## 📂 Project Structure

fitness-activity-classifier/
├── data/
│ ├── raw/ # downloaded dataset (ignored in Git)
│ └── processed/ # processed features
├── notebooks/
│ └── har_model.ipynb # Jupyter notebook
├── src/
│ ├── get_data.py # script to download/extract dataset
│ └── train.py # script to train model & save outputs
├── tests/
│ └── test_train.py # simple accuracy test
├── outputs/ # confusion matrix, trained model
├── requirements.txt # Python dependencies
├── LICENSE # license text (MIT)
├── .gitignore # files/folders Git ignores
└── README.md # this file

## Results
Achieved ~90–95% accuracy on the UCI HAR test set.

![Confusion Matrix](outputs/confusion_matrix.png)

## Project Structure
fitness-activity-classifier/
├── data/
│ ├── raw/ # downloaded dataset (ignored in Git)
│ └── processed/
├── notebooks/
│ └── har_model.ipynb
├── src/
│ ├── get_data.py
│ └── train.py
├── tests/
│ └── test_train.py
├── outputs/
│ └── confusion_matrix.png # tracked (model artifact ignored)
├── requirements.txt
├── LICENSE
├── .gitignore
└── README.md
