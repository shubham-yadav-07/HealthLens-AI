# Data folder

`pima_diabetes.csv` is the **Pima Indians Diabetes Database** (768 rows, 8 features + `Outcome`).

- **Original source:** National Institute of Diabetes and Digestive and Kidney Diseases; described in Smith et al. (1988).
- **Public copies:** UCI Machine Learning Repository, Kaggle ("Pima Indians Diabetes Database"), and the mirror used by this project:
  `https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv` (no header row; the project adds the column names).
- **No synthetic or invented records are used anywhere in this project.**

## If the file is missing
```bash
python -m src.data_loader      # downloads the file and saves it here with a header row
```
If you have no internet access, download the CSV manually from Kaggle, save it as `data/pima_diabetes.csv`, and make sure the header is:

`Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age,Outcome`

(A file with the 9 numeric columns and no header is also accepted by the upload page.)

## Known limitations
Single population (Pima women aged 21+), 768 records, zeros that mean "not measured" in Glucose, BloodPressure,
SkinThickness, Insulin and BMI, collected in the 1980s. Details are in the app (Dataset Explorer -> Source and limitations).
