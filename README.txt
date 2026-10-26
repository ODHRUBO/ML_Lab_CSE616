CSE 615 Lab Report - LaTeX source
=================================

Compile with XeLaTeX (run twice so the contents and page numbers settle):

    xelatex main_share.tex
    xelatex main_share.tex

VS Code
-------
Open this folder as the workspace in VS Code. Install the recommended
`LaTeX Workshop` extension if VS Code does not install it automatically.
The workspace settings already select XeLaTeX and define a two-pass
`XeLaTeX` recipe. Open `main_share.tex`, then use the LaTeX Workshop build
button or the Command Palette command `LaTeX Workshop: Build LaTeX project`.
The generated PDF can be opened from the LaTeX Workshop PDF viewer.

Contents
    main_share.tex        the report
    chapters_share/       one .tex file per lab
    fonts/                Times New Roman, DejaVu Sans Mono and TeX Gyre Termes Math
    01_KNN/ ... 13_Genetic_Algorithm/   the Python scripts that are listed in the report
    main_share.pdf        the compiled report

Python experiments
-------------------

The experiments use the virtual environment in .venv. The project and all
datasets remain on the E: drive. From the project root, install the packages
once with:

    .\.venv\Scripts\python.exe -m pip install numpy pandas matplotlib

Each experiment has its required CSV file in its own data folder:

    01_KNN/knn.py                         data/wine.csv
    02_Linear_Regression/linear_regression.py  data/california_housing.csv
    03_Gradient_Descent/gradient_descent.py   data/concrete.csv
    04_MLE_MAP/mle_map.py                 data/diabetes.csv
    05_Naive_Bayes/naive_bayes.py         data/sms_spam.csv
    06_Logistic_Regression/logistic_regression.py  data/breast_cancer.csv
    07_Perceptron/perceptron.py           data/iris_binary.csv
    08_MLP/mlp.py                         data/iris.csv
    09_RBF_Network/rbf_network.py         data/seeds.csv
    10_SVM/svm.py                         data/sonar.csv
    11_SVR/svr.py                         data/auto_mpg.csv
    12_Decision_Tree_ID3/id3.py           data/mushrooms.csv
    13_Genetic_Algorithm/genetic_algorithm.py  data/breast_cancer.csv

Run one experiment from the project root, for example:

    .\.venv\Scripts\python.exe .\07_Perceptron\perceptron.py

Each script writes its figures to its figures folder and its text output to
results.txt in the corresponding lab folder. The CSV files are already
prepared in the expected format; do not run the scripts from another working
directory.

The cover-page details (teacher's name, designation, date) are at the top of main_share.tex.
If you use Overleaf, upload this whole folder and set the compiler to XeLaTeX.
