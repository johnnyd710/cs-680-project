# arcelormittal-project
A Machine Learning project, through Waterloo CS 680: Introduction to Machine Learning. Provides a predictive maintenance solution to industrial fans at a steel making plant.


## dataset information
Furnance 1 1-1-17 - furnance 1 fans 2 weeks before failure (earliest)
Furnance 1 1730 to .. - furnance 1 day of failure (the 14th)
Furnance 1 1-16-17 - furnance 1 after repair

## target to beat: one week in advance is when they were noticing it

# how to run:

in src/ folder:

python3 make_data.py (name of data set) (interval for feature engineering)
python3 make_som.py (which data set to train)

* note: data is protected for confidentially
