### Used car predictions + EDA
So, I had a task to analyze a ready dataset and build a predictive model based on it. It was also necessary to place the ready solution on the Streamlint platform.

1. As a ready dataset we chose [data](https://raw.githubusercontent.com/evgpat/edu_stepik_from_idea_to_mvp/main/datasets/cars.csv) on the prices of used cars with different characteristics.

2. After selecting the dataset, it was analyzed.

3. Next I had to build the model. Since the target variable was the price of the car, the model had to perform a regression task. I chose XGBoostRegressor as the base solution. Then good hyperparameters were found using GridSearchCv, which resulted in good prediction quality (r2_score = 0.96)

4. Finally, a web interface based on the predictive model was implemented - https://elmowx.streamlit.app/ .

main.py - Executive file for Streamlint

model.py - File for building prediction for Streamlint

data - Folder with image file and model file
