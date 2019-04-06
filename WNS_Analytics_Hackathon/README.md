<b>Problem Statement</b>

Your client is a large MNC and they have 9 broad verticals across the organisation. One of the problem your client is facing is around identifying the right people for promotion (only for manager position and below) and prepare them in time. Currently the process, they are following is:

They first identify a set of employees based on recommendations/ past performance.
Selected employees go through the separate training and evaluation program for each vertical. These programs are based on the required skill of each vertical.
At the end of the program, based on various factors such as training performance, KPI completion (only employees with KPIs completed greater than 60% are considered) etc., employee gets promotion
For above mentioned process, the final promotions are only announced after the evaluation and this leads to delay in transition to their new roles. Hence, company needs your help in identifying the eligible candidates at a particular checkpoint so that they can expedite the entire promotion cycle. 

<b>Approach</b>

1. Tried MICE imputation for Null values, median and mode for numerical and categorical eventually worked better.
2. Label Encoding and OneHot Encoding for Categorical and Numerical Variables.
3. Robust Scaling for all Variables with outliers.
4. Eventually applied multiple models out of which CatBoost worked the best . Rest lead to overfitting and some lead to models which were not generalized .
