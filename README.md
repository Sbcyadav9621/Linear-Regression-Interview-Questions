# Linear-Regression-Interview-Questions
    Linear Regression is a statistical method used to finding the existenceof an association relationship between a dependent and indpendent variables. It is a supervised learning algorithm. We can establish that change in the value of the dependent variable associated with change in the value of features. A regression model requires the knowledge of both indpendent and dependent variables in the training dataset.
    Linear regression implies that the mathematical function is linear with respect to regression parameters.
    Linear regression model finds the relatinship between a target variable and one or more input variables by fitting the a linear equation to observed data.
    Linear regressin aims to find the best straight line that fits the data by minimizing the differences between the actual values andthe predicted values. 
    The measure we use to quanitfy this difference is called the cost function, typically the Mean Squared Error,
    The MSE sums up the squared differences between actual and predicted outputs. 
    The goal is to adjust the model parameters to minimize this cost. 
    It is an iterative process.
    It uses optimization technique like Gradient Descent to updating the parameters until the error is minimized. 
    The goal is to find a linear function that best predicts the dependent variable based on the independent variables.
    y = w0 + w1X1 + w2X2 +.....+wnXn
      - y is the dependent varaible(Target)
      - x1,x2,..xn are the feature values
      - w1,w2,wn are the weights(coefficients)
      - w0 is the bias(intercept) term
    The goal is to determine the weights that best fits the data. we achieve through Gradient Descent method, 
    The best fit line has minimum loss, the line passes through data points as close as possible.
  # How we find best fit line
  The objective of linear regression is to find a line that best explains or predicts the data. 'Best' here means the line that makes the predictions as close as possible to the actual data points.
  
  1. start with initial guess
  2. we randomly chose parameter values.
  3. measure the error using cost function
  4. look at the direction in which the error decreases the fastest. this is called 'gradient' or 'slope' of the error function.
  5. If Increasing a parameter makes the error worse, we decrease it.
  6. If decreasing a parameter makes the error better, we decrease it accordingly.
  8. Next adjust the line to make better. Which is done by adjusting the model weights and bias term. update the parameter values
  9. now measure the error again.
  10. Repeat the process, until the error is very small or stops decreasing significatnly. 
# Loss Function
  The loss function quantifies the error for a single data point, typically the squared difference between the predicted and actual value. In LInear regression we use MSE(Mean Squared Error) as loss function or cost function

# Assumptions of Linear Regression 
  - Linearity:
      The relationship between independent variables and the dependent variable is linear.
  - Independence:
      Observations are independent of each other.
  - Homoscedasticity:
      The variance of residuals is constant across all levels of independent variables.
  - Normality of errors:
      The residuals are approximately normally distributed.
  - No Multicollinearity:
      Independent variables are not highly correlated with each other, high multicollinearity can destabalize coefficient estimates.
# Model Evaluation metrics
  1. Mean Squared Erro(MSE):
     Measures average squared differences between observed and predicted value
  2. Root Mean Squared Error(RMSE):
     Square of MSE, interpretable in the same units as the target.
  3. Mean Absolute Error:
     Average of absolute deviations, less sensitive to outliers.
  4. R-squared:
     Represents the proportion of varaince in the dependent variable explained by the model, ranging from0 to 1. It indicates the goodness-of-fit.
# Limitations of Linear Regression
  1. It assumes linear relationships.
  2. Sensitive to outliers, which can disproportionately influence the model

# Important Interview questions

## How can you explain linear regression in laymen term?
    Linear regression is a way of drawing the best possible straight line through some data points to show the relationship between two things. It helps us understand and predict how changing one thing might affect another.

## What is regression analysis?
  Regression analysis is a statistical and machine learning technique used to understand the releationship between a dependent(target) variable and one or more independent variables and to predict continuous outcomes.

## What is meant by the term "linear regression"?
  Linear regression is a supervised learning and statistical method used to model the linear relationship between a dependent variable and one or more independent variables by fitting straight line equation to the data.

## What is coefficient of Determination?
  The coefficient of determination, denoted as R2, measures how well a regression model explains the variability of the dependent variable.

## What is Multi-collinearity?
  Multicollinearity occurs when two or more independent variables in a regression model are highly correlated with each other.

## How do you detect Multi-collinearity and how will you remove it?
How to detect Multi-collinearity
1. Correlation Matrix
   Check pairwise correlations between independent variables.
2. VIF
   Measures how much the variance of a coefficient is inflated due to correlation.
How to remove or handle multicollinearity
1. Remove Highly Correlated Variables
   Drop one of the correlated features based on domain knowledge.
2. Regularization Techniques
   Lasso Regression (L1): can shrink some coefficients to zero
   Ridge Regression (L2): Reduces coefficients magnitude.
3. PCA
   Convert correlated features into uncorrelated components.
   Useful when prediction is more important than interpretability.

# What is Variance Inflation Factor(VIF)?
  VIF tells us how much a regression coefficient is inflated because an independent variable is highly correlated with other independent variables.
  VIF measures the amount of multicollinearity in a regression model.

# Why is Linear Regression called "Linear"?
  Linear regression is called linear because the model is linear in its parameters(coefficients), not necessarily in the input variables.
  The dependent variable is expressed as a linear combination of coefficients.

# Can you define regression in layman terms?
    Regression is a method used to predict a value by looking at how it changes when other related things change.
# Why do you require regression?
    We require regression to understand relationships in data and to predict numerical outcomes.
  1. Prediction
     To predict continuous values like price, sales, revenue, salary, or demand.
  2. Understand Impact
     To know how much one factor affects another.
     (e.g how much sales increase when advertising spend increase).
  3. Decision Making
     Helps business make data-driven decisions based on trends and patterns.
  4. Trend Analysis
     To identify and quantify trendsd in historical data.
  5. What-if analysis
     To answer questions like:
     "What happens to profit if cost increases by 1 unit?
  We require regression to predict values and understand how different factors influence an outcome.

# Can you explain in what different situations you have used linear regression?
  I have used linear regression mainly in situations where the target variable is continuous, the relationship is approximately linear, and interpretability is important.
  1. Sales/Revenue Forecasting
     I used linear regression to predict monthly sales based on variables like advertising spend.
  2. Price or Cost Estimation
     I applied linear regression to estimate prices or costs, such as:
     - House price vs area, location, number of rooms.
     - Manufacturing cost vs raw material and labor cost.
  3. Financial Analysis
     I used linear regression to:
     - Model returns vs risk factors.
     - Estimate impact of interest rates or inflation on portfolio returns.
     
# Explain the major steps inlinear regression model building?
  Major steps in Linear Regression Model Building
  1. Problem Understanding
     - Clearly define the business objective.
     - Identify the dependent variable and independent variables
  2. Data collection
     - Gather relevant historical data from reliable sources.
  3. Exploratory Data Analysis
     - Understand data distribution.
     - Check relationships between variables.
     - Identify missing values and outliers.
  4. Data preprocessing
     - Handling missing value.
     - Treat outliers
     - Encode categorical variables.
     - Scale features if required.
  5. Check Assumptions of Linear Regression.
     - Linearity
     - Independence of errors
     - Homoscedasticity.
     - Normal of residuals
     - No multicollinearity(using VIF)
  6. Train-Test Split
     - Split data into training and testing sets to evaluate model performance.
  7. Model building
     - Fit the linear regression model using methods like OLS.
  8. Model Evaluation
     - Evaluate using R2, Adjusted R2, RMSE, MAE.
  9. Model Improvement
      - Feature selection.
      - Remove multicollinearity
      - Apply transformations if needed.
  10. Model Interpretation & Deployment
      - Interpret coefficients.
      - Use the model for prediction and business decisio- making.

# How do you test goodness of fit?
  1. Coefficient of Determination
     Shows the percentage of variance explained by the model.
     Higher R2 better model or better fit.
  2. Adjusted R2
     Adjusted R2 for the number of predictors.
     Useful when comparing models with multiple variables.
  3. Error Metrics
     MSE,RMSE,MAE
  4. Residual Analysis
     Plot residuals vs predicted values.
     Residuals should be randomly scattered around zero



