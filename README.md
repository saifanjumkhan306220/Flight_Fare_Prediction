# Flight_Fare_Prediction
🌟 In this project, I harness the power of Random Forest, a robust ML algorithm, to accurately predict flight fares. Here's a sneak peek into the steps involved:

1️⃣ Select subsets of data points and features for each decision tree.
2️⃣ Construct individual decision trees for each sample.
3️⃣ Generate outputs from each decision tree.
4️⃣ Combine outputs through Majority Voting for classification or Averaging for regression.

💡 Why Random Forest? It's not just powerful but also prevents overfitting and provides accurate predictions, making it superior to linear regression and decision trees.

Inorder to deploy the model first build the model.

📁 Files required for deployment:
- `model.py`: ML model.
- `model.pkl`: Pickle file of the ML model.
- `app.py`: Flask application.
- `Index.html`: Template for the application.
- Dataset: Data used to build the ML model.
