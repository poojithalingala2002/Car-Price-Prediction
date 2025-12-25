<h1 align="center">🚗 Car Price Prediction System</h1>

<p align="center">
An end-to-end Machine Learning project that predicts the selling price of a used car based on historical data and vehicle attributes.
</p>

<hr>

<h2>📌 Project Overview</h2>
<p>
This project aims to predict the <b>selling price of used cars</b> using Machine Learning techniques.
It covers the complete lifecycle of a data science project including:
</p>
<ul>
  <li>Data preprocessing</li>
  <li>Feature engineering</li>
  <li>Outlier handling & transformation</li>
  <li>Categorical encoding</li>
  <li>Feature scaling</li>
  <li>Model training & evaluation</li>
  <li>Model deployment using Flask</li>
</ul>

<hr>

<h2>📊 Dataset Description</h2>
<p>The dataset contains historical car sale information with the following features:</p>

<table border="1" cellpadding="6">
<tr><th>Feature</th><th>Description</th></tr>
<tr><td>Car_Name</td><td>Name of the car</td></tr>
<tr><td>Year</td><td>Manufacturing year</td></tr>
<tr><td>Present_Price</td><td>Current showroom price (in lakhs)</td></tr>
<tr><td>Kms_Driven</td><td>Total kilometers driven</td></tr>
<tr><td>Fuel_Type</td><td>Petrol / Diesel / CNG</td></tr>
<tr><td>Seller_Type</td><td>Individual / Dealer</td></tr>
<tr><td>Transmission</td><td>Manual / Automatic</td></tr>
<tr><td>Owner</td><td>Number of previous owners</td></tr>
<tr><td><b>Selling_Price</b></td><td><b>Target variable</b></td></tr>
</table>

<hr>

<h2>⚙️ Project Architecture</h2>

<pre>
Car_Price_Prediction/
│
├── data.csv
├── main.py
├── missing_value_handling.py
├── variable_trans_out_handle.py
├── cat_to_num.py
├── feature_select.py
├── data_scaling.py
├── regression.py
├── log_code.py
├── app.py
├── templates/
│   └── index.html
├── requirements.txt
├── reg_model.pkl
├── scaler.pkl
├── target_encode.pkl
└── README.md
</pre>

<hr>

<h2>🔄 End-to-End Workflow</h2>

<h3>1️⃣ Data Loading & Splitting</h3>
<ul>
  <li>Dataset is loaded from <code>data.csv</code></li>
  <li>Target variable: <b>Selling_Price</b></li>
  <li>Data split into training (80%) and testing (20%)</li>
</ul>

<h3>2️⃣ Missing Value Handling</h3>
<p>
Missing values are handled using a <b>random sample imputation</b> technique,
where missing values are replaced by randomly sampled existing values from the same column.
</p>

<h3>3️⃣ Feature Engineering</h3>
<ul>
  <li>Extracts <b>Brand</b> from <code>Car_Name</code></li>
  <li>Handles multi-word brands like <i>Royal Enfield</i></li>
  <li>Drops high-cardinality <code>Car_Name</code> column</li>
</ul>

<h3>4️⃣ Outlier Handling & Transformation</h3>
<ul>
  <li>Year is converted to <b>car age</b></li>
  <li>Log transformation applied to skewed numeric features</li>
  <li>Quantile capping (1%–99%) used to control extreme outliers</li>
  <li>KDE and boxplots saved before & after transformation</li>
</ul>

<h3>5️⃣ Categorical Encoding</h3>
<ul>
  <li>Fuel_Type, Seller_Type, Transmission → Label Encoding</li>
  <li>Brand → <b>Target Encoding</b></li>
</ul>

<h3>6️⃣ Feature Scaling</h3>
<ul>
  <li>StandardScaler applied to numerical features</li>
  <li>Scaler object saved for inference</li>
</ul>

<h3>7️⃣ Model Training</h3>
<ul>
  <li>Algorithm used: <b>Linear Regression</b></li>
  <li>Evaluation Metrics:
    <ul>
      <li>R² Score</li>
      <li>Mean Squared Error</li>
    </ul>
  </li>
  <li>Trained model saved as <code>reg_model.pkl</code></li>
</ul>

<hr>

<h2>🧠 Machine Learning Model</h2>
<ul>
  <li>Algorithm: Linear Regression</li>
  <li>Target Variable: Selling_Price</li>
  <li>Model trained on transformed & scaled features</li>
  <li>Artifacts saved for deployment</li>
</ul>

<hr>

<h2>🌐 Web Application (Flask)</h2>

<h3>Backend</h3>
<ul>
  <li>Flask framework used</li>
  <li>Loads trained model, scaler & encoder</li>
  <li>Ensures preprocessing consistency with training pipeline</li>
</ul>

<h3>Frontend</h3>
<ul>
  <li>Simple HTML form</li>
  <li>User inputs car details</li>
  <li>Displays predicted selling price</li>
</ul>

<hr>

<h2>🚀 How to Run the Project</h2>

<pre>
pip install -r requirements.txt
python main.py
python app.py
</pre>

<p>Open browser and go to:</p>
<pre>http://127.0.0.1:5000/</pre>

<hr>

<h2>📦 Requirements</h2>
<p>All dependencies are listed in <code>requirements.txt</code></p>

<hr>

<h2>📈 Future Improvements</h2>
<ul>
  <li>Try advanced models (Random Forest, XGBoost)</li>
  <li>Hyperparameter tuning</li>
  <li>Add more vehicle features</li>
  <li>Deploy on cloud (AWS / Render / Heroku)</li>
</ul>

<hr>

<h2>👤 Author</h2>
<p>
<b>Bala venu Balineni</b><br>
Machine Learning & Data Science Enthusiast
</p>

<hr>

<p align="center">
⭐ If you like this project, give it a star on GitHub!
</p>
