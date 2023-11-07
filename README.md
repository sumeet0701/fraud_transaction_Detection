
# Fraud Transaction Detection and Prediction 

# problem Statement:
There is a lack of publicly available datasets on financial services especially in the emerging mobile money transactions domain. Financial datasets are important to many researchers and in particular to us performing research in the domain of fraud detection. Part of the problem is the intrinsically private nature of financial transactions, that leads to no publicly available datasets.
We present a synthetic dataset generated using the simulator called PaySim as an approach to such a problem. PaySim uses aggregated data from the private dataset to generate a synthetic dataset that resembles the normal operation of transactions and injects malicious behavior to later evaluate the performance of fraud detection methods.

**PaySim simulates mobile money transactions based on a sample of real transactions extracted from one month of financial logs from a mobile money service implemented in an African country. The original logs were provided by a multinational company, who is the provider of the mobile financial service which is currently running in more than 14 countries all around the world.**

# About Dataset:
  **Dataset Link** : https://www.kaggle.com/datasets/ealaxi/paysim1
1. step - maps a unit of time in the real world. In this case 1 step is 1 hour of time. Total steps 744 (30 days simulation).
2. type - CASH-IN, CASH-OUT, DEBIT, PAYMENT and TRANSFER.
3. amount - amount of the transaction in local currency.
4. nameOrig - customer who started the transaction.
5. oldbalanceOrg - initial balance before the transaction.
6. newbalanceOrig - new balance after the transaction.
7. nameDest - customer who is the recipient of the transaction
8. oldbalanceDest - initial balance recipient before the transaction. 
9. newbalanceDest - new balance recipient after the transaction.
10. isFraud - This is the transactions made by the fraudulent agents inside the simulation. In this specific dataset the fraudulent behavior of the agents aims to profit by taking control or customers accounts and try to empty the funds by transferring to another account and then cashing out of the system.
11. isFlaggedFraud - The business model aims to control massive transfers from one account to another and flags illegal attempts. An illegal attempt in this dataset is an attempt to transfer more than 200.000 in a single transaction.



## License

[MIT](https://choosealicense.com/licenses/mit/)


## 🛠 Skills
●	Python Programming

    ○	OOPS
    ○	Modularity

●	Library used

    ○	Numpy
    ○	Pandas
    ○	Matplotlib
    ○	Seaborn
    ○	Sklearn
●	Machine Learning Algorithms Used

    ○	Logistic Regression 
    ○	Decision Tree Classification
    ○	KNN Classification
    ○	Randon forest Classification
    ○	XGBoot Classifer


●	Machine Learning

    ○	Single Prediction
    ○	Batch Prediction

●	AWS Cloud

●	Mlops
     
     ○ MLFlow for Tracking
     ○ Evidently for Data Drift

●	MongoDB Database

●	Docker

●	Flask API (For Backend Programming)

●	Version: Git 

●	HTML, CSS


## Process Flow
![Fraud Transaction Detection](https://github.com/sumeet0701/US-Visa-Prediction/assets/63961794/12524a0e-e36f-4254-9a97-661fe5636e80)

## UI:

<img width="935" alt="Screenshot 2023-11-07 120423" src="https://github.com/sumeet0701/fraud_transaction_Detection/assets/63961794/7a127516-7392-4b69-b6f0-e3589913cdf5">


## Installation

```bash


```
  Create Git clone
  ```
  git clone {URL of Rep}
  ```
  Create new env using below command:
  ```
  conda create -n env_name python==3.6.9
  ```
  Activate your envirnment
  ```
  conda activate env_name
  ```
  Run your app.py file name
  ```
  python main.py

  ```
  -> Enter your values and predict the single prediction

  -> for the batch prediction load your dataset and predict.


## 🔗 Links
[![Github](https://img.shields.io/badge/GitHub-000?style=for-the-badge&logo=ko-fi&logoColor=white)](https://github.com/sumeet0701/)

[![linkedin](https://img.shields.io/badge/linkedin-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/sumeet-maheshwari/)



## Authors

- [@Maheshwari Sumeet](https://github.com/sumeet0701)

