# Internship-Project-on-South-German-Bank-Credit-Risk

## Introduction
To minimize the risk of non-performing credit loans and optimize the bank's wealth obtained from providing credit loans, a data mining technique can be used. The aim is to build a model using classification to find hidden patterns in the existing lending data and predict whether a person is a good (1) or bad (0) credit risk based on their attributes in the dataset.

## Solution Proposed

To reduce credit risk, it is important to minimize False Positive Rate (FPR) and False Negative Rate (FNR). FPR refers to the percentage of good credits that are incorrectly classified as bad credits. FNR refers to the percentage of bad credits that are incorrectly classified as good credits. Reducing FPR and FNR helps ensure that credit risk is accurately assessed and managed, minimizing the potential for losses.

**Input variables are:**
~~~
1. laufkont = status
2. laufzeit = duration
3. moral = credit_history
4. verw = purpose
5. hoehe = amount
6. sparkont = savings
7. beszeit = employment_duration
8. rate = installment_rate
9. famges = personal_status_sex
10. buerge = other_debtors
11. wohnzeit = present_residence
12. verm = property
13. alter = age
14. weitkred = other_installment_plans
15. wohn = housing
16. bishkred = number_credits
17. beruf = job
18. pers = people_liable
19. telef = telephone
20. gastarb = foreign_worker
~~~

**Output variables:**

`1. kredit = credit_risk`

## Technology Used

1. Python
2. Machine learning algorithms
3. Docker
4. MongoDB

## Technology Used

1. AWS S3
2. AWS EC2
3. AWS ECR
4. Git Actions

## User Interface
The Prediction of Credit Risk Final Model Run in Local Enviornment

1. Main Page :

![Home_Page](https://user-images.githubusercontent.com/19362546/216977537-fa26c220-a699-46bd-a8cd-c9090d60a7f0.PNG)

https://www.youtube.com/watch?v=nvPOUdz5PL4


2. Result Page :

![Result_Page](https://user-images.githubusercontent.com/19362546/216977768-ecd9adca-9343-468b-bd3b-ac2dd935c63a.PNG)

## Deployment Link

Streamlit App : https://vishnu024-south-german-bank-credit-risk-ml-project-app-sgys0a.streamlit.app/

Deployed on AWS also but due to AWS Billing i had terminated the instance on AWS.



## Installtion
The Code is written in Python 3.8.11. If you don't have Python installed you can find it [your link here](https://www.python.org/downloads/). If you are using a lower version of Python you can upgrade using the pip package, ensuring you have the latest version of pip. To install the required packages and libraries, run this command in the project directory after [cloning](https://docs.github.com/en/github/creating-cloning-and-archiving-repositories/cloning-a-repository) the repository.

~~~
Create a Virtual Env with conda create "Your Env name"
~~~
~~~
pip install -r requirements.txt
~~~
~~~
Run app.py file
~~~



## Document

Created High Level Design (HLD) and Low Level Design (LLD) for the project.


