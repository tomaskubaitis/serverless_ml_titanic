def titanic_prep():
    # import necessary packages
    import pandas as pd
    import numpy as np

    # import raw titanic dataset nand inspect
    titanic_raw = pd.read_csv("https://raw.githubusercontent.com/tomaskubaitis/serverless_ml_titanic/assets/titanic.csv")
    # print(titanic_raw)

    # drop redundant columns and columns w/o predictive power
    titanic_df = titanic_raw.drop(columns={"PassengerId","Name","Ticket","Cabin"})

    # check for null and NaN values
    # print(f"Nan values in the dataset: \n {titanic_df.isna().sum()}")

    # round up the Age feature where not NA, so infants that are younger than 1 are now 1 for easier and more consistent future application
    titanic_df.loc[titanic_df.Age.notna(),"Age"] = np.ceil(titanic_df.loc[titanic_df.Age.notna(),"Age"])

    # create array of random ages to fill the missing age value with
    random_ages = np.random.randint(titanic_df.Age.min(), titanic_df.Age.max(),titanic_df.Age.isnull().sum())
    titanic_df.loc[titanic_df.Age.isnull(),"Age"] = random_ages

    # fill missing values for "Embarked" feature with "Unknown" category
    titanic_df.fillna(value = {"Embarked":"Unknown"}, inplace=True)

    # check if missing values are truly replaced
    # print(f"Nan values in the dataset: \n {titanic_df.isna().sum()}")
    # print(titanic_df)

    # turn categorical variables ("Embarked" and "Sex") into numerical variables
    # first get dummies
    dummies_sex = pd.get_dummies(titanic_df.Sex, prefix="Sex")
    dummies_embarked = pd.get_dummies(titanic_df.Embarked, prefix="Embarked")
    titanic_df = pd.concat([titanic_df, dummies_sex, dummies_embarked], axis='columns')
    # drop old categorical columns "Sex" and "Embarked"
    titanic_df.drop(columns={"Sex","Embarked"}, inplace=True)

    # inspect final clean dataset
    # print(titanic_df)

    # return cleaned and prepped titanic dataset
    return titanic_df