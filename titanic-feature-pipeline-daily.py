import os
import modal
from titanic_feature_prep import titanic_prep
    
BACKFILL=False
LOCAL=False

if LOCAL == False:
   stub = modal.Stub()
   image = modal.Image.debian_slim().pip_install(["hopsworks==3.0.4","joblib","seaborn","sklearn","dataframe-image"]) 

   @stub.function(image=image, schedule=modal.Period(days=1), secret=modal.Secret.from_name("HOPSWORKS_API_KEY"))
   def f():
       g()


def generate_passenger( survived, pclass_min, pclass_max, age_min, age_max, 
                        sibsp_min, sibsp_max, parch_min, parch_max, fare_min, fare_max,
                        sex_choices, embarked_choices):
    """
    Returns a passenger as a single row in a DataFrame
    """
    import pandas as pd
    import numpy as np

    # max value for randint has to be one higher than the actual max value since randint will otherwise exclude it
    titanic_df = pd.DataFrame({ "Pclass": np.random.randint(pclass_min,pclass_max, size=1).astype(int),
                        "Age": np.random.randint(age_min,age_max, size=1).astype(float),
                        "SibSp": np.random.randint(sibsp_min,sibsp_max, size=1).astype(int),
                        "Parch": np.random.randint(parch_min,parch_max, size=1).astype(int),
                        "Fare": np.random.uniform(fare_min,fare_max, size=1).astype(float),
                        "Sex": np.random.choice(sex_choices, size=1),
                        "Embarked": np.random.choice(embarked_choices, size=1),
                      })

    titanic_df['Survived'] = np.array(survived).astype(int)
    
    # turn categorical variables ("Embarked" and "Sex") into numerical variables
    # first get dummies
    dummies_sex = pd.get_dummies(titanic_df.Sex, prefix="Sex")
    dummies_embarked = pd.get_dummies(titanic_df.Embarked, prefix="Embarked")
    titanic_df = pd.concat([titanic_df, dummies_sex, dummies_embarked], axis='columns')
    
    # create missing binary columns
    # for sex
    for sex in (sex for sex in ["male","female"] if sex != titanic_df.Sex[0]):
      titanic_df[f"Sex_{sex}"] = np.array([0]).astype("uint8")
    # for embarked
    for loc in (loc for loc in ["C","Q","S","Unknown"] if loc != titanic_df.Embarked[0]):
      titanic_df[f"Embarked_{loc}"] = np.array([0]).astype("uint8")
    
    # drop old categorical columns "Sex" and "Embarked"
    titanic_df.drop(columns={"Sex","Embarked"}, inplace=True)

    return titanic_df

def get_random_passenger():
    """
    Returns a DataFrame containing one random passenger
    """
    import pandas as pd
    import random

    survivor_df = generate_passenger    (1, pclass_min=1, pclass_max=4, age_min=1, age_max=81,
                                            sibsp_min=0, sibsp_max=5, parch_min=0, parch_max=6,
                                            fare_min=0, fare_max=513, sex_choices=["male","female"],
                                            embarked_choices=["C","Q","S","Unknown"])
    victim_df = generate_passenger      (0, pclass_min=1, pclass_max=4, age_min=1, age_max=78,
                                            sibsp_min=0, sibsp_max=9, parch_min=0, parch_max=7,
                                            fare_min=0, fare_max=263, sex_choices=["male","female"],
                                            embarked_choices=["C","Q","S"])

    # randomly pick one of these 2 and write it to the featurestore
    pick_random = random.uniform(0,2)

    if pick_random >= 1:
        titanic_df = survivor_df
        print("Survivor added")
    else:
        titanic_df = victim_df
        print("Victim added")
    print(titanic_df)

    return titanic_df


def g():
    import hopsworks
    import pandas as pd

    project = hopsworks.login()
    fs = project.get_feature_store()

    if BACKFILL == True:
        titanic_df = titanic_prep()
    else:
        titanic_df = get_random_passenger()

    titanic_fg = fs.get_or_create_feature_group(
        name="titanic_modal",
        version=1,
        primary_key=["Pclass","Age","SibSp","Parch","Fare","Sex_female","Sex_male","Embarked_C","Embarked_Q","Embarked_S","Embarked_Unknown"], 
        description="Titanic passenger dataset")
    titanic_fg.insert(titanic_df, write_options={"wait_for_job" : False})

if __name__ == "__main__":
    if LOCAL == True :
        g()
    else:
        stub.deploy("feature_pipeline_daily")
