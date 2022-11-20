import gradio as gr
import numpy as np
from PIL import Image
import requests

import hopsworks
import joblib
import pandas as pd

project = hopsworks.login()
fs = project.get_feature_store()


mr = project.get_model_registry()
model = mr.get_model("titanic_modal", version=1)
model_dir = model.download()
model = joblib.load(model_dir + "/titanic_model.pkl")


def titanic(pclass,age,sibsp,parch,fare,sex,embarked):
    input_list = []
    input_list.append(pclass)
    input_list.append(age)
    input_list.append(sibsp)
    input_list.append(parch)
    input_list.append(fare)
    if sex == "male":
        input_list.append(0)
        input_list.append(1)
    elif sex == "female":
        input_list.append(1)
        input_list.append(0)
    
    if embarked == "C":
        input_list.append(1)
        input_list.append(0)
        input_list.append(0)
        input_list.append(0)
    elif embarked == "Q":
        input_list.append(0)
        input_list.append(1)
        input_list.append(0)
        input_list.append(0)
    elif embarked == "S":
        input_list.append(0)
        input_list.append(0)
        input_list.append(1)
        input_list.append(0)
    elif embarked == "Unknown":
        input_list.append(0)
        input_list.append(0)
        input_list.append(0)
        input_list.append(1)
    
    # input_df = pd.DataFrame(data=input_list, columns = ['Pclass', 'Age', 'SibSp', 'Parch', 
    #                                                     'Fare', 'Sex_female','Sex_male', 
    #                                                     'Embarked_C', 'Embarked_Q', 'Embarked_S',
    #                                                     'Embarked_Unknown'])
    # 'res' is a list of predictions returned as the label.
    res = model.predict(np.asarray(input_list).reshape(1, -1))
    # We add '[0]' to the result of the transformed 'res', because 'res' is a list, and we only want 
    # the first element.
    if res[0] == 1:
        res_str = "survivor"
    else:
        res_str = "victim"
    passenger_url = "https://raw.githubusercontent.com/tomaskubaitis/serverless_ml_titanic/assets/" + res_str + ".png"
    img = Image.open(requests.get(passenger_url, stream=True).raw)            
    return img
    # if res[0] == 1:
    #     return "The passenger is predicted to be a survivor."
    # else:
    #     return "The passenger is predicted to be a victim."
        
demo = gr.Interface(
    fn=titanic,
    title="Titanic Passenger Predictive Analytics",
    description="Experiment with passenger data to predict whether the passenger is a survivor or not.",
    allow_flagging="never",
    inputs=[
        gr.inputs.Number(default=2, label="Passenger class (choose from either 1, 2 or 3)"),
        gr.inputs.Number(default=30, label="Age in full years (if child younger than 1 round up to 1)"),
        gr.inputs.Number(default=1, label="Number of siblings or spouses"),
        gr.inputs.Number(default=0, label="Number of parents or children"),
        gr.inputs.Number(default=100, label="Fare (cost between 0 and 513)"),
        gr.inputs.Textbox(default="male", label="Sex (choose from either male or female)"),
        gr.inputs.Textbox(default="Unknown", label="Embarked (choose from either C, Q, S or Unknown)"),
        ],
    # outputs=gr.outputs.Textbox())
    outputs=gr.Image(type="pil"))

demo.launch()