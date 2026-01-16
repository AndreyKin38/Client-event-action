import streamlit as st
import pandas as pd

import matplotlib.pyplot as plt

from streamlit_dependancy import predict, params

from model import event_action_pipeline


plt.rcParams.update(params)

df_full = pd.read_parquet("preprocessed_data/df_full.parquet")
target = pd.read_parquet("preprocessed_data/target.parquet")

model = event_action_pipeline()['model']


st.title("Client dataset")

predict(df_full, model, 'Event action prediction')








