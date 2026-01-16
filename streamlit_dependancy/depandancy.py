import streamlit as st


params = {'legend.fontsize': 'small',
          'figure.figsize': (12, 8),
          'axes.labelsize': 'small',
          'axes.titlesize': 'small',
          'xtick.labelsize': 'small',
          'ytick.labelsize': 'small'}


def font_size(size: int, label: str):
    s = f"<p style='font-size: {size}px;'>{label}</p>"
    st.markdown(s, unsafe_allow_html=True)


def predict(df, model, header: str):

    font_size(32, header)

    client_id = st.selectbox('Client id', df['client_id'].sample(frac=0.001, random_state=3))

    row = df.loc[df['client_id'] == client_id, :]

    with st.form('input'):

        button = st.form_submit_button()

    if button:

        prediction = model.predict(row)[0]

        st.write(row)

        text = f"Prediction: {prediction}"
        st.write(text)


