import streamlit as st
import pickle
import pandas as pd
import base64
import time

pickle_file = pickle.load(open('ML_Streamlit.pkl','rb'))


class FileDownloader(object):

    def __init__(self, data, filename='Predicted_File', file_ext='txt'):
        super(FileDownloader, self).__init__()
        self.data = data
        self.filename = filename
        self.file_ext = file_ext

    def download(self):
        b64 = base64.b64encode(self.data.encode()).decode()
        new_filename = "{}.{}".format(self.filename, self.file_ext)
        st.markdown("#### Download File ###")
        href = f'<a href="data:file/{self.file_ext};base64,{b64}" download="{new_filename}">Click Here!!</a>'
        st.markdown(href, unsafe_allow_html=True)


st.title("Loan Sanction App")

menu = ["Dataset"]
choice = st.sidebar.selectbox("Menu", menu)

if choice == "Image":
    st.subheader("Image")

elif choice == "Dataset":
    st.subheader("Dataset")
    data_file = st.file_uploader("Upload CSV", type=["csv"])

    if data_file is not None:
        file_details = {"filename": data_file.name, "filetype": data_file.type, "filesize": data_file.size}

        st.write(file_details)
        df = pd.read_csv(data_file)

        st.dataframe(df)

        columns = list(pickle_file['Selected_Columns'])
        #columns.remove('TARGET')
        median_values = pickle_file['Median']
        #del median_values['TARGET']
        mode_values = pickle_file['Mode']
        encoded_columns = list(pickle_file['Encoded_Columns'])
        #encoded_columns.remove('TARGET')
        #encoded_columns.remove('NAME_INCOME_TYPE_Unemployed')

        scaler = pickle_file['Scaler']

        model = pickle_file['Model']

        data = df[columns]

        for i in median_values.keys():
            data[i].fillna(median_values[i], inplace=True)

        for j in mode_values.keys():
            data[j].fillna(mode_values[j][0], inplace=True)

        X_Categorical = data.select_dtypes(exclude=['int64', 'float64']).copy()
        application_data = pd.get_dummies(data, columns=X_Categorical.columns)

        #encoded_columns = set(encoded_columns)
        #test_encoded_columns = set(application_data.columns)

        final_data = pd.DataFrame()

        for k in encoded_columns:
            if k in application_data.columns:
                final_data[k] = application_data[k]
            else:
                final_data[k] = 0

        Data_Final = scaler.transform(final_data)
        Data_Final = pd.DataFrame(Data_Final, columns=encoded_columns)

        #Data_Final = Data_Final.drop(["NAME_INCOME_TYPE_Unemployed"], axis="columns")


        y_pred = model.predict_proba(Data_Final)[:, 1]

        df_test = pd.DataFrame(y_pred, columns=['probability'])

        df_test.loc[df_test['probability'] <= 0.15, 'predictedvalue'] = 'Non-Defaulter'
        df_test.loc[df_test['probability'] > 0.15, 'predictedvalue'] = 'Defaulter'

        result = pd.concat([data, df_test], axis=1, join='inner')

        downloadmenu = ["CSV"]

        choice = st.sidebar.selectbox("Menu", downloadmenu)

        if choice == "Text":
            st.subheader("Text")
            my_text = st.text_area("Your Message")
            if st.button("Save"):
                st.write(my_text)
                download = FileDownloader(my_text).download()

        elif choice == "CSV":
            download = FileDownloader(result.to_csv(), file_ext='csv').download()






