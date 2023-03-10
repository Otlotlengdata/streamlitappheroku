import streamlit as st
import numpy as np
import pickle

loaded_model = pickle.load(open('trained_model.sav', 'rb'))

#Creating a function

def survival_prediction(input_data):

    input_array = np.array(input_data)
    input_reshaped = input_array.reshape(1,-1)
    prediction = loaded_model.predict(input_reshaped)
    print(prediction)
    if (prediction[0] == 0):
        return 'The person did not survive'
    else:
        return 'The person survived'
    

def main():

    #creating a title
    st.title('Survival Prediction Web App')

    #Getting the input data from the user

    PassengerId = st.number_input('Enter passenger ID')
    Age = st.number_input('Enter the age of the passenger')
    SibSp = st.number_input('How many Siblings')
    Parch = st.number_input('How many parents with children')
    male_gender = st.number_input('Are they males')
    Embarked_C = st.number_input('Embarked on C')
    Embarked_Q = st.number_input('Embarked on Q')
    Embarked_S = st.number_input('Embarked on S')

    #code for prediction
    survival = ''

    #creating a button for prediction
    if st.button('Survival test result'):
        survival = survival_prediction([PassengerId, Age, SibSp, Parch, male_gender, Embarked_C, Embarked_Q, Embarked_S])

    st.success(survival)


if __name__ == '__main__':
    main()


    
 