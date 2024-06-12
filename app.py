import streamlit as st
import pickle
import numpy as np
import pandas

Ridge=pickle.load(open,'C:\Users\Anmol\Downloads\Ridge.pkl','wb')
Lasso=pickle.load(open,'C:\Users\Anmol\Downloads\Lasso.pkl','wb')
KNN=pickle.load(open,'C:\Users\Anmol\Downloads\KNN.pkl','wb')

st.title('car_details')
 
st.subheader('information about car')
 
options=st.sidebar.selectbox('Select ML Model',['Ridge','KNeighborsRegressor','Lasso'])
 
# name=st.slider()
year=st.slider(1992,2020) 
selling_price=st.slider(2.000000e+04,8.900000e+06)
km_driven=st.slider(1.000000,806599.000000)
fuel=st.selectbox('fule',['Petrol','Diesel','CNG'])
seller_type=st.selectbox('seller_type',['Individual','Dealer','Trustmark Dealer'])
transmission=st.selectbox('transmission',['Manual','Automatic']) 
owner=st.selectbox('owner',['First Owner','Second Owner','Third Owner','Fourth & Above Owner','Test Drive Car']) 
 
 
if st.button('predict'): 
    
    if fuel == "Petrol":
        fuel=0
    elif fuel =="Diesel":
        fuel=1
    else:
        fuel=2
    if seller_type =="Individual":
        seller_type=0
    elif seller_type=="Dealer":
        seller_type=1
    else:
        seller_type=2
    if transmission =="Manual":
      transmission=0
    else:
      transmission=1
    if owner =="First Owner":
        Owner=0
    elif owner =="Second Owner":
        Owner=1
    elif owner =="Third Owner":
        Owner=2
    elif owner=="Fourth & Above Owner":
        Owner=3
    else:
        Owner=4
        
        
    test = np.array([year,selling_price,km_driven,fuel,seller_type,transmission,owner])
    test = test.reshape(1,7)
    if options == "Ridge":
        st.success(Ridge.predict(test)[0])
    elif options =="KNeighborsRegressor":
        st.success(KNN.predict(test)[0])
    else:
        st.success(Lasso.predict(test)[0])
                
     
 
 