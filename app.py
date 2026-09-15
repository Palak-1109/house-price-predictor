import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="House Price Predictor", page_icon="🏠")
model = joblib.load("model.pkl")

st.title("🏠 House Price Predictor")
st.write("Enter property details to estimate the house price.")

c1,c2=st.columns(2)
with c1:
    area=st.number_input("Area (sq. ft.)",300,10000,1200,50)
    bedrooms=st.number_input("Bedrooms",1,10,2,1)
    bathrooms=st.number_input("Bathrooms",1,10,2,1)
with c2:
    floors=st.number_input("Floors",1,10,2,1)
    parking=st.number_input("Parking Spaces",0,5,1,1)
    location=st.selectbox("Location",["Pune","Mumbai","Bengaluru","Hyderabad","Nashik"])

if st.button("Predict Price",type="primary"):
    data=pd.DataFrame([{"area_sqft":area,"bedrooms":bedrooms,"bathrooms":bathrooms,
                        "floors":floors,"parking":parking,"location":location}])
    prediction=model.predict(data)[0]
    st.success(f"Estimated House Price: ₹{prediction:,.0f}")

st.caption("Educational project — predictions are estimates.")
