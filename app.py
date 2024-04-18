import streamlit as st
import joblib
import numpy as np

# Load pre-trained model and scaler
model = joblib.load('best_model.pkl')
scaler = joblib.load('scaler.pkl')
label_encoder_location = joblib.load('encoder_location.pkl')  # Load location label encoder

# Function to scale input data
def scale_input(input_data):
    scaled_data = scaler.transform(input_data)
    return scaled_data

# Function to encode location options
def encode_location(location):
    encoded_location = label_encoder_location.transform([location])[0]
    return encoded_location


# Function to decode location labels
def decode_location(encoded_location):
    decoded_location = label_encoder_location.inverse_transform([encoded_location])[0]
    return decoded_location



# Function to make predictions
def predict(input_data):
    scaled_input = scale_input(input_data)
    print(scaled_input)
    prediction = model.predict(scaled_input)
    return prediction

# Streamlit UI
st.title('House Price Prediction')

# Input fields
# Decode location labels for display to the user
locations_encoded = np.arange(len(label_encoder_location.classes_))
locations_decoded = [decode_location(encoded) for encoded in locations_encoded]
location = st.selectbox('Location', locations_decoded)

# Decode size labels for display to the user
total_sqft = st.number_input('Total Sqft', value=1000)
bath = st.number_input('Bathrooms', value=2)
balcony = st.number_input('Balcony', value=1)
bhk = st.number_input('BHK #bedroom', value=2)

def main():
    # styles 
    st.markdown("""
    <style>
    div.stButton > button:first-child {
    margin-left: auto;
    margin-right: auto;
    display: block;
    }
    </style>
    """, unsafe_allow_html=True)

# Check if any of the values are empty
    if st.button('Estimate the Price'):
        # Encode location selected by the user
        encoded_location = encode_location(location)
        user_input = [[encoded_location, total_sqft, bath, balcony, bhk]]

        if not all(user_input):
            st.warning("Please enter a value for all fields")
        else:
            prediction = predict(user_input)
            st.write(f'<div style="text-align:center; font-weight:bold; background-color:green; padding:10px;">Estimated Price: {prediction}</div>',
                     unsafe_allow_html=True)



if __name__ =='__main__':
    main()