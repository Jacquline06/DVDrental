import streamlit as st
import numpy as np
import base64
from tensorflow.keras.models import load_model

# Function to convert image to base64
def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

# Path to your background image
background_image_path = 'img/img1.png'  # Fixed path separator for compatibility

# Convert the image to base64
base64_image = get_base64_of_bin_file(background_image_path)

# CSS to inject
page_bg_img = f"""
<style>
[data-testid="stAppViewContainer"] {{
    background-image: url("data:image/jpg;base64,{base64_image}");
    background-size: cover;
    background-repeat: no-repeat;
    background-attachment: fixed;
}}
</style>
"""
st.set_page_config(
    page_title="DVD Rental Deep Learning",
    page_icon='img/dvd.png'  # Fixed path separator for compatibility
)

# Inject CSS with markdown
st.markdown(page_bg_img, unsafe_allow_html=True)

# Load the trained deep learning model
model = load_model('DL_Algo.h5')

# Apply CSS styling
st.markdown("""
<style>
h1 {
    color: black; /* Change to your desired color */
    font-size: 34px;
}
h2 {
    color: white; /* Change header color */
}
</style>
""", unsafe_allow_html=True)

# Application Title
st.title("DVD Rental Deep Learning Project")
st.write("""
    The DVD Rental sample dataset is a PostgreSQL database designed for learning and practicing SQL queries and database management concepts.
    This project aims to predict the rental amount using Deep Learning.
""")

# Displaying the image
st.image("img/dvd.png", caption="DVD Rental Dataset", use_column_width=True)

# Mega Menu
#st.subheader("Mega Menu")

# Expanders for Mega Menu
with st.expander("Rental Information", expanded=True):
    rental_rate = st.slider("Rental Rate", min_value=0.0, max_value=10.0, value=0.0)
    length = st.slider("Length of the Movie", min_value=45, max_value=185, value=90)
    replacement_cost = st.selectbox("Replacement Cost", ["Select Replacement Cost"] + [9.99, 10.99, 11.99, 12.99, 13.99,
                                                                                      14.99, 15.99, 16.99, 17.99, 18.99,
                                                                                      19.99, 20.99, 21.99, 22.99, 23.99,
                                                                                      24.99, 25.99, 26.99, 27.99, 28.99,
                                                                                      29.99])
    rental_actual_duration = st.slider("Rental Actual Duration", min_value=1, max_value=100, value=1)

with st.expander("Movie Details"):
    rating = st.selectbox("Rating", ["Select Rating"] + ['PG-13', 'NC-17', 'PG', 'R', 'G'])
    category = st.selectbox("Category", ['Select Category'] + ['Horror', 'Documentary', 'New', 'Classics', 'Games',
                                                             'Sci-Fi', 'Foreign', 'Family', 'Travel', 'Music',
                                                             'Sports', 'Comedy', 'Drama', 'Action', 'Children',
                                                             'Animation'])
    active = st.selectbox("Status", ['Select Status'] + ['Active', 'Not Active'])

# Preprocess input function
def preprocess_input(rental_rate, length, replacement_cost, rental_actual_duration, rating, category, active):
    # Convert categorical values into numeric
    rating_dict = {'PG-13': 1, 'NC-17': 2, 'PG': 3, 'R': 4, 'G': 5}
    category_dict = {'Horror': 1, 'Documentary': 2, 'New': 3, 'Classics': 4, 'Games': 5, 'Sci-Fi': 6, 'Foreign': 7,
                     'Family': 8, 'Travel': 9, 'Music': 10, 'Sports': 11, 'Comedy': 12, 'Drama': 13, 'Action': 14,
                     'Children': 15, 'Animation': 16}
    active_dict = {'Active': 1, 'Not Active': 0}

    # Convert inputs to numerical values
    rating_value = rating_dict.get(rating, 0)
    category_value = category_dict.get(category, 0)
    active_value = active_dict.get(active, 0)

    # Format the inputs for the model
    return np.array([[rental_rate, length, replacement_cost, rental_actual_duration, rating_value, category_value, active_value]])

# Submit Button for Prediction
if st.button("Predict Amount"):
    if rental_rate and length and replacement_cost and rental_actual_duration and rating != "Select Rating" and category != "Select Category" and active != "Select Status":
        # Preprocess the input
        input_data = preprocess_input(rental_rate, length, replacement_cost, rental_actual_duration, rating, category, active)
        
        # Make prediction
        prediction = model.predict(input_data)
        predicted_amount = prediction[0][0]

        # Display the predicted amount
        st.write(f"### Predicted Rental Amount: ${predicted_amount:.2f}")

    else:
        st.warning("Please select all the inputs correctly.")

