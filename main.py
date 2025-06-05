import streamlit as st
import numpy as np
import tensorflow as tf

# Tensorflow Model Prediction
def model_predictipn(test_img):
    model = tf.keras.models.load_model('trained_model.keras')
    image = tf.keras.preprocessing.image.load_img(test_img,target_size=(128,128))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr]) # Convert single image to a batch
    prediction = model.predict(input_arr)
    result_index = np.argmax(prediction)
    return result_index


# Sidebar
st.sidebar.title("Dashboard")
app_mode = st.sidebar.selectbox("Menu",["Home","Disease Detection","About"])


# Home Page
if(app_mode=="Home"):
    st.header("PLANT DISEASE DETECTION SYSTEM")
    image_path = ("home_page.png")
    st.image(image_path,use_column_width=True)
    st.markdown("""
    ### WASSSSUPPPP !!!!!!!

    Its your boii PDD (Plant Disease Detector),
    here to help you detect those nastyy plant diseases 
                
    
    #### WHY CHOOSE ME??
                
    I am free to use
                
    """)

    st.header("Service Available For")
    st.markdown("""
    ##### PDD is only able to recognize the images for the classes shown in the picture below.
               
    """)
    
    if (st.button("Show Image")):
        image_path = ("services_available_for.png")
        st.image(image_path,use_column_width=True)
        st.write("Stay updated for more ;)")

   
# About Page
elif (app_mode=="About"):
    st.header("HERE SOME MORE INFORMATION ABOUT THIS PROJECT")
    st.markdown("""
    #### About Dataset

    This dataset is recreated using offline augmentation from the original
    dataset. The original dataset can be found on this github repo. This
    dataset consists of about 87K rgb images of healthy and diseased crop
    leaves which is categorized into 38 different classes. The total dataset
    is divided into 80/20 ratio of training and validation set preserving the
    directory structure. A new directory containing 33 test images is created
    later for prediction purpose.


    #### Context
    1. Train (70,295 images)
    2. Valid (17,572 images)
    3. Test (33 images)       
                
    """)


# Disease Detection Page
elif (app_mode=="Disease Detection"):
    st.header("DISEASE DETECTOR")
    test_image = st.file_uploader("Lemme have a look")
    # Displays the image uploaded by the user
    if (st.button("Show Image")):
        st.image(test_image,use_column_width=True)

    # Predict Button
    if (st.button("Predict")):
        # Some effects
        with st.spinner("Please wait"):
            st.write("Prediction Result:")
            result_index = model_predictipn(test_image)

            # Define class name
            class_name = ['Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust',
                        'Apple___healthy', 'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew',
                        'Cherry_(including_sour)___healthy', 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
                        'Corn_(maize)___Common_rust_', 'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy',
                        'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
                        'Grape___healthy', 'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot',
                        'Peach___healthy', 'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy',
                        'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy',
                        'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew',
                        'Strawberry___Leaf_scorch', 'Strawberry___healthy', 'Tomato___Bacterial_spot',
                        'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold',
                        'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite',
                        'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
                        'Tomato___Tomato_mosaic_virus', 'Tomato___healthy']
        
        st.success("PDD is detecting: {}".format(class_name[result_index]))
        
