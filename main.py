import streamlit as st
import cv2
from PIL import Image
import os

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import tensorflow as tf
import numpy as np

model = tf.keras.models.load_model('trained_model.h5')

def is_leaf(image):
 
    image = np.array(image)
    image_hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

 
    lower_green = np.array([35, 50, 50])
    upper_green = np.array([85, 255, 255])


    mask = cv2.inRange(image_hsv, lower_green, upper_green)
    green_area = cv2.countNonZero(mask)
    
    
    if green_area > 10000: 
        return True
    return False
 

def model_prediction(test_image):
    model = tf.keras.models.load_model('trained_model.h5')
    image = tf.keras.preprocessing.image.load_img(test_image,target_size=(128,128))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr])#convert single image to a batch
    predictions = model.predict(input_arr)
    return np.argmax(predictions[0])
    

st.sidebar.title("Dashboard")
app_mode = st.sidebar.selectbox("Select Page",["Home","About","Disease Recognition"])


if(app_mode=="Home"):
    st.divider()
    st.subheader("PLANT DISEASE RECOGNITION SYSTEM🌿")
    
    st.divider()
    image_path = "images.jpg"
    st.image(image_path,width=600)
    st.markdown(""" 
    #### Welcome to the Plant Disease Recognition System! 🌿🔍
    
    Our mission is to help in identifying plant diseases efficiently. Upload an image of a plant, and our system will analyze it to detect any signs of diseases. Together, let's protect our crops and ensure a healthier harvest!

    #### How It Works
    1. **Upload Image:** Go to the **Disease Recognition** page and upload an image of a plant with suspected diseases.
    2. **Analysis:** Our system will process the image using advanced algorithms to identify potential diseases.
    3. **Results:** View the results and recommendations for further action.

    #### Why Choose Us?
    - **Accuracy:** Our system utilizes state-of-the-art machine learning techniques for accurate disease detection.
    - **User-Friendly:** Simple and intuitive interface for seamless user experience.
    - **Fast and Efficient:** Receive results in seconds, allowing for quick decision-making.

    #### Get Started
    Click on the **Disease Recognition** page in the sidebar to upload an image and experience the power of our Plant Disease Recognition System!

    #### About Us
    Learn more about the project, our team, and our goals on the **About** page.
    """)


elif(app_mode=="About"):
    st.header("About 🗂️")
    st.markdown("""
                 #### About Dataset
                This dataset is recreated using offline augmentation from the original dataset.The original dataset can be found on this github repo.
                This dataset consists of about 87K rgb images of healthy and diseased crop leaves which is categorized into 38 different classes.The total dataset is divided into 80/20 ratio of training and validation set preserving the directory structure.
                A new directory containing 33 test images is created later for prediction purpose.
                #### Content
                1. Train (70295 images)
                2. Test (33 images)
                3. validation (17569 images)
                """)


elif(app_mode=="Disease Recognition"):
    st.write("Upload an image of a leaf to check if it's healthy or diseased.")
    def main():
     st.header("Disease Recognition ☘️")
    test_image = st.file_uploader("Choose an Image:", type=["jpg", "jpeg"])
    

       
        
    if __name__ == "__main__":
      main()
  
       
    if(st.button("Show Image 🖼️")):
       
        if test_image is not None:
   
         image = Image.open(test_image)
        st.image(image, caption="Uploaded Image",width=200)
        
        if is_leaf(image):
            st.success("Valid Leaf Image")
        
        else:
         st.error("Invalid Input: This is not a leaf image.Please check it and upload again")
 
    if(st.button("Predict 💡")):
        st.snow()
        with st.spinner("please wait..."):
         st.write("Our Prediction")
        result_index = model_prediction(test_image)
        if test_image is not None:
       
         image = Image.open(test_image)
    
         
        class_name=['Apple___Apple_scab','Apple___Black_rot','Apple___Cedar_apple_rust','Apple___healthy','Blueberry___healthy','Cherry          (including_sour)___Powdery_mildew','Cherry_(including_sour)___healthy','Cherry_(including_sour)___healthy','Corn_(maize___Cercospora_leaf_spot Gray_leaf_spot','Corn_(maize)___Common_rust_','Corn_(maize)___Northern_Leaf_Blight','Corn_(maize___healthy','Grape___Black_rot','Grape___Esca_(Black_Measles)','Grape___Leaf_blight_(Isariopsis_Leaf_Spot)','Grape___healthy','Orange___Haunglongbing(Citrus_greening)','Peach___Bacterial_spot','Peach___healthy', 'Pepper,_bell___Bacterial_spot','Pepper,_bell___healthy','Potato___Early_blight','Potato___Late_blight','Potato___healthy','Raspberry___healthy','Soybean___healthy','Squash___Powdery_mildew','Strawberry___Leaf_scorch','Strawberry___healthy','Tomato___Bacterial_spot','Tomato___Early_blight','Tomato___Late_blight','Tomato___Leaf_Mold','Tomato___Septoria_leaf_spot','Tomato___Spider_mites Two-spotted_spider_mite','Tomato___Target_Spot','Tomato___Tomato_Yellow_Leaf_Curl_Virus','Tomato___Tomato_mosaic_virus','Tomato___healthy']
        if is_leaf(image):
           
         st.success("Model is Predicting it's a : {}".format(class_name[result_index]))
            # Here, you can add more logic to detect diseases if needed.
        else:
            st.error("Invalid Input: This is not a leaf image.Please check it and upload again")
        #st.success("Model is Predicting it's a {}".format(class_name[result_index]))
