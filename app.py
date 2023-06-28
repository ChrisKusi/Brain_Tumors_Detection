
import torch
import streamlit as st
import av
from PIL import Image
from torchvision import transforms
from ultralytics.yolo.engine.model import YOLO #import YOLO algorithm from ultralyrics
import os
import glob
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration

st.set_page_config(page_title="Brain-Tumor", page_icon="🧠")

# Add CSS styles to the footer section
# Add content to the footer section
footer_style = """
<style>

footer:after{
    content: 'This project is developed by Christian Kusi. - Copyright @ 2023 ';
    color: tomato;
    padding: 10px;
    text-align: center;
    position: relative;
    

}
.css-cio0dv {
    font-size: 0;
}

.css-cio0dv::after {
    content: 'Developed by Christian Kusi. Copyright @ 2023.';
    color: tomato;
    padding: 10px;
    text-align: right;
    position: relative;
    font-size: 16px;
    layout: wide;
}
a.css-z3au9t.egzxvld2 {
    display: none;
}



</style>
"""

# Apply the CSS styles
st.markdown(footer_style, unsafe_allow_html=True)
model = YOLO("runs/detect/brain_tumor_custom_#42/weights/best.pt") #Load your trained model

#logo = Image.open('brain-tumor-100.png')

with st.sidebar:
    #st.image(logo)
    st.title("Brain Tumor Detection")
   
    option = st.radio("Brain tumor detection", ["Upload", "Real-time Detection"])
    
    st.info("This application allows you to detect brain tumors using Machine Learning")
    

if option == "Upload":
    

    st.title("\n\n")

    st.title("Brain Tumor detection with YOLO v8!")
    file = st.file_uploader("Upload your image to detect Brain Tumor!", type=["png","jpg", "jpeg"])
    class_btn = st.button("Predict Disease")
    if file is not None:    
        imgs = Image.open(file)
        st.image(imgs, caption='Uploaded Image', use_column_width=True)
        # im2 = cv2.imread(file)
    if class_btn:
        if file is None:
            st.write("Invalid command, please upload an image")
        else:
            with st.spinner('Model working.... please wait!'):
      
                # model = YOLO("runs/detect/brain_tumor_custom_#1/weights/best.pt")

                with torch.no_grad():
                    res = model.predict(imgs)
                    boxes = res[0].boxes
                    res_plotted = res[0].plot()[:, :, ::-1]
                    st.image(res_plotted, caption='Detected Image', width=None)
               
                st.success('Success!, brain tumor detected',  icon="✅")
                #st.snow()
                #st.balloons()

                # st.write("seek medical care!: ")
                # Set the path to the directory containing the predictions
                # --predictions_dir = "runs/detect/predict*/"

                # Get a list of all the prediction files in the directory
                # --prediction_files = glob.glob(os.path.join(predictions_dir, "*.jpg"))
                # --print(prediction_files)

                # Get the most recent prediction file
                # --most_recent_prediction = max(prediction_files, key=os.path.getctime)

                

                # Load the most recent prediction image using PIL
                # --prediction_image = Image.open(most_recent_prediction)
                # --prediction_image = prediction_image.resize((340,340))

                # Display the most recent prediction image in Streamlit
                # --st.image(prediction_image, caption="Most recent prediction")
                # --st.error("Brain Tumor detection, seek immediate health care", icon="🚨")

                

if option == "Real-time Detection":
     st.title("Real-time Brain Tumor detection")
     st.write("Real Time Detection Coming Soon...")

#     def callback(frame):
#         img = frame.to_ndarray(format="bgr24")
#         model.predict(source=img, save=True, save_txt=True)
              
        
    
#         st.success('Success!, brain tumor detected',  icon="✅")
#                 #st.snow()
#                 #st.balloons()

#                 # st.write("seek medical care!: ")

#                 # Set the path to the directory containing the predictions
#         predictions_dirs = "runs/detect/predict*/"

#                 # Get a list of all the prediction files in the directory
#         prediction_files_1 = glob.glob(os.path.join(predictions_dirs, "*.jpg"))
#         # print(prediction_files)

#                 # Get the most recent prediction file
#         most_recent_predictions = max(prediction_files_1, key=os.path.getctime)

#                 # Load the most recent prediction image using PIL
#         prediction_images = Image.open(most_recent_predictions)

#                 # Display the most recent prediction image in Streamlit
#         st.image(prediction_images, caption="Most recent prediction")
#         st.error("Brain Tumor detection, seek immediate health care", icon="🚨")

    
#         return av.VideoFrame.from_ndarray(img, format="bgr24")



#     webrtc_streamer(key="example", video_frame_callback=callback, rtc_configuration={  # Add this line
#     "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})
        



    
