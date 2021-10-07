FROM tensorflow/tensorflow:2.4.3-gpu

RUN apt-get update 
RUN apt-get install ffmpeg libsm6 libxext6  -y

RUN pip install --upgrade pip
RUN pip install opencv-python
RUN pip install sklearn
RUN pip install tqdm