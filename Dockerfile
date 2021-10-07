FROM tensorflow/tensorflow:latest-gpu

RUN apt update 
RUN apt install ffmpeg libsm6 -y
RUN apt install vim -y

RUN pip install --upgrade pip
RUN pip install opencv-python
RUN pip install sklearn
RUN pip install tqdm