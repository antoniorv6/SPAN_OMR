FROM tensorflow/tensorflow:2.4.3-gpu

RUN pip install --upgrade pip
RUN pip install opencv-python
RUN pip install sklearn
RUN pip install tqdm