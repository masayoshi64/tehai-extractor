FROM python:3.6

# opencv-devのインストール
RUN apt-get update -y && apt-get install -y libopencv-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# opencvのインストール
RUN pip3 install numpy opencv-python

WORKDIR /usr/src/tehai

COPY . .

ENTRYPOINT [ "python", "./image_processing.py" ]