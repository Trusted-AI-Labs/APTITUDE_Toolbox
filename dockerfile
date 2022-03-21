# set base image (host OS)
FROM nvidia/cuda:11.3.1-cudnn8-runtime-ubuntu20.04

RUN apt-get update \
&& apt-get install software-properties-common -y \
&& add-apt-repository ppa:deadsnakes/ppa -y \
&& apt install python3.9 -y \
&& apt-get update \
\
&& apt-get install build-essential -y \
&& apt-get install --no-install-recommends python3.9-dev python3-pip -y  \
&& update-alternatives --install /usr/bin/python python /usr/bin/python3.9 1 \
&& python -m pip install --upgrade pip \
\
&& apt-get install python3.9-tk -y \
&& apt-get install libgtk2.0-0 -y \
&& apt-get install ffmpeg libsm6 libxext6 -y \
&& apt-get clean \
&& rm -rf /var/lib/apt/lists/*

# set the working directory in the container
WORKDIR /code

# copy the dependencies file to the working directory
COPY requirements/requirements.txt .
COPY setup.py .
COPY requirements/opencv_contrib_python-4.5.5.64-cp39-cp39-linux_x86_64.whl .

# install dependencies
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir opencv_contrib_python-4.5.5.64-cp39-cp39-linux_x86_64.whl
RUN pip install --no-cache-dir torch==1.10.2+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html \
&& pip install --no-cache-dir torchvision==0.11.3+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html
RUN pip install --no-cache-dir detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu113/torch1.10/index.html
RUN pip install -e .

# set the working directory in the container
WORKDIR /code


# copy the content of the local src directory to the working directory
COPY pytb/ pytb/
COPY clients/ clients/
COPY configs/ configs/
