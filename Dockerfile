ARG PYTORCH="1.8.0"
ARG CUDA="11.1"
ARG CUDNN="8"

FROM pytorch/pytorch:${PYTORCH}-cuda${CUDA}-cudnn${CUDNN}-runtime
RUN apt-get update && \
	apt-get install -y ffmpeg libsm6 libxext6 graphviz && \
	apt-get install -y --reinstall xdg-utils gcc


RUN pip install -U pip && pip install --no-cache-dir numpy graphviz pycocotools opencv-python wandb
