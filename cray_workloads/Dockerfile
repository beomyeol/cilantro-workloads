# FROM rayproject/ray:latest
FROM python:3.7.11-slim-buster

# Set up the development environment
WORKDIR /cray_workloads

# Install extra packages
RUN apt update && \
    yes | apt install nano wget curl vim

# Install dependencies
COPY requirements.txt setup.py ./
RUN pip install -U pip && \
    pip install --no-cache-dir -r requirements.txt

RUN yes | pip install ray==1.6.0 protobuf==3.18.1
RUN yes | pip install torch==1.12.1+cpu torchvision==0.13.1+cpu torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cpu

# Copy the data files 
# COPY db_data ./db_data/
COPY train_data ./train_data/
COPY traces ./traces/
COPY db_bins ./db_bins/

# Copy the cray_workloads files
COPY cray_workloads ./cray_workloads/
# Install cilantro
RUN pip install -e .
# ADD dist/cray_workloads-0.1-py3-none-any.whl cray_workloads-0.1-py3-none-any.whl
# RUN pip install cray_workloads-0.1-py3-none-any.whl

ENV PYTHONUNBUFFERED 1

CMD ["python", "/cray_workloads/cray_workloads/drivers/cray_runscript.py"]
