# Start with NVIDIA CUDA 12.2 base image with Python support
FROM nvidia/cuda:12.2.2-cudnn8-runtime-ubuntu22.04

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Create symbolic links for python
RUN ln -s /usr/bin/python3 /usr/bin/python

# Create a working directory
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app/

# Install PyTorch first
RUN pip3 install --upgrade pip && \
    pip3 install --no-cache-dir \
    torch==2.2.1 \
    torchvision==0.17.1 \
    torchaudio==2.2.1

# Then install other requirements
RUN pip3 install -r requirements.txt

# Verify CUDA availability
#RUN python3 -c "import torch; print('CUDA available:', torch.cuda.is_available())"


# Set the correct working directory for nnunet
WORKDIR /app/nnunet

# Set the default command to run the inference script
CMD ["python", "../nnunet/runs.py"]