FROM nvidia/cuda:8.0-cudnn6-devel

# Install curl and sudo
RUN apt-get update && apt-get install -y \
    curl \
    ca-certificates \
    sudo \
 && rm -rf /var/lib/apt/lists/*

# Use Tini as the init process with PID 1
RUN curl -Lso /tini https://github.com/krallin/tini/releases/download/v0.14.0/tini \
 && chmod +x /tini
ENTRYPOINT ["/tini", "--"]

# Create a working directory
RUN mkdir /app
RUN mkdir /clevr
WORKDIR /app

# Create a non-root user and switch to it
RUN adduser --disabled-password --gecos '' --shell /bin/bash user \
 && chown -R user:user /app
RUN echo "user ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/90-user
USER user

# Install Git, bzip2, and X11 client
RUN sudo apt-get update && sudo apt-get install -y --no-install-recommends \
    git \
    bzip2 \
 && sudo rm -rf /var/lib/apt/lists/*

# Install Miniconda
RUN curl -so ~/miniconda.sh https://repo.continuum.io/miniconda/Miniconda3-4.3.14-Linux-x86_64.sh \
 && chmod +x ~/miniconda.sh \
 && ~/miniconda.sh -b -p ~/miniconda \
 && rm ~/miniconda.sh

# Create a Python 3.6 environment
RUN /home/user/miniconda/bin/conda install conda-build \
 && /home/user/miniconda/bin/conda create -y --name pytorch-py36 \
    python=3.6.0 numpy pyyaml scipy ipython mkl \
 && /home/user/miniconda/bin/conda clean -ya
ENV PATH=/home/user/miniconda/envs/pytorch-py36/bin:$PATH \
    CONDA_DEFAULT_ENV=pytorch-py36 \
    CONDA_PREFIX=/home/user/miniconda/envs/pytorch-py36

# CUDA 8.0-specific steps
RUN conda install -y --name pytorch-py36 -c soumith \
    magma-cuda80 \
 && conda clean -ya

COPY requirements.txt /tmp/
RUN pip install -r /tmp/requirements.txt


# Set the default command to python3
CMD ["python3"]
