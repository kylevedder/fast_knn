FROM kylevedder/cudagl:11.8.0-devel-ubuntu22.04
SHELL ["/bin/bash", "-c"]
# Set the timezone info because otherwise tzinfo blocks install
# flow and ignores the non-interactive frontend command 🤬🤬🤬
RUN ln -snf /usr/share/zoneinfo/America/New_York /etc/localtime && echo "/usr/share/zoneinfo/America/New_York" > /etc/timezone

# Core system packages
RUN apt-get update --fix-missing
RUN apt install -y software-properties-common wget curl gpg gcc git make g++ unzip apt-utils

# Install miniconda to /miniconda
RUN curl -LO http://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh
RUN bash Miniconda3-latest-Linux-x86_64.sh -p /miniconda -b
RUN rm Miniconda3-latest-Linux-x86_64.sh
ENV PATH=/miniconda/bin:${PATH}
RUN conda update -y conda


ENV FORCE_CUDA="1"
# ENV TORCH_CUDA_ARCH_LIST="7.0 7.5 8.0 8.6 8.9 9.0"
ENV TORCH_CUDA_ARCH_LIST="8.6"
RUN conda install -y numpy python=3.11 pytorch==2.3 torchvision torchaudio pytorch-cuda=11.8 fvcore iopath ninja matplotlib=3.8 -c pytorch -c nvidia -c fvcore -c iopath -c conda-forge
ENV MAX_JOBS=48

# Copy bashrc to the root bashrc in the container
COPY bashrc /root/.bashrc

WORKDIR /project