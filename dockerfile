FROM continuumio/miniconda3
ADD environment.yaml
RUN conda create -f environment.yaml
RUN echo "source activate env" > ~/.bashrc
ENV PATH /opt/conda/envs/env/bin:$PATH
