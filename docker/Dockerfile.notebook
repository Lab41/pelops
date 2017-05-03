FROM l41-pelops-base

MAINTAINER Lab41 <info@lab41.org>

# The startup script installs Pelops with pip from this directory
RUN mkdir /pelops
WORKDIR /pelops

# Run a notebook
EXPOSE 8888

# Install Jupyter notebook
RUN conda install --quiet --yes \
    'notebook=4.1*' \
    && conda clean -tipsy

ADD pelops_start.sh /

CMD ["/pelops_start.sh"]
