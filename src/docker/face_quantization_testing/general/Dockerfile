FROM {PRIVATE_DOCKER_IMAGE_BASED_ON_UBUNTU_20}

RUN python3.10 -m pip install --upgrade pip

ENV PWD=/home/python-dev/workdir/
ENV PATH="/home/python-dev/.local/bin:$PATH"

WORKDIR /opt
COPY face-requirements.txt ./

ARG PIP_EXTRA_INDEX_URL
RUN pip install -r face-requirements.txt --extra-index-url $PIP_EXTRA_INDEX_URL

CMD ["bash"]