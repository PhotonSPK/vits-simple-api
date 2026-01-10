FROM ubuntu:24.04


ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHON_VERSION=3.10.11
ENV PYTHON_MAJOR=3
ENV PYTHON_MINOR=10
ENV PYTORCH_VERSION=2.9.1
ENV LANG C.UTF-8
ENV LC_ALL C.UTF-8
ENV PATH /usr/local/bin:$PATH

RUN apt-get update && apt-get upgrade -y && \
    apt-get install -y --no-install-recommends ca-certificates && \
    apt-get install --no-install-recommends -y \
    wget \
    build-essential \
    libssl-dev \
    zlib1g-dev \
    libreadline-dev \
    libsqlite3-dev \
    libexpat1-dev \
    liblzma-dev \
    libffi-dev \
    libbz2-dev && \
    apt-get autoremove -y && apt-get clean && rm -rf /var/lib/apt/lists/*

RUN wget https://www.python.org/ftp/python/${PYTHON_VERSION}/Python-${PYTHON_VERSION}.tgz && \
    tar xzf Python-${PYTHON_VERSION}.tgz && \
    cd Python-${PYTHON_VERSION} && \
    ./configure --enable-optimizations && \
    make -j$(nproc) altinstall && \
    update-alternatives --install /usr/bin/python3 python3 /usr/local/bin/python${PYTHON_MAJOR}.${PYTHON_MINOR} 1 && \
    update-alternatives --install /usr/bin/python python /usr/local/bin/python${PYTHON_MAJOR}.${PYTHON_MINOR} 1 && \
    rm -rf /Python-${PYTHON_VERSION}.tgz /Python-${PYTHON_VERSION}

RUN python -m ensurepip && \
    python -m pip install --upgrade pip && \
    ln -sf $(which pip) /usr/local/bin/pip3 && \
    pip config set global.index-url https://mirrors.cernet.edu.cn/pypi/web/simple && \
    pip install wheel

ENV CFLAGS="-I/usr/local/include/python${PYTHON_MAJOR}.${PYTHON_MINOR}/"

RUN ARCH=$(uname -m) && \
    if [ "$ARCH" = "x86_64" ]; then \
        PYTORCH_SUFFIX="+cpu"; \
    elif [ "$ARCH" = "aarch64" ]; then \
        PYTORCH_SUFFIX=""; \
    else \
        echo "Unsupported architecture"; exit 1; \
    fi && \
    pip install --no-cache-dir torch==${PYTORCH_VERSION}$PYTORCH_SUFFIX --extra-index-url https://download.pytorch.org/whl/cpu

RUN mkdir -p /app
WORKDIR /app

ENV DEBIAN_FRONTEND=noninteractive


RUN apt-get update && \
    apt-get install -yq build-essential espeak-ng cmake wget ca-certificates tzdata&& \
    update-ca-certificates && \
    apt-get clean && \
    apt-get purge -y --auto-remove -o APT::AutoRemove::RecommendsImportant=false && \
    rm -rf /var/lib/apt/lists/* 


# Install jemalloc
RUN wget https://github.com/jemalloc/jemalloc/releases/download/5.3.0/jemalloc-5.3.0.tar.bz2 && \
    tar -xvf jemalloc-5.3.0.tar.bz2 && \
    cd jemalloc-5.3.0 && \
    ./configure && \
    make -j$(nproc) && \
    make install && \
    cd .. && \
    rm -rf jemalloc-5.3.0* && \
    ldconfig

ENV LD_PRELOAD=/usr/local/lib/libjemalloc.so

COPY requirements.txt /app/
RUN pip install gunicorn --no-cache-dir && \
    pip install -r requirements.txt --no-cache-dir&& \
    rm -rf /root/.cache/pip/*

COPY . /app
COPY data /data_bak

RUN chmod +x /app/entrypoint.sh
ENTRYPOINT ["/app/entrypoint.sh"]

EXPOSE 23456

CMD ["gunicorn", "-c", "gunicorn_config.py", "app:app"]
