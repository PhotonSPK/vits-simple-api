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
    apt-get install -yq \
    build-essential \
    cmake \
    wget \
    ca-certificates \
    tzdata \
    autoconf \
    automake \
    libtool \
    pkg-config \
    gettext && \
    update-ca-certificates && \
    apt-get clean && \
    apt-get purge -y --auto-remove -o APT::AutoRemove::RecommendsImportant=false && \
    rm -rf /var/lib/apt/lists/* 

ENV ESPEAK_NG_VERSION=1.52.0

# Build and install espeak-ng from source.
RUN wget -O /tmp/espeak-ng.tar.gz https://github.com/espeak-ng/espeak-ng/archive/refs/tags/${ESPEAK_NG_VERSION}.tar.gz && \
    tar -xzf /tmp/espeak-ng.tar.gz -C /tmp && \
    cd /tmp/espeak-ng-${ESPEAK_NG_VERSION} && \
    ./autogen.sh && \
    ./configure --prefix=/usr/local && \
    make -j$(nproc) && \
    make install && \
    ldconfig && \
    rm -rf /tmp/espeak-ng.tar.gz /tmp/espeak-ng-${ESPEAK_NG_VERSION}


# Link espeak-ng to a fixed path for runtime library detection.
RUN ESPEAK_PATH=$(ldconfig -p | awk '/libespeak-ng.so.1/{print $NF; exit}') && \
    test -n "$ESPEAK_PATH" && \
    ln -sf "$ESPEAK_PATH" /usr/local/lib/libespeak-ng.so.1 && \
    ln -sf "$ESPEAK_PATH" /usr/local/lib/libespeak-ng.so


ENV MIMALLOC_VERSION=2.1.7

# Build and install mimalloc from source.
RUN wget -O /tmp/mimalloc.tar.gz https://github.com/microsoft/mimalloc/archive/refs/tags/v${MIMALLOC_VERSION}.tar.gz && \
    tar -xzf /tmp/mimalloc.tar.gz -C /tmp && \
    cd /tmp/mimalloc-${MIMALLOC_VERSION} && \
    cmake -S . -B build -DMI_BUILD_SHARED=ON -DMI_BUILD_TESTS=OFF -DCMAKE_BUILD_TYPE=Release && \
    cmake --build build -j$(nproc) && \
    cmake --install build && \
    ldconfig && \
    rm -rf /tmp/mimalloc.tar.gz /tmp/mimalloc-${MIMALLOC_VERSION}


# Link mimalloc to a fixed path for LD_PRELOAD across architectures.
RUN MIMALLOC_PATH=$(ldconfig -p | awk '/libmimalloc.so.2/{print $NF; exit}') && \
    test -n "$MIMALLOC_PATH" && \
    ln -sf "$MIMALLOC_PATH" /usr/local/lib/libmimalloc.so

ENV LD_PRELOAD=/usr/local/lib/libmimalloc.so

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
