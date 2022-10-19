FROM ubuntu:latest

# Fix for tzdata installation from
# https://rtfm.co.ua/en/docker-configure-tzdata-and-timezone-during-build/
ENV TZ=Europe/Oslo
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

RUN apt-get update                  \
    && apt-get install -y           \
    openbabel                  \
    pkg-config                      \
    curl                            \
    python3-pip                     \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --upgrade pip setuptools                  \
    && python3 -m pip install --no-cache-dir --upgrade numpy swig    \
    && python3 -m pip install --no-cache-dir --upgrade openbabel     \
    && python3 -m pip install --no-cache-dir 'qml @ git+https://github.com/qmlcode/qml@develop'

COPY . /app
WORKDIR /app