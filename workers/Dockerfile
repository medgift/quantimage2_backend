# Install MCR
FROM python:3.8 AS build-env

# Add installer options
ADD ./external_deps/installer_input.txt /installer_input.txt

# Install external dependencies (MCR) - Should be mounted in docker-compose.yml
ADD ./external_deps/MATLAB_Runtime_R2019b_Update_1_glnxa64.tar.gz /mcr_setup
ADD ./external_deps/installer_input.txt /installer_input.txt
RUN /mcr_setup/MATLAB_Runtime_R2019b_Update_1_glnxa64/install -inputFile /installer_input.txt

# Build the Python app
FROM python:3.8
MAINTAINER Roger Schaer

# Copy the MCR
COPY --from=build-env /mcr /mcr

# Define app folder
WORKDIR /usr/src/app

# Install ncdu to investigate size
RUN apt-get update && apt-get install ncdu && rm -rf /var/lib/apt/lists/*

# Copy dependency files
COPY ./workers/requirements*.txt ./

# Install numpy first (for PyRadiomics)
RUN pip install --no-cache-dir numpy

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Install okapy dependencies (temporary)
RUN pip install --no-cache-dir -r requirements-okapy.txt

# Copy required SSH config
ADD ./workers/ssh-config /root/.ssh/config
RUN chmod 0600 /root/.ssh/config

# Bust cache to ensure installation of latest versions of Okapy & ZRad
ADD http://worldtimeapi.org/api/timezone/Europe/Zurich cache-bust.json

# Install okapy manually (temporary)
# Requires BuildKit to be enabled with the environment variable DOCKER_BUILDKIT=1
# Requires ~/.ssh to contain a id_rsa_okapy deploy key that can read the repo
RUN --mount=type=ssh pip install git+ssh://git@github.com-okapy/voreille/okapy.git@quantimage2

# Install ZRad manually (temporary)
# Requires BuildKit to be enabled with the environment variable DOCKER_BUILDKIT=1
# Requires ~/.ssh to contain a id_rsa_zrad deploy key that can read the repo
RUN --mount=type=ssh pip install git+ssh://git@github.com-zrad/danielabler/zrad_QUANTIMAGE.git

ENV MCR_HOME /mcr/v97
ENV LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6:/usr/lib/x86_64-linux-gnu/libexpat.so
ENV LD_LIBRARY_PATH $MCR_HOME/runtime/glnxa64:$MCR_HOME/bin/glnxa64:$MCR_HOME/sys/os/glnxa64:$MCR_HOME/extern/bin/glnxa64:$LD_LIBRARY_PATH

# Copy source code
COPY ./workers .

# Start app
CMD ["celery", "--app=tasks", "worker", "--pool=prefork", "--loglevel=INFO"]