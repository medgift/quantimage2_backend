# Install MCR
FROM python:3.8 AS build-env

WORKDIR /mcr_setup

# Download MCR from MathWorks server
RUN wget https://ssd.mathworks.com/supportfiles/downloads/R2019b/Release/9/deployment_files/installer/complete/glnxa64/MATLAB_Runtime_R2019b_Update_9_glnxa64.zip

# Untar & delete archive
RUN unzip MATLAB_Runtime_R2019b_Update_9_glnxa64.zip -d MATLAB_Runtime_R2019b_Update_9_glnxa64

# Copy installer input for silent install
RUN printf "destinationFolder=/mcr\nagreeToLicense=yes\nmode=silent" > /installer_input.txt

# Install MCR
RUN /mcr_setup/MATLAB_Runtime_R2019b_Update_9_glnxa64/install -inputFile /installer_input.txt

# Delete setup files
RUN rm -rf /mcr_setup

# Build the Python app
FROM python:3.8
MAINTAINER Roger Schaer

# Copy the MCR
COPY --from=build-env /mcr /mcr

# Define app folder
WORKDIR /usr/src/app

# Install ncdu to investigate directory size
RUN apt-get update && apt-get install ncdu && apt-get clean && rm -rf /var/lib/apt/lists/*

# Copy dependency files
COPY ./workers/requirements*.txt ./

# Install numpy first (for PyRadiomics)
RUN pip install --no-cache-dir numpy

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Install okapy dependencies (temporary)
RUN pip install --no-cache-dir -r requirements-okapy.txt

# Bust cache to ensure installation of latest versions of Okapy & ZRad
ADD "https://www.random.org/cgi-bin/randbyte?nbytes=10&format=h" skipcache

# Add GitHub public key as a known host
RUN mkdir -p -m 0600 ~/.ssh && ssh-keyscan github.com >> ~/.ssh/known_hosts

# Install okapy manually (temporary)
# Requires BuildKit to be enabled with the environment variable DOCKER_BUILDKIT=1
# Requires ~/.ssh (on the host) to contain a id_rsa_okapy deploy key that can read the repo
# Requires docker-compose.yml to contain an SSH mapping okapy=$HOME/.ssh/id_rsa_okapy
RUN --mount=type=ssh,id=okapy pip install git+ssh://git@github.com/voreille/okapy.git@quantimage2

# Install ZRad manually (temporary)
# Requires BuildKit to be enabled with the environment variable DOCKER_BUILDKIT=1
# Requires ~/.ssh (on the host) to contain a id_rsa_zrad deploy key that can read the repo
# Requires docker-compose.yml to contain an SSH mapping zrad=$HOME/.ssh/id_rsa_zrad
RUN --mount=type=ssh,id=zrad pip install git+ssh://git@github.com/danielabler/zrad_QUANTIMAGE.git

ENV MCR_HOME /mcr/v97
ENV LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6:/usr/lib/x86_64-linux-gnu/libexpat.so
ENV LD_LIBRARY_PATH $MCR_HOME/runtime/glnxa64:$MCR_HOME/bin/glnxa64:$MCR_HOME/sys/os/glnxa64:$MCR_HOME/extern/bin/glnxa64:$LD_LIBRARY_PATH

# Copy source code
COPY ./workers .

# Start app
CMD ["celery", "--app=tasks", "worker", "--pool=prefork", "--loglevel=INFO"]