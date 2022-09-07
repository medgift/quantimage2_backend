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

# Install shared module
COPY ./shared /shared
RUN pip install /shared

# Copy dependency files
COPY ./workers/requirements*.txt ./

# Install numpy first (for PyRadiomics)
RUN pip install --no-cache-dir numpy

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Bust cache if version of okapy has changed
ADD https://api.github.com/repos/voreille/okapy/git/refs/heads/quantimage2 version-okapy.json

# Install okapy manually
RUN pip install git+https://github.com/voreille/okapy.git@quantimage2

# Bust cache to ensure installation of latest versions of ZRad
ADD "https://www.random.org/cgi-bin/randbyte?nbytes=10&format=h" skipcache

# Add GitHub public key as a known host
RUN mkdir -p -m 0600 ~/.ssh && ssh-keyscan github.com >> ~/.ssh/known_hosts

########################################################################################################################
# OPTIONAL - Requires "docker-compose.zrad.yml" to be included in the "docker compose build" command to execute        #
########################################################################################################################
# Install Privatee ZRad repository manually (temporary)
# Requires BuildKit to be enabled with the environment variable DOCKER_BUILDKIT=1
# Requires ~/.ssh (on the host) to contain an id_rsa_zrad deploy key that can read the repo
# Requires docker-compose.yml to contain an SSH mapping zrad=$HOME/.ssh/id_rsa_zrad
# Fails silently in order not to break the build process when the installation is not successful
RUN --mount=type=ssh,id=zrad pip install git+ssh://git@github.com/danielabler/zrad_QUANTIMAGE.git; exit 0

ENV MCR_HOME /mcr/v97
ENV LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6:/usr/lib/x86_64-linux-gnu/libexpat.so
ENV LD_LIBRARY_PATH $MCR_HOME/runtime/glnxa64:$MCR_HOME/bin/glnxa64:$MCR_HOME/sys/os/glnxa64:$MCR_HOME/extern/bin/glnxa64:$LD_LIBRARY_PATH

# Copy source code
COPY ./workers .

# Start app
CMD ["celery", "--app=tasks", "worker", "--pool=prefork", "--loglevel=INFO"]