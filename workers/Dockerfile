# Install MCR
FROM python:3.6 AS build-env

# Add installer options
ADD ./external_deps/installer_input.txt /installer_input.txt

# Install external dependencies (MCR) - Should be mounted in docker-compose.yml
ADD ./external_deps/MATLAB_Runtime_R2019b_Update_1_glnxa64.tar.gz /mcr_setup
ADD ./external_deps/installer_input.txt /installer_input.txt
RUN /mcr_setup/MATLAB_Runtime_R2019b_Update_1_glnxa64/install -inputFile /installer_input.txt

# Build the Python app
FROM python:3.6
MAINTAINER Roger Schaer

ARG GIT_ACCESS_TOKEN

# Copy the MCR
COPY --from=build-env /mcr /mcr

# Define app folder
WORKDIR /usr/src/app

# Copy dependency files
COPY ./workers/requirements*.txt ./

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Install melampus dependencies (temporary)
RUN pip install --no-cache-dir -r requirements-melampus.txt

# Install melampus manually (temporary)
#RUN pip install --index-url https://test.pypi.org/simple/ --no-deps quantimage-melampus==0.0.7
RUN pip install git+https://github.com/medgift/quantimage-melampus.git@dev-da

# Install okapy dependencies (temporary)
RUN pip install --no-cache-dir -r requirements-okapy.txt

# Install okapy manually (temporary)
#RUN pip install --index-url https://test.pypi.org/simple/ --no-deps okapy==0.2.0
RUN pip install git+https://${GIT_ACCESS_TOKEN}@github.com/voreille/okapy.git@quantimage2


ENV MCR_HOME /mcr/v97
ENV LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libexpat.so
ENV LD_LIBRARY_PATH $MCR_HOME/runtime/glnxa64:$MCR_HOME/bin/glnxa64:$MCR_HOME/sys/os/glnxa64:$MCR_HOME/extern/bin/glnxa64:$LD_LIBRARY_PATH

# Install ncdu to investigate size
RUN apt-get update && apt-get install ncdu && rm -rf /var/lib/apt/lists/*

# Copy source code
COPY ./workers .

# Start app
CMD ["celery", "--app=tasks", "worker", "--loglevel=INFO"]