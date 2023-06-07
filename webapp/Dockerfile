FROM python:3.8
LABEL MAINTAINER="Roger Schaer"

# Define app folder
WORKDIR /usr/src/app

# Install shared module
COPY ./shared /shared
RUN pip install -e /shared

# Copy dependency files
COPY ./webapp/requirements*.txt ./

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Bust cache if version of melampus has changed
ADD https://api.github.com/repos/medgift/quantimage-melampus/git/refs/heads/dev-da version-melampus.json

# Install melampus manually
RUN pip install git+https://github.com/medgift/quantimage-melampus.git@dev-da

# Copy source code
COPY ./webapp .

# Need a directory to store the migrations
RUN mkdir -p /usr/src/app/alembic/versions

# Start app
CMD ["./entrypoint.sh"]