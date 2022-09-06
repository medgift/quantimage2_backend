FROM python:3.8
MAINTAINER Roger Schaer

# Define app folder
WORKDIR /usr/src/app

# Copy dependency files
COPY requirements*.txt ./

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Bust cache if version of melampus has changed
ADD https://api.github.com/repos/medgift/quantimage-melampus/git/refs/heads/dev-da version-melampus.json

# Install melampus manually
RUN pip install git+https://github.com/medgift/quantimage-melampus.git@dev-da

# Copy source code
COPY . .

# Start app
CMD ["./entrypoint.sh"]