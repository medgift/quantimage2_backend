# Build the Python app
FROM python:3.12
LABEL MAINTAINER="Lluis Borras Ferris"

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
RUN pip install --no-cache-dir numpy==1.26.4

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Install test dependencies
COPY ./tests/requirements-test.txt ./tests/requirements-test.txt
RUN pip install --no-cache-dir -r tests/requirements-test.txt

# Install pyradiomics from GitHub source (no cp312 wheels on PyPI)
# Use --no-deps to prevent upgrading numpy to 2.x (master requires numpy>=2.0)
RUN pip install --no-cache-dir --no-deps git+https://github.com/AIM-Harvard/pyradiomics.git@master
# Install pyradiomics' other runtime deps explicitly (numpy already installed)
RUN pip install --no-cache-dir PyWavelets pykwalify

# Bust cache if version of okapy has changed
ADD https://api.github.com/repos/voreille/okapy/git/refs/heads/quantimage2 version-okapy.json

# Install okapy manually
# Pre-install okapy's deps (pydicom-seg, scikit-image need numpy at build time)
# then install okapy with --no-deps to avoid rebuilding pyradiomics
RUN pip install --no-cache-dir pydicom-seg scikit-image termcolor tqdm
RUN pip install --no-cache-dir --no-deps git+https://github.com/voreille/okapy.git@quantimage2

# Copy source code
COPY ./workers .

# Start app
CMD ["celery", "--app=tasks", "worker", "--pool=prefork", "--loglevel=INFO"]