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
RUN pip install --no-cache-dir -c requirements-constraints.txt -r requirements.txt

# Install test dependencies
COPY ./tests/requirements-test.txt ./tests/requirements-test.txt
RUN pip install --no-cache-dir -c requirements-constraints.txt -r tests/requirements-test.txt

# Install pyradiomics from the v3.1.0 release TAG (last stable release; no cp312
# wheels on PyPI so it is built from source). Pinned to a tag, not master: master
# requires numpy 2.x and returns some features as complex numbers. v3.1.0 uses the
# numpy 1.x C API, so numpy stays at 1.26.4 (held by requirements-constraints.txt).
RUN pip install --no-cache-dir --no-deps git+https://github.com/AIM-Harvard/pyradiomics.git@v3.1.0
# Install pyradiomics' other runtime deps explicitly (numpy already installed)
RUN pip install --no-cache-dir -c requirements-constraints.txt PyWavelets pykwalify

# Bust cache if version of okapy has changed
ADD https://api.github.com/repos/voreille/okapy/git/refs/heads/refactor/major-restructure version-okapy.json

# Install okapy manually
# Pre-install okapy's deps (pydicom-seg, scikit-image need numpy at build time).
# scikit-image is pinned to 0.24.0, the last release that supports numpy 1.x;
# 0.25+ requires numpy>=2 and would drag numpy off 1.26.4. The constraint file is
# a second guard so nothing here upgrades numpy.
RUN pip install --no-cache-dir -c requirements-constraints.txt pydicom-seg scikit-image==0.24.0 termcolor tqdm
RUN pip install --no-cache-dir -c requirements-constraints.txt git+https://github.com/voreille/okapy.git@refactor/major-restructure

# Copy source code
COPY ./workers .

# Start app
CMD ["celery", "--app=tasks", "worker", "--pool=prefork", "--loglevel=INFO"]