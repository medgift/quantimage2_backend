FROM python:3.6
MAINTAINER Roger Schaer

# Define app folder
WORKDIR /usr/src/app

# Copy dependency files
COPY requirements*.txt ./

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Install okapy dependencies (temporary)
RUN pip install --no-cache-dir -r requirements-okapy.txt

# Install okapy manually (temporary)
RUN pip install --index-url https://test.pypi.org/simple/ --no-deps okapy==0.1.2

# Copy source code
COPY . .

# Start app
CMD ["celery", "worker", "--app=tasks", "--loglevel=DEBUG"]