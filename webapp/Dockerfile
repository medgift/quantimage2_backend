FROM python:3.6
MAINTAINER Roger Schaer

# Define app folder
WORKDIR /usr/src/app

# Copy dependency files
COPY requirements*.txt ./

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Install melampus manually (temporary)
#RUN pip install --index-url https://test.pypi.org/simple/ --no-deps quantimage-melampus==0.0.7
RUN pip install git+https://github.com/medgift/quantimage-melampus.git@dev-da

# Copy source code
COPY . .

# Start app
CMD ["python", "app.py"]