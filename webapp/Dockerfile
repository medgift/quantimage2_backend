FROM python:3.7
MAINTAINER Roger Schaer

# Define app folder
WORKDIR /usr/src/app

# Copy dependency files
COPY requirements*.txt ./

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY . .

# Start app
CMD ["flask", "run"]