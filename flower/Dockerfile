FROM python:3.9-alpine

# Get latest root certificates
RUN apk add --no-cache ca-certificates && update-ca-certificates

# Install the required packages
RUN pip install --no-cache-dir redis==5.2.1 https://github.com/medgift/flower/zipball/master

# PYTHONUNBUFFERED: Force stdin, stdout and stderr to be totally unbuffered. (equivalent to `python -u`)
# PYTHONHASHSEED: Enable hash randomization (equivalent to `python -R`)
# PYTHONDONTWRITEBYTECODE: Do not write byte files to disk, since we maintain it as readonly. (equivalent to `python -B`)
ENV PYTHONUNBUFFERED=1 PYTHONHASHSEED=random PYTHONDONTWRITEBYTECODE=1

# Default port
EXPOSE 3333

# Run as a non-root user by default, run as user with least privileges.
USER nobody

ENTRYPOINT ["flower", "--broker=redis://redis:6379/0", "--result-backend=redis://redis:6379/0", "--port=3333"]