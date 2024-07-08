# Docker for Python 3.10, postgres, pgvector, OLLama, Streamlit

FROM python:3.10-slim-buster

# Keeps Python from generating .pyc files in the container
ENV PYTHONDONTWRITEBYTECODE=1

# Turns off buffering for easier container logging
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install necessary system packages
RUN apt-get update && \
    apt-get install -y postgresql-client libpq-dev gcc && \
    rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies
COPY requirements.txt requirements.txt
RUN pip3 install --upgrade pip && \
    pip3 install -r requirements.txt

# Copy the application code
COPY . .

EXPOSE 5000

CMD ["python3", "app.py"]

