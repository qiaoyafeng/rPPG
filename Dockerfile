# 1. Base Image
# Use a slim Python image as a base
FROM python:3.10-slim

# 2. Set Environment Variables
# Prevents Python from writing .pyc files to disc
ENV PYTHONDONTWRITEBYTECODE 1
# Ensures Python output is sent straight to the terminal without buffering
ENV PYTHONUNBUFFERED 1

# 3. Install System Dependencies
# Install libraries required for OpenCV and other packages
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# 4. Set Working Directory
WORKDIR /app

# 5. Install Python Dependencies
# Copy only the requirements file first to leverage Docker cache
COPY requirements.txt .

# First, install uv itself
RUN pip install --no-cache-dir uv

# Now, use uv to install dependencies from requirements.txt
RUN uv pip install --no-cache-dir -r requirements.txt

# 6. Copy Application Code
# Copy the rest of the application files into the working directory
COPY . .

# 7. Expose Port
# Expose the port the app runs on
EXPOSE 8000

# 8. Set Start Command
# Run uvicorn server. Note: --host 0.0.0.0 is crucial to allow external connections to the container.
# Do not use --reload in production.
CMD ["uvicorn", "main_api:app", "--host", "0.0.0.0", "--port", "8000"]
