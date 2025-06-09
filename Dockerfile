# Use the ROCm JAX base image
FROM rocm/jax-community:latest

# Set environment variables for ROCm (optional, adjust if necessary)
ENV PATH=/opt/rocm/bin:$PATH
ENV LD_LIBRARY_PATH=/opt/rocm/lib:$LD_LIBRARY_PATH

# Set working directory
WORKDIR /app

# Copy requirements.txt into the container
COPY requirements.txt .

# Install Python dependencies from requirements.txt
RUN pip install -r requirements.txt

COPY . .
