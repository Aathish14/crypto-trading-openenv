# Use Python 3.12 slim image
FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Copy the entire application
COPY . .

# Install dependencies and the package itself
RUN pip install --no-cache-dir uv && uv pip install --system .

# Expose port
EXPOSE 8000

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Run the server
CMD ["uv", "run", "server"]