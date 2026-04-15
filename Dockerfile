FROM python:3.9-slim

# Create a non-root user for Hugging Face
RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH"

WORKDIR /app

# Copy requirements and install
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --upgrade -r requirements.txt

# Copy the rest of the application
COPY --chown=user . .

# Hugging Face Spaces expects port 7860
EXPOSE 7860

# Run the app. Note: api.app because it's in the api folder
CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "7860"]
