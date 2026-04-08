# SQL Debug Environment — Docker image
# HF Spaces runs this from the repo root.
# All files (app.py, requirements.txt) are in server/ subdirectory.

FROM python:3.11-slim

# Create a non-root user (HF Spaces requirement)
RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH"

WORKDIR /app

# Copy and install dependencies
COPY --chown=user requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir --upgrade -r requirements.txt

# Copy server application
COPY --chown=user server/app.py /app/app.py

# HF Spaces uses port 7860
EXPOSE 7860

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
