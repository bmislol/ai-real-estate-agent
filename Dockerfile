# Use a lightweight version of Python 3.12
FROM python:3.12-slim

# Set the working directory inside the container
WORKDIR /app

# Install 'uv' globally inside the container
RUN pip install uv

# Copy ONLY the dependency files first (This makes future builds much faster)
COPY pyproject.toml uv.lock ./

# Install the dependencies using uv
RUN uv sync --frozen

# Copy the rest of your application code into the container
COPY api/ api/
COPY models/ models/
COPY main.py .

# Tell Docker that FastAPI uses port 8000
EXPOSE 8000

# The command that runs when the container starts
CMD ["uv", "run", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]