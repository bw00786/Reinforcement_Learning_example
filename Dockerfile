FROM python:3.13-slim

WORKDIR /app
COPY requirements.txt .
RUN python -m pip install --no-cache-dir -r requirements.txt
COPY pyproject.toml .
COPY src ./src

ENV PYTHONPATH=/app/src
CMD ["python", "-m", "alien_garden", "--episodes", "10", "--seed", "42"]
