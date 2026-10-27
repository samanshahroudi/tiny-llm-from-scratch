FROM python:3.11-slim
WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 TINYLLM_THREADS=2
RUN useradd --create-home --uid 1000 learner
COPY pyproject.toml ./
COPY src ./src
RUN pip install --no-cache-dir torch==2.7.1 --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir .
COPY artifacts/demo ./artifacts/demo
USER learner
EXPOSE 7860
CMD ["uvicorn", "tinyllm.api:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "1"]
