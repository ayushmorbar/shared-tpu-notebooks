FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1

COPY requirements.txt /app/requirements.txt
RUN pip install -r /app/requirements.txt

COPY tpu_submit /app/tpu_submit
WORKDIR /app

RUN useradd --uid 1000 --create-home tpusubmit
USER 1000

EXPOSE 8080
CMD ["python", "-m", "tpu_submit"]
