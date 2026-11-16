FROM python:3.12-slim-bookworm
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --uid 50000 --gid 0 --create-home airflow \
    && mkdir -p /opt/airflow /opt/project \
    && chown -R airflow:root /opt/airflow /opt/project
ENV AIRFLOW_HOME=/opt/airflow \
    PYTHONPATH=/opt/project \
    PATH=/home/airflow/.local/bin:$PATH \
    AIRFLOW__CORE__LOAD_EXAMPLES=false
WORKDIR /opt/project
USER airflow
COPY --chown=airflow:root requirements.txt requirements-airflow.txt ./
RUN pip install --user --no-cache-dir -r requirements-airflow.txt
COPY --chown=airflow:root . .
ENTRYPOINT ["/opt/project/docker-entrypoint.sh"]
