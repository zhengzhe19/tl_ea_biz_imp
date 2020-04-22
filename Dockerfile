ARG ARTIFACTS_DIR=dist
ARG CONDA_ENV=root
FROM wce-sse-docker-local.artifactory.swg-devops.com/sse/sse-pyspark-onbuild-environment:0.0.0-experimental as conda_env_file
FROM wce-sse-docker-local.artifactory.swg-devops.com/sse/sse-pyspark-onbuild:2.2.1_0.7.0-experimental

CMD ["/opt/scripts/run-spark-submit.sh", "test.py"]

ENV WRITE_MODE=overwrite