FROM public.ecr.aws/lambda/python:3.11

COPY requirements.txt ${LAMBDA_TASK_ROOT}

RUN pip install --upgrade pip
RUN apt update && apt upgrade
RUN pip install -r requirements.txt
RUN apt install poppler-utils

COPY src/* ${LAMBDA_TASK_ROOT}

CMD ...
