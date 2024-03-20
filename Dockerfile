FROM python:3.8

WORKDIR /app

COPY . /app

RUN pip install --upgrade pip
RUN pip install uvicorn fastapi numpy transformers joblib pydantic tf-keras tensorflow

CMD uvicorn summarizationapi:app --reload --port=8000 --host=0.0.0.0