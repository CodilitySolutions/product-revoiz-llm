FROM python:3.12-slim

ENV PYTHONUNBUFFERED=True

ENV APP_HOME=/app
WORKDIR $APP_HOME

COPY requirements.txt requirements.txt

RUN pip install --no-cache-dir --upgrade -r requirements.txt

COPY . ./

CMD exec uvicorn server:app --host 0.0.0.0 --port ${PORT} --workers 2
# CMD [ "uvicorn", "server:app", "--host", "", "0.0.0.0", "--port", "${PORT}", "--workers", "2" ]
