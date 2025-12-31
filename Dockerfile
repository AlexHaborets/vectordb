FROM python:3.12-slim AS builder

WORKDIR /app

COPY requirements.txt ./

RUN pip wheel --no-cache-dir --no-deps --wheel-dir wheels -r requirements.txt

FROM python:3.12-slim AS runner

WORKDIR /app

COPY --from=builder /app/wheels /wheels
RUN pip install --no-cache /wheels/* && rm -rf /wheels

COPY src ./src
COPY alembic ./alembic
COPY alembic.ini .
COPY entrypoint.sh .

RUN mkdir -p /app/data

RUN chmod +x entrypoint.sh

ENTRYPOINT ["./entrypoint.sh"]

EXPOSE 8000

CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]

