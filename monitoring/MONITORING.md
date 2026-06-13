# Мониторинг сервиса рекомендаций банковских продуктов

## Метрики инфраструктуры

| Метрика | Тип | Описание | Действие |
|---------|-----|----------|----------|
| `recommendation_requests_total` | Counter | Число запросов к `/recommend` | Рост 5xx / падение RPS |
| `recommendation_latency_seconds` | Histogram | Latency API | p95 > 500 ms — алерт |
| `http_requests_total` | Counter | HTTP-коды FastAPI | Ошибки 5xx > 1% |

## Метрики входящего трафика (data drift proxy)

| Метрика | Тип | Описание | Действие |
|---------|-----|----------|----------|
| `client_renta_avg` | Gauge | Средний доход в запросах | Сдвиг > 20% от baseline |
| `client_products_avg` | Gauge | Среднее число продуктов у клиента | Резкое изменение профиля |

## Offline-метрики модели

Файл `artifacts/metrics.json` (validation, дек 2015 → янв 2016):

| Метрика | Значение |
|---------|----------|
| MAP@7 | 0.745 |
| Precision@7 | 0.0046 |
| Recall@7 | 0.934 |
| Coverage@7 | 1.0 |

Baseline MAP@7: 0.543.

## Как смотреть метрики

```bash
# Prometheus endpoint сервиса
curl http://127.0.0.1:8000/metrics

# Prometheus UI (если поднят docker-compose)
http://127.0.0.1:9090
```

## Где реализовано в коде

- `services/ml_service/app.py` — Counter, Histogram, Gauge
- `prometheus_fastapi_instrumentator` — стандартные HTTP-метрики
- `src/train.py` — offline-метрики в MLflow и `artifacts/metrics.json`
