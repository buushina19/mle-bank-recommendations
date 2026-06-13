# Рекомендации банковских продуктов

## Описание

Банк хочет предложить клиенту продукты (кредиты, депозиты, карты и т.д.), которые он с высокой вероятностью оформит.  
По истории поведения клиентов за 2015 год предсказываем **новые** продукты, которые клиент получит в следующем месяце.

**Технологии:** Python, pandas, LightGBM, MLflow, FastAPI, Docker, Prometheus.

## Структура репозитория

```
mle-bank-recommendations/
├── Readme.md
├── requirements.txt
├── .gitignore
├── data/                         # train_ver2.csv (локально, не в git)
├── notebooks/
│   ├── 01_eda.ipynb              # EDA
│   └── 02_modeling.ipynb         # обучение и MLflow
├── src/
│   ├── config.py                 # константы и параметры
│   ├── data.py                   # загрузка и таргет
│   ├── features.py               # feature engineering
│   ├── metrics.py                # MAP@7, precision@7 и др.
│   ├── eda.py                    # генерация EDA-артефактов
│   └── train.py                  # обучение + MLflow
├── scripts/
│   ├── start_mlflow.sh           # MLflow UI
│   ├── train.sh                  # EDA + обучение
│   └── start_service.sh          # FastAPI локально
├── models/                       # recommender.pkl
├── artifacts/                    # метрики, encoders, графики EDA
├── services/
│   ├── ml_service/               # FastAPI приложение
│   ├── Dockerfile
│   └── docker-compose.yaml       # сервис + Prometheus
├── monitoring/
│   └── MONITORING.md
└── test_service.py
```

## Установка

```bash
git clone <ссылка-на-ваш-репозиторий>
cd mle-bank-recommendations

python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Данные

Положите `train_ver2.csv` в `data/` (файл из урока Практикума).

## Запуск

### 1. EDA + обучение (одной командой)

```bash
chmod +x scripts/train.sh
./scripts/train.sh
```

Скрипт:
- строит EDA-графики в `artifacts/eda/`;
- обучает 24 LightGBM-модели;
- сохраняет `models/recommender.pkl`;
- логирует эксперимент в MLflow.

### 2. MLflow UI

```bash
chmod +x scripts/start_mlflow.sh
./scripts/start_mlflow.sh
```

UI: http://127.0.0.1:5000

### 3. REST API (локально)

```bash
chmod +x scripts/start_service.sh
./scripts/start_service.sh
```

Пример запроса:

```bash
curl -X POST "http://127.0.0.1:8000/recommend" \
  -H "Content-Type: application/json" \
  -d '{
    "ncodpers": 123456,
    "age": 45,
    "renta": 120000,
    "segmento": "02",
    "products": {"ind_cco_fin_ult1": 1, "ind_recibo_ult1": 1}
  }'
```

### 4. Docker

```bash
cd services
docker compose up --build
```

- API: http://127.0.0.1:8000
- Prometheus: http://127.0.0.1:9090

### 5. Тест сервиса

```bash
python test_service.py
```

---

## Руководство по проекту

### Бизнес-задача → ML-задача

**Бизнес:** увеличить cross-sell — предложить релевантный продукт активному клиенту.

**ML:** для каждого клиента ранжировать 24 продукта.  
**Таргет:** продукт появился в месяце `t`, но отсутствовал в `t-1`.

**Метрики:**
- **MAP@7** — основная ranking-метрика (Santander)
- **Precision@7 / Recall@7** — качество топ-7
- **Coverage@7** — разнообразие рекомендаций

### EDA — ключевые выводы

- ~950k клиентов в месяц, данные с января 2015 по июнь 2016;
- продукты несбалансированы: `ind_cco_fin_ult1`, `ind_recibo_ult1` самые частые;
- пропуски в `renta` — заполняем медианой;
- корреляции между продуктами подтверждают cross-sell-паттерны (счета, карты, депозиты).

### Моделирование

- **Baseline:** popularity по частоте новых продуктов;
- **Модель:** 24 LightGBM-классификатора, по одному на продукт;
- **Train:** пары месяцев окт–ноя и ноя–дек 2015 (сэмпл 400k строк);
- **Validation:** дек 2015 → янв 2016 (сэмпл 200k строк);
- Продукты, уже имеющиеся у клиента, исключаются из рекомендаций (score = -1).

**Результаты на validation** (`artifacts/metrics.json`):

| Метрика | Baseline | LightGBM |
|---------|----------|----------|
| MAP@7 | 0.543 | **0.745** |
| Recall@7 | 0.903 | 0.934 |
| Coverage@7 | 0.29 | 1.0 |

### MLflow

Логируются: параметры LightGBM, baseline-метрики, метрики модели, `recommender.pkl`, encoders, `metrics.json`.

### Мониторинг

См. `monitoring/MONITORING.md`. Метрики отправляются из `services/ml_service/app.py` на endpoint `/metrics`.

---

**Random seed:** `42`
