# Кто это? Классификатор персонажей «Симпсонов»

Загрузи скриншот из «Симпсонов», и нейросеть скажет, кто на нём. Модель знает 42 персонажа
и набирает **F1 0.9986** на закрытом лидерборде Kaggle: это около двух ошибок на тысячу картинок.

![Интерфейс](docs/screenshot.png)

## Запустить за минуту

Нужен только [Docker](https://www.docker.com/products/docker-desktop/):

```bash
docker run -p 7860:7860 <DOCKERHUB_USER>/simpsons-who
```

Открой http://localhost:7860. Работает на обычном процессоре, без видеокарты; после скачивания
образа (~0,5 ГБ) интернет не нужен. Картинки обрабатываются локально и никуда не отправляются.

<details>
<summary>Без Docker, на чистом Python 3.10+</summary>

```bash
git clone https://github.com/<GITHUB_USER>/simpsons-who
cd simpsons-who/app
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python download_model.py           # модель из GitHub Releases, ~200 МБ
uvicorn server:app --port 8000     # → http://localhost:8000
```
</details>

## Как пользоваться

- Перетащи скриншот в телевизор, вставь его через Ctrl+V или выбери файл.
- Широкий кадр модель просматривает по частям (слева, по центру, справа), показывает рамкой,
  где нашла персонажа, и перечисляет других, кого уверенно увидела.
- Если персонажей несколько, обведи нужного мышкой или пальцем: модель посмотрит только в рамку.
- При уверенности ниже 50% сервис честно пишет, что не уверен.

## Результаты

| Где | F1 (micro) |
|---|---|
| Kaggle «Journey to Springfield», private | **0.99855** |
| Kaggle «Journey to Springfield», public | 0.99663 |

Метрики на отложенной валидации (accuracy, F1 macro, калибровка, скорость) лежат в
[`docs/metrics.json`](docs/metrics.json), качество по каждому персонажу в [`docs/per_class.csv`](docs/per_class.csv).

| Обучение | Качество по персонажам | Где модель ошибается |
|---|---|---|
| ![](docs/plots/curves.png) | ![](docs/plots/per_class_f1.png) | ![](docs/plots/confusion_errors.png) |

## Как это устроено

```
Kaggle: 21 тыс. кадров, 42 персонажа
  → дообучение ConvNeXt-Small (ImageNet-22k) на V100, ~40 минут
  → экспорт в ONNX с проверкой, что ответы совпадают с PyTorch
  → FastAPI + ONNX Runtime на CPU (PyTorch не нужен)
  → Docker-образ
```

Рецепт обучения: картинки 320px, AdamW с косинусным расписанием и прогревом,
Mixup/CutMix и label smoothing, более частый показ редких персонажей (у некоторых всего 3–30 кадров),
EMA весов, fp16, на предсказании усреднение с зеркальным отражением.

## Обучить заново

1. Скачай данные соревнования [Journey to Springfield](https://www.kaggle.com/competitions/journey-springfield/data)
   и распакуй в `data/journey/` (внутри должны быть `train/`, `testset/`, `sample_submission.csv`).
2. Окружение (для V100 нужна сборка PyTorch под CUDA 12.6; на новых картах подойдёт обычная):
   ```bash
   conda create -n simpsons python=3.11 -y && conda activate simpsons
   pip install "torch==2.14.*" "torchvision==0.29.*" --index-url https://download.pytorch.org/whl/cu126
   pip install -r requirements.txt
   python classification/prefetch_cls.py      # предобученные веса ConvNeXt
   ```
3. Обучение. На любой машине с GPU:
   ```bash
   python classification/train_cls.py --data data/journey/train --name convnext_s            # с 10% валидации
   python classification/train_cls.py --data data/journey/train --name convnext_s_full --full
   ```
   Или на Slurm-кластере: `sbatch classification/run_cls.sh` (строки `#SBATCH` поправь под свой кластер).
4. Графики и метрики:
   `python classification/plots_cls.py --run runs_cls/convnext_s --data data/journey/train`
5. Сабмит на Kaggle:
   `python classification/predict_cls.py --ckpt runs_cls/convnext_s_full/best.pt --test-dir data/journey/testset --sample-submission data/journey/sample_submission.csv`
6. Модель для сервиса:
   `python classification/export_onnx.py --ckpt runs_cls/convnext_s_full/best.pt --out app/model`

## Структура

```
classification/
  train_cls.py      обучение (timm, fp16, Mixup/CutMix, EMA)
  predict_cls.py    предсказание и сабмит для Kaggle, ансамбли
  plots_cls.py      графики и метрики
  export_onnx.py    экспорт в ONNX с проверкой
  prefetch_cls.py   скачать предобученные веса заранее (для GPU-нод без интернета)
  run_cls.sh        задача для Slurm
app/
  server.py         веб-сервис: FastAPI + ONNX Runtime
  static/           страница
  download_model.py скачать модель из GitHub Releases
  Dockerfile
docs/               графики, метрики, скриншот
```

## API

```bash
curl -F image=@frame.png http://localhost:7860/api/predict
curl -F image=@frame.png -F box=100,50,400,500 http://localhost:7860/api/predict   # только внутри рамки
```

Ответ: топ-5 персонажей с вероятностями, рамка, по которой принято решение, и другие найденные персонажи.
Ещё есть `GET /api/info` (модель и список персонажей) и `GET /health`.

## Ограничения

- Модель знает только 42 персонажа. Незнакомого она назовёт похожим, но обычно с низкой уверенностью.
- Обучающие кадры в основном крупные планы одного персонажа. На общих планах с толпой
  надёжнее обвести нужного героя рамкой.

## Данные и права

Обучающие данные взяты из соревнования Kaggle [Journey to Springfield](https://www.kaggle.com/competitions/journey-springfield),
собранного на основе [The Simpsons Characters Data](https://www.kaggle.com/datasets/alexattia/the-simpsons-characters-dataset).
«Симпсоны» и все изображения принадлежат их правообладателям; кадры в репозиторий не входят.
Проект учебный и некоммерческий. Идея: [girafe-ai/mlops-simpsons](https://github.com/girafe-ai/mlops-simpsons).
