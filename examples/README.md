# 🚀 Примеры разработки для Swarm

Этот каталог содержит практические примеры и инструменты для разработки и обучения RL моделей в подсети Swarm.

## 📁 Структура

```
examples/
├── training/           # Скрипты обучения
│   └── train_advanced.py
├── testing/           # Инструменты тестирования
│   └── comprehensive_test.py
├── configs/           # Конфигурационные файлы
│   └── training_config.yaml
├── automation/        # Автоматизация процессов
│   └── full_pipeline.py
└── notebooks/         # Jupyter notebooks (будут добавлены)
```

## 🎯 Быстрый старт

### 1. Базовое обучение

```bash
# Активация окружения
source miner_env/bin/activate

# Простое обучение
python examples/training/train_advanced.py --experiment-name my_first_model --total-timesteps 100000

# Обучение с конфигурацией
python examples/training/train_advanced.py --config examples/configs/training_config.yaml
```

### 2. Тестирование модели

```bash
# Комплексное тестирование
python examples/testing/comprehensive_test.py --model model/ppo_policy.zip

# Тестирование с визуализацией
python examples/testing/comprehensive_test.py --model model/ppo_policy.zip --gui
```

### 3. Автоматизированный пайплайн

```bash
# Полный пайплайн разработки
python examples/automation/full_pipeline.py --experiment-name production_model

# Быстрый тест
python examples/automation/full_pipeline.py --quick --experiment-name quick_test
```

## 📋 Конфигурации

### Доступные конфигурации в `training_config.yaml`:

- **default**: Стандартные параметры для обучения
- **quick_test**: Быстрое тестирование (100K шагов)
- **advanced**: Продвинутое обучение (5M шагов)
- **curriculum**: Обучение с постепенным усложнением
- **production**: Финальное обучение для продакшена

### Использование конфигураций:

```bash
# Быстрый тест
python examples/training/train_advanced.py --config examples/configs/training_config.yaml:quick_test

# Продвинутое обучение
python examples/training/train_advanced.py --config examples/configs/training_config.yaml:advanced
```

## 🔧 Параметры командной строки

### train_advanced.py

```bash
--config              # Путь к YAML конфигурации
--experiment-name     # Имя эксперимента
--total-timesteps     # Общее количество шагов обучения
--n-envs             # Количество параллельных сред
--learning-rate      # Скорость обучения
--model-dir          # Директория для сохранения моделей
--log-dir            # Директория для логов
--resume-from        # Путь к модели для продолжения обучения
```

### comprehensive_test.py

```bash
--model              # Путь к модели для тестирования
--output-dir         # Директория для результатов
--basic-episodes     # Количество эпизодов для базового теста
--robustness-episodes # Количество эпизодов для теста робастности
--gui                # Показать GUI для первого эпизода
```

### full_pipeline.py

```bash
--config             # Путь к конфигурации пайплайна
--experiment-name    # Имя эксперимента
--quick              # Быстрый режим тестирования
```

## 📊 Результаты и отчеты

### Структура результатов:

```
pipeline_results/
└── experiment_name/
    ├── models/          # Сохраненные модели
    ├── logs/           # TensorBoard логи
    ├── results/        # Результаты тестирования
    ├── pipeline_report.json  # Полный отчет
    └── summary.txt     # Краткое резюме
```

### Анализ результатов:

```bash
# Просмотр TensorBoard логов
tensorboard --logdir pipeline_results/experiment_name/logs/

# Анализ JSON отчета
python -c "import json; print(json.load(open('pipeline_results/experiment_name/pipeline_report.json')))"
```

## 🎓 Примеры использования

### Пример 1: Быстрая разработка и тестирование

```bash
# 1. Быстрое обучение
python examples/training/train_advanced.py \
    --experiment-name quick_dev \
    --total-timesteps 50000 \
    --n-envs 2

# 2. Тестирование
python examples/testing/comprehensive_test.py \
    --model models/quick_dev_final.zip \
    --basic-episodes 20

# 3. Подготовка к развертыванию
cp models/quick_dev_final.zip model/ppo_policy.zip
```

### Пример 2: Продакшен пайплайн

```bash
# Полный автоматизированный пайплайн
python examples/automation/full_pipeline.py \
    --experiment-name production_v1 \
    --config examples/configs/training_config.yaml:production
```

### Пример 3: Curriculum Learning

```bash
# Обучение с постепенным усложнением
python examples/training/train_advanced.py \
    --config examples/configs/training_config.yaml:curriculum \
    --experiment-name curriculum_model
```

## 🔍 Мониторинг и отладка

### TensorBoard

```bash
# Запуск TensorBoard
tensorboard --logdir logs/

# Просмотр в браузере
# http://localhost:6006
```

### Логи обучения

```bash
# Просмотр логов в реальном времени
tail -f logs/training.log

# Поиск ошибок
grep -i error logs/training.log
```

## 🚨 Устранение неполадок

### Частые проблемы:

1. **Ошибка памяти**: Уменьшите `n_envs` или `batch_size`
2. **Медленное обучение**: Увеличьте `learning_rate` или уменьшите `n_steps`
3. **Нестабильное обучение**: Уменьшите `learning_rate` или увеличьте `batch_size`
4. **Низкая производительность**: Проверьте функцию награды и параметры среды

### Отладка:

```bash
# Запуск с отладочной информацией
python examples/training/train_advanced.py --experiment-name debug --total-timesteps 1000 --n-envs 1

# Тестирование среды
python -c "
from swarm.utils.env_factory import make_env
from swarm.validator.task_gen import random_task
from swarm.constants import SIM_DT, HORIZON_SEC

task = random_task(SIM_DT, HORIZON_SEC)
env = make_env(task, gui=True)
obs = env.reset()
print('Environment created successfully!')
print(f'Observation shape: {obs.shape}')
env.close()
"
```

## 📚 Дополнительные ресурсы

- [Руководство по разработке](../docs/development_guide.md)
- [Анализ архитектуры](../docs/analysis/)
- [Документация майнера](../docs/miner.md)
- [Stable Baselines 3 документация](https://stable-baselines3.readthedocs.io/)

## 🤝 Вклад в развитие

Если у вас есть идеи для улучшения примеров или вы нашли ошибки:

1. Создайте issue в репозитории
2. Предложите pull request с улучшениями
3. Поделитесь своими конфигурациями и результатами

## 📄 Лицензия

Все примеры распространяются под лицензией MIT, как и основной проект Swarm.