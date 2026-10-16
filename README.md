# Contrastive RL на JAX: Craftax + MiniGrid

Откройте [`notebooks/contrastive_rl_ru.ipynb`](notebooks/contrastive_rl_ru.ipynb).
Это автономная учебная реализация основных механизмов статьи
*Scaling Contrastive Reinforcement Learning to Open-Ended Environments* с подробными
пояснениями на русском и комментариями, как соответствующие операции выглядели бы в PyTorch.

Вычисления обучения выполняются на **JAX + Flax + Optax**. PyTorch не требуется.
Среды — настоящие pip-пакеты **Craftax** и **MiniGrid**; самописных правил MDP нет.
MiniGrid использует свой Python/Gymnasium API; Craftax векторизуется нативно на JAX.

В notebook есть:

- Формулы semantic occupancy, Rao–Blackwell targets, absorbing class и termination discount.
- MiniGrid Empty, MiniGrid LavaCrossing и Craftax Classic; через настройку доступен DoorKey.
- Раздельная обработка смерти, успешного terminal, time-limit, final observation и reset.
- CRL-абляции, полноценный PPO baseline и совместное обучение Dual-CRL-PPO с нуля.
- Кривые обучения, успех по целям, посещения MiniGrid, здоровье/reward Craftax,
  инспектор logits/probabilities, target mass, PPO diagnostics и native анимации.
- Ползунки и `run_custom` для независимых экспериментов.

## Запуск

Проверенный набор версий требует **Python 3.12+**.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-notebook.txt
python -m jupyterlab
```

Выберите kernel этого окружения, откройте notebook и выполните **Restart Kernel and Run All Cells**.
Все определения находятся внутри notebook: локальные Python-модули, API-ключи и PDF при запуске не нужны.
Зависимости и версии установленных пакетов выводятся в первой кодовой ячейке.
Для GPU установите совместимый с вашей CUDA вариант JAX отдельно; CPU-профиль используется по умолчанию.

Параметры находятся в `Config`, `BASE`, `CRAFTAX_CONFIG`, `SEEDS` и `RUN_PPO`.
Начальная JIT-компиляция занимает дополнительное время. Включён один training seed;
для оценки устойчивости задайте несколько seed — графики покажут mean ± std.

```python
sampled = run_custom(rao_blackwell=False)
penalty = run_custom(gamma_term=0.5)
doorkey = run_custom(environment="MiniGrid-DoorKey-6x6-v0", rounds=100)
dual = run_custom(environment="craftax", algorithm="dual", bc_beta=0.1, rounds=200)
launch_widgets()  # В активном Jupyter frontend.
```

В статическом просмотрщике доступны сохранённые графики.
Анимации и виджеты требуют поддержки JavaScript/ipywidgets; выполните их ячейки в Jupyter.

## Что означают результаты

Это учебный CPU-бюджет и компактная MLP. Craftax Classic использует стандартную карту,
все 17 действий и 46 выбранных семантических предикатов, а не точный набор goals авторов.
Для MiniGrid включено полное наблюдение и фиксирован layout.
Миллиардный бюджет, большие CNN, таблицы Craftax и ускорение 38× из статьи не воспроизводятся.

Learning curves используют фиксированный panel целей, а финальная таблица — все цели.
Проверка запуска не доказывает сходимость; дальний крафт на малом бюджете может не освоиться.
В notebook отдельно описаны ограничения occupancy, logit ranking, goal coverage и evaluation.
