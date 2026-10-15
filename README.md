# Учебная лаборатория Contrastive RL

Откройте [`notebooks/contrastive_rl_ru.ipynb`](notebooks/contrastive_rl_ru.ipynb).
В нём находятся теория, вся реализация, эксперименты и объяснения на русском по статье
*Scaling Contrastive Reinforcement Learning to Open-Ended Environments* из предоставленного PDF.

- Простая среда Grid и более сложная MiniCraft: ключ, крафт, дверь, лава, здоровье.
- Семантические метки, Rao–Blackwell targets, absorbing class и termination discount.
- Абляции CRL; полноценный PPO и совместно обучаемый Dual-CRL-PPO.
- Графики, анимация, инспектор критика, ползунки и настройки новых экспериментов.

Это CPU-версия для изучения механизмов; результаты Craftax из статьи не воспроизводятся.
В notebook явно описаны отличия, ограничения и смысл каждой метрики.

## Запуск

Python 3.10+; проверено с Python 3.12 и PyTorch 2.6 CPU.

```bash
python -m venv .venv
source .venv/bin/activate
# Для машины без GPU:
python -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements-notebook.txt
python -m jupyterlab
```

Выберите kernel этого окружения, откройте notebook и выполните **Restart Kernel and Run All Cells**.
Все вычисления находятся внутри notebook; дополнительных Python-модулей, внешних сервисов,
ключей API и самого PDF во время запуска не требуется.

Основные настройки находятся в `Config`, `BASE`, `SEEDS`, `PPO_ITERATIONS`.
Дополнительные опыты не запускаются автоматически: примеры в `run_custom` можно раскомментировать.
Ползунки требуют поддержки ipywidgets; без неё используйте те же функции с аргументами.

Ползунки открываются вызовом `launch_widgets()` в разделе 10.
В статическом просмотрщике доступны готовые графики; для анимаций и интерактивного
исследования откройте файл в Jupyter и выполните соответствующие ячейки.
