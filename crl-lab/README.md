# CRL Lab — лаборатория будущего

Интерактивный русскоязычный разбор **Scaling Contrastive Reinforcement Learning to Open-Ended Environments**, §3, Eq. 3–9.

Сайт вычисляет все числа в браузере. Это лаборатория точных заданных ситуаций, а не симуляция Craftax и не запуск обученной нейросети.

- Semantic predicates, перекрытие целей, похожие предикаты и пустой goal set.
- Rao–Blackwell: геометрические веса, sampled offsets, MC mean и условная дисперсия.
- Смерть, успешный terminal, time-limit, конец rollout, reset, `Kmax < Δ`, сдвиг anchor.
- Три target-формулы рядом; conservation of mass и остаточный класс ∅.
- Logits против вероятностей; cross-entropy и реальные шаги градиента отдельных logits.
- PPO clipped surrogate для обоих знаков advantage; teacher CE, annealing β, аналитический update.
- Ссылки, сохраняющие параметры ситуации; адаптивная вёрстка и keyboard controls.

## Пошаговая среда

Новая страница: **`walkthrough.html`** (пункт **06 · Пошаговая среда** в меню).
Адрес при включённом GitHub Pages: https://loxylon.github.io/done-with-ai/walkthrough.html.

На поле 4 × 3 есть стена, переключаемая опасность и ровно пять включаемых/выключаемых предикатов: флажок, кристалл, источник, верхний ряд и правая зона. Цели могут пересекаться. В исходном опыте (`seed=1724`) первая траектория показывает награду, три одновременно выполненных предиката и смерть.

Выберите шаг `t`, затем пройдите шесть вкладок: **действие → метки → будущее → критик → GAE/PPO → обновление**. Нажатие на источник слагаемого в target подсвечивает соответствующую клетку и шаг. Можно смотреть каждый PPO epoch, все локальные и общие градиенты, а в инспекторе — любую строку параметров до и после обучения. `Σ update → Следующая итерация` переносит обновлённые параметры в следующий rollout. Уже открытые итерации доступны для повторного просмотра без пересэмплирования. Клавиши ← / → переключают шаги; внутри tablist они переключают формулы.

Это реальное небольшое обучение: факторизованный critic φ(s,a)·ψ(g) с двумя признаками, табличные policy/value и синхронный SGD. Один critic batch-update предшествует нескольким PPO epochs. Есть полный soft CE, fixed-zero ∅, geometric RB, terminal discount, GAE с разными bootstrap/trace-масками, clipped surrogate для обоих знаков advantage, entropy, value loss и линейное уменьшение teacher β. Все шаги одного rollout собираются неизменной policy. Параметры, πold, advantages, returns и teacher labels фиксируются в соответствующих snapshots.

Упрощения: самописная маленькая среда служит только наглядной странице; пакетные Craftax/MiniGrid notebook не меняются. Здесь команда фиксирована на весь rollout и награда начисляется при каждом посещении предиката, а не только при первом достижении. Нейросетей, Adam/RAdam и advantage normalization нет. Каждый rollout начинается из стартовой клетки; его лимит — time-limit с value bootstrap. Набор целей и параметры опыта можно изменить с явным пересчётом с начала. При выключении всех целей p∅ = y∅ = 1, critic update нулевой, PPO пропускается.

Кнопка **«Ссылка на опыт»** сохраняет настройки, открытую итерацию, шаг и вкладку; **«Все числа · JSON»** экспортирует все открытые итерации без округления, включая полные tensors и gradients каждого epoch. Показанные числа округлены до пяти десятичных знаков; полное значение отдельного числа видно при наведении.

- `src/walkthrough-engine.js` — чистые правила среды, forward passes, targets, аналитические gradients и snapshots.
- `src/walkthrough-view.js` — визуализация происхождения чисел и таблиц; не меняет параметры модели.
- `src/walkthrough.js` — навигация по журналу, настройки, ссылки и экспорт.
- `tests/walkthrough.test.js` — геометрическая масса, все 32 подмножества целей, конечные разности critic/PPO/value, повторные посещения, GAE и неизменность истории.
- `tests/walkthrough.spec.js` — весь пошаговый путь, несколько итераций, goal switches, восстановление ссылок, JSON и мобильные размеры.

## Локально

Node.js 22+:

```sh
cd crl-lab
npm ci
npm run dev
```

Сервер использует порт 4173. Изменили исходники — повторите `npm run build`.
Никакие API-ключи не нужны. KaTeX и Manrope со своими лицензиями копируются в статический build: внешних runtime CDN нет.

## Проверка

```sh
npm test
npm run build
npm run test:browser
```

Для браузерных тестов по умолчанию используется `/usr/bin/chromium`. Можно задать `CHROMIUM_PATH`.
В CI Playwright устанавливает свой Chromium (`npx playwright install --with-deps chromium`).

Численные проверки охватывают JAX fixture из notebook, множество anchor/boundary/discount/cap комбинаций, отсутствие leakage после reset, нормировку, MC convergence и finite-difference проверки градиентов CE/PPO. Браузерные тесты проходят все разделы, проверяют фактические изменения, мобильные размеры, отсутствие ошибок формул и внешних runtime-запросов.

## GitHub Pages

Build находится в `dist/`, исходники — в `src/` и `styles.css`.

```sh
npm run build
npm test
npm run test:browser
npm run publish:pages
```

Последняя команда публикует **только dist** в ветку `gh-pages`, не меняя индекс `main` и не выполняя force-push. Сохранены предыдущие deployment-коммиты. Токены в файлы не записываются.

Одноразовая настройка репозитория для автоматической публикации: **Settings → Pages → Build and deployment → Source → GitHub Actions**. Затем запустите **Actions → Publish CRL Lab → Run workflow**, если первый запуск завершился до включения Pages.
Доступность GitHub Pages для приватного репозитория зависит от тарифа GitHub; скрипт не меняет visibility.
После включения адрес: **https://loxylon.github.io/done-with-ai/**.

Workflow `.github/workflows/crl-lab-pages.yml` автоматически проверяет сайт, обновляет резервную ветку `gh-pages`, затем публикует проверенный artifact через официальные `upload-pages-artifact` / `deploy-pages`. Это нужно, потому что push с workflow-токеном сам по себе не запускает сборку Pages. Для разовой публикации без workflow можно выбрать **Deploy from a branch → gh-pages → / (root)**. Публикация файлов в ветку и фактическое включение Pages — разные операции: успешный push не доказывает, что публичная страница уже доступна.

## MEGA / OMEGA · ICML 2020

Отдельный сайт **`mega.html`** разбирает новую статью: Silviu Pitis, Harris Chan, Stephen Zhao, Bradly Stadie, Jimmy Ba, *Maximum Entropy Gain Exploration for Long Horizon Multi-goal Reinforcement Learning*. PMLR 119, ICML 2020.

Адрес при включённом Pages: **https://loxylon.github.io/done-with-ai/mega.html**.

Пять разделов:

1. **Проблема:** почему HER не создаёт опыт из ненаблюдавшейся области; forward KL и несовпадение поддержек.
2. **SELECT:** редактируемая гистограмма пяти целей → выборка N кандидатов → фильтр достижимости → минимум плотности. На одном pool сравниваются HER/Desired, Achieved, Diverse, MEGA и OMEGA. Сохраняются повторы кандидатов; у Diverse вес каждого экземпляра равен 1/pAG. Для MEGA отдельно вычислена точная вероятность победы по многим повторным выборкам, включая ties и отсутствие допустимого кандидата.
3. **Среда:** реальные независимые табличные Q-learning агенты, общий warmup, одинаковые seed, бюджет и future HER 4:1. Коридор имеет пять goal regions и 1–4 физических шага между соседними целями. Можно запускать 1/10/100 эпизодов, смотреть любой переход, Q, ε-greedy, Go Exploration, реальный replay и шесть примеров TD updates. Кривые показывают эмпирическую H[pAG]; greedy evaluation и первое посещение E — отдельные показатели. Данные опыта можно передать в инспектор.
4. **Энтропия:** точный finite-buffer H, Eq.5 score и ожидание по редактируемому q(g′|ĝ). Сравнение с oracle показывает, почему rare-but-unreachable goal ломает minimum-density approximation. Неизвестная цель допускается как новый исход, но не как буферный behavioral candidate. Показаны Propositions 1–3 и разница между score и фактической ΔH.
5. **OMEGA:** декомпозиция KL, α, bias b, смесь objective pα и действительное behavioral distribution. Есть presets разрыва поддержек, первого контакта и покрытия внешней цели. При пустом буфере или пустом desired distribution objective явно не определён.

Исходные материалы прочитаны из официального репозитория PMLR (тот же издательский PDF, поскольку домен proceedings.mlr.press заблокирован прокси этой среды):

- [Статья](https://proceedings.mlr.press/v119/pitis20a/pitis20a.pdf), [официальная копия PDF](https://raw.githubusercontent.com/mlresearch/v119/gh-pages/pitis20a/pitis20a.pdf): §2–3, Eq.1–7, Algorithm 2, §4 Fig.3–6.
- [Supplement](https://proceedings.mlr.press/v119/pitis20a/pitis20a-supp.pdf), [официальная копия](https://raw.githubusercontent.com/mlresearch/v119/gh-pages/pitis20a/pitis20a-supp.pdf): B (доказательства), C (KDE, cutoff, Go Exploration, baseline SELECT, rfaab).
- [Репозиторий авторов](https://github.com/spitis/mrl).

Упрощения явно подписаны на сайте: histogram вместо KDE; Q-table вместо DDPG; goal-terminal TD target; future HER вместо rfaab; известная проходимость вместо обученного cutoff в rollout. Редактируемая учебная Q-оценка и динамический cutoff статьи объясняются в инспекторе отдельно. Никакой предобученной навигации нет. Один seed на игрушечном коридоре не воспроизводит бенчмарки статьи и не доказывает универсальное превосходство. Сравнение с опубликованными PPO+SR результатами дополнительно оговаривает разницу базовых алгоритмов.

Кнопка **«Ссылка на сценарий»** сохраняет параметры, буфер, q, выбранную страницу, количество обученных эпизодов и просматриваемый шаг. Восстановление пересчитывает опыт детерминированно. Всё работает локально, runtime-запросов к CDN нет.

`src/mega/engine.js` содержит чистую математику и исполняемое обучение, `view.js` — визуализацию, `app.js` — управление. `tests/mega.test.js` проверяет finite-pool probabilities через Monte Carlo, точную нормировку entropy gain, KL/α, воспроизводимость и TD/HER semantics. `tests/mega.spec.js` проходит пять разделов, edge cases, обучение, восстановление ссылки и мобильные размеры. Существующие страницы CRL и JAX notebooks сохранены.

## Математические соглашения

`src/math.js` — чистые функции без DOM. Финальная observation смерти исключена из живого будущего (Δ=0 у fatal action); успешная terminal observation учитывается (Δ=1). Time-limit не получает terminal penalty. Вариант без ∅ нормирует по ненулевой semantic mass, как учебный baseline JAX notebook. Полный результат — soft target, а не независимые вероятности выполнения целей.

При γterm=γ и K=Δ восстановлена геометрическая масса до поглощения. Ограничение окна и ненаблюдаемый terminal сохраняют смещение. PPO-пример намеренно изолирует policy clipping + behavioral cloning, опуская value/entropy losses.
