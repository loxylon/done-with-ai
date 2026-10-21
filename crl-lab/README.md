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

## Математические соглашения

`src/math.js` — чистые функции без DOM. Финальная observation смерти исключена из живого будущего (Δ=0 у fatal action); успешная terminal observation учитывается (Δ=1). Time-limit не получает terminal penalty. Вариант без ∅ нормирует по ненулевой semantic mass, как учебный baseline JAX notebook. Полный результат — soft target, а не независимые вероятности выполнения целей.

При γterm=γ и K=Δ восстановлена геометрическая масса до поглощения. Ограничение окна и ненаблюдаемый terminal сохраняют смещение. PPO-пример намеренно изолирует policy clipping + behavioral cloning, опуская value/entropy losses.
