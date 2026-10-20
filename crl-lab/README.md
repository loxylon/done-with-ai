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
