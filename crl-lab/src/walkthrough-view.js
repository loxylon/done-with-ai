/** Представление журнала. Здесь нет обучения: все числа приходят из engine. */
import { ACTIONS, GOALS, WALL, DANGER, BOUNDS, coordinates, semantics, criticForward } from './walkthrough-engine.js';
import { sum, softmax } from './math.js';

export const STAGES = ['Действие', 'Метки', 'Будущее', 'Критик', 'GAE / PPO', 'Обновление'];
export const escape = value => String(value).replace(/[&<>"']/g, c => ({ '&':'&amp;', '<':'&lt;', '>':'&gt;', '"':'&quot;', "'":'&#39;' }[c]));
export const format = (value, digits = 5) => Number.isFinite(value) ? value.toLocaleString('ru-RU', { maximumFractionDigits: digits }) : '∞';
const number = value => `<span title="${value}">${format(value)}</span>`;
const percent = value => `${format(value * 100, 2)}%`;
const n = value => {
  if (value === null) return '\\infty';
  if (Math.abs(value) > 0 && Math.abs(value) < .00001) {
    const [mantissa, exponent] = value.toExponential(3).split('e');
    return String.raw`${mantissa}\cdot 10^{${Number(exponent)}}`;
  }
  return Number(value.toFixed(5)).toString();
};
const math = (label, source) => `<div class="math-box"><div class="math-label">${label}</div><div class="math" data-tex="${escape(source)}"></div></div>`;
const note = text => `<p class="trace-note">${text}</p>`;
const callout = (text, kind = '') => `<div class="trace-callout ${kind}">${text}</div>`;
const detail = (title, body, open = false) => `<details class="trace-detail"${open ? ' open' : ''}><summary>${title}</summary>${body}</details>`;
const point = s => `(${coordinates(s).join(', ')})`;
const actionName = a => `a${a} · ${ACTIONS[a].name}`;
const goalName = g => g < 0 ? 'нет команды' : `${GOALS[g].id} · ${GOALS[g].name}`;
const goalHeaders = () => [...GOALS.map(g => g.id), '∅'];
const table = (headers, rows, classes = []) => `<div class="trace-table-wrap"><table class="trace-table"><thead><tr>${headers.map(h => `<th scope="col">${h}</th>`).join('')}</tr></thead><tbody>${rows.map((row, i) => `<tr class="${classes[i] || ''}">${row.map((cell, j) => j === 0 ? `<th scope="row">${cell}</th>` : `<td>${cell}</td>`).join('')}</tr>`).join('')}</tbody></table></div>`;
const values = entries => `<div class="trace-values">${entries.map(([label, value, kind = '']) => `<div class="trace-value ${kind}"><small>${label}</small><strong>${value}</strong></div>`).join('')}</div>`;
const vector = xs => `<div class="trace-vector">${xs.map((x, g) => `<div class="trace-vector-item"><span style="color:${GOALS[g]?.color || '#4d5845'}">${GOALS[g]?.id || '∅'}</span>${number(x)}</div>`).join('')}</div>`;
const distribution = xs => `<div class="trace-distribution" aria-label="${xs.map((x,g) => `${GOALS[g]?.id || '∅'} ${percent(x)}`).join('; ')}">${xs.map((x, g) => `<span title="${GOALS[g]?.id || '∅'}: ${percent(x)}" style="width:${100*x}%;background:${GOALS[g]?.color || '#424b3d'}">${x > .09 ? `${GOALS[g]?.id || '∅'} ${percent(x)}` : ''}</span>`).join('')}</div>`;
const policyComparison = (before, after, labels = ['До', 'После']) => `<div class="trace-policy-comparison"><div class="trace-note">${labels[0]} — светлая полоса; ${labels[1]} — зелёная.</div>${ACTIONS.map((action,a)=>`<div class="trace-policy-row"><span>a${a} · ${action.name}</span><div><div class="trace-policy-track"><i style="width:${100*before[a]}%"></i></div><div class="trace-policy-track after"><i style="width:${100*after[a]}%"></i></div></div><span>${percent(before[a])}<br><b>${percent(after[a])}</b></span></div>`).join('')}</div>`;

export function scenario(config) {
  const specs = [
    ['seed', 'Seed', 'number', 1, '1724: в первом rollout видны награда, пересечение целей и смерть.'],
    ['horizon', 'Лимит шагов', 'number', 1, 'Конец лимита — time-limit, не смерть.'],
    ['iterations', 'Всего итераций', 'number', 1, 'β уменьшается от первой к последней.'],
    ['epochs', 'PPO epochs', 'number', 1, 'Повторные обновления по одному rollout.'],
    ['gamma', 'Дисконт γ', 'number', .01, 'Веса будущего и value bootstrap.'],
    ['terminalGamma', 'Дисконт конца γterm', 'number', .01, 'γterm = γ даёт исходную геометрическую массу при полном будущем.'],
    ['lambda', 'GAE λ', 'number', .05, 'Как далеко распространяется TD-ошибка.'],
    ['cap', 'Максимальный offset Kmax', 'number', 1, 'Не меняет расстояние до известной смерти.'],
    ['criticLR', 'Шаг критика ηC', 'number', .01, 'Для φ и ψ; один средний batch-gradient.'],
    ['actorLR', 'Шаг policy ηπ', 'number', .01, 'После каждого PPO epoch.'],
    ['valueLR', 'Шаг value ηV', 'number', .01, 'Отдельная таблица V.'],
    ['clip', 'PPO clip ε', 'number', .05, 'Граница ratio вокруг 1.'],
    ['beta', 'Teacher β₀', 'number', .05, '0 отключает подсказку, но не critic update.'],
    ['entropy', 'Коэффициент entropy cH', 'number', .005, 'Поощрение разнообразия policy.'],
    ['valueCoefficient', 'Коэффициент value cV', 'number', .05, 'Вес квадратичной ошибки V.'],
  ];
  return `<div class="trace-scenario"><div class="trace-scenario-head"><div><h2>Пять целей. Один общий мир.</h2><p>Добавляйте и убирайте предикаты. Набор целей меняет и метки, и softmax; опыт пересчитается с начала.</p></div><button class="trace-button compact" data-command="default">Исходный опыт</button></div>
    <div class="trace-goals">${GOALS.map((g, i) => `<label class="trace-goal ${config.enabled[i] ? '' : 'off'}"><input type="checkbox" data-goal-toggle="${i}" ${config.enabled[i] ? 'checked' : ''}><span class="trace-goal-dot" style="background:${g.color}">${g.id}</span><span><strong>${g.name}</strong><small>${i < 3 ? ['(2, 0)', '(3, 2)', '(0, 2)'][i] : i === 3 ? 'y = 0' : 'x ≥ 2'}</small></span></label>`).join('')}</div>
    <details class="trace-config"><summary>Параметры · γ ${format(config.gamma)} · γterm ${format(config.terminalGamma)} · ${config.horizon} шагов · ${config.iterations} итераций</summary><form id="trace-config-form"><div class="trace-config-grid">
      <label>Команда агенту<select name="command" ${config.command < 0 ? 'disabled' : ''}>${GOALS.map((g, i) => config.enabled[i] ? `<option value="${i}" ${i === config.command ? 'selected' : ''}>${g.id} · ${g.name}</option>` : '').join('') || '<option value="-1">Нет активных целей</option>'}</select><small>CRL учит все цели; reward и PPO — только эту.</small></label>
      <label>Опасная клетка<select name="danger"><option value="true" ${config.danger ? 'selected' : ''}>Смерть в (3, 1)</option><option value="false" ${!config.danger ? 'selected' : ''}>Безопасна</option></select><small>Её observation при смерти исключается из будущего.</small></label>
      ${specs.map(([key, label, type, step, help]) => `<label>${label}<input name="${key}" type="${type}" min="${BOUNDS[key][0]}" max="${BOUNDS[key][1]}" step="${step}" value="${config[key]}" required><small>${help}</small></label>`).join('')}</div><div class="trace-config-actions"><button class="trace-button primary" type="submit">Применить и начать заново</button><span>Seed тот же: меняйте один параметр для сравнения.</span></div></form></details></div>`;
}

function map(iteration, cursor, source) {
  const step = iteration.steps[Math.min(cursor, iteration.steps.length - 1)];
  const completed = cursor === iteration.steps.length;
  const config = iteration.config;
  const center = s => { const [x,y] = coordinates(s); return [48 + x * 76, 48 + y * 76]; };
  const [sx,sy] = center(step.state), [nx,ny] = center(step.next);
  const path = iteration.steps.slice(0, cursor).map(s => center(s.next));
  const points = [center(iteration.steps[0].state), ...path].map(p => p.join(',')).join(' ');
  const [dx,dy] = [ACTIONS[step.action].dx, ACTIONS[step.action].dy];
  const ex = step.blocked ? sx + dx * 24 : nx;
  const ey = step.blocked ? sy + dy * 24 : ny;
  const focus = source === null ? null : iteration.steps[source];
  return `<svg class="trace-map" viewBox="0 0 328 258" role="img" aria-label="Поле 4 на 3. Агент ${point(completed ? step.next : step.state)}, действие ${ACTIONS[step.action].name}, следующая клетка ${point(step.next)}${step.fatal ? ', смерть' : step.blocked ? ', движение заблокировано' : ''}">
    <defs><marker id="step-arrow" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="4" markerHeight="4" orient="auto-start-reverse"><path d="M0 0L10 5L0 10z" fill="#d46c47"/></marker></defs>
    ${Array.from({length:12}, (_, s) => {
      const [cx,cy] = center(s); const predicates = semantics(s, config.enabled).predicates;
      return `<g><rect class="map-cell" x="${cx-33}" y="${cy-33}" width="66" height="66" rx="9" fill="${s === WALL ? '#d3d8ca' : s === DANGER && config.danger ? '#f5dfd4' : '#fffefa'}"/>
        <text x="${cx-27}" y="${cy-20}" font-size="7" fill="#9da68e">${point(s)}</text>
        ${s === WALL ? `<path d="M${cx-14} ${cy-8}h28m-28 8h28m-28 8h28" stroke="#acb69d" stroke-width="3"/><text x="${cx}" y="${cy+24}" text-anchor="middle" font-size="7" fill="#7d886e">СТЕНА</text>` :
          `<g>${predicates.map((on,g) => on ? `<rect x="${cx-28+g*11}" y="${cy+20}" width="10" height="10" rx="2" fill="${GOALS[g].color}"/><text x="${cx-23+g*11}" y="${cy+27}" text-anchor="middle" font-size="6.5" font-weight="700" fill="#fff">${GOALS[g].id}</text>` : '').join('')}</g>`}
        ${s === DANGER && config.danger ? `<path d="M${cx-9} ${cy-10}l18 18m0-18l-18 18" stroke="#be7859" stroke-width="2"/>` : ''}
      </g>`;
    }).join('')}
    <polyline class="map-visit" points="${points}"/>
    ${focus ? (()=>{const [x,y]=center(focus.next);return `<rect class="source-halo" x="${x-32}" y="${y-32}" width="64" height="64" rx="10"/>`;})() : ''}
    ${!completed ? `<path class="map-arrow" d="M${sx} ${sy}L${ex} ${ey}" marker-end="url(#step-arrow)"/>` : ''}
    <circle cx="${completed ? nx : sx}" cy="${completed ? ny : sy}" r="13" fill="#4b6041" stroke="#fffefa" stroke-width="2"/>
    <path d="M-4 4L0-5L4 4L0 2z" fill="#fff" transform="translate(${completed ? nx : sx} ${completed ? ny : sy}) rotate(${step.action*90})"/>
    ${!completed && !step.blocked ? `<circle cx="${nx}" cy="${ny}" r="16" fill="none" stroke="${step.fatal ? '#b86149' : '#d46c47'}" stroke-width="2" stroke-dasharray="3 2"/>` : ''}
    ${step.fatal && completed ? `<text x="${nx}" y="${ny+5}" text-anchor="middle" font-size="18" fill="white">×</text>` : ''}
  </svg>`;
}

export function world(context) {
  const { iteration: it, history, iterationIndex, cursor, source } = context;
  const s = it.steps[Math.min(cursor, it.steps.length-1)];
  const update = cursor === it.steps.length;
  const nextLabel = update ? iterationIndex+1 === it.config.iterations ? 'Опыт разобран' : 'Следующая итерация' : cursor === it.steps.length-1 ? 'Посмотреть обновление' : 'Следующий шаг';
  return `<div class="trace-box"><div class="trace-map-section"><div class="trace-world-head"><span class="trace-kicker">МИР 4 × 3</span><span class="small-badge" data-testid="iteration-label">Итерация ${iterationIndex+1} / ${it.config.iterations}</span></div>
    <div class="trace-iteration-pills" aria-label="Просмотр итераций">${Array.from({length:it.config.iterations},(_, i)=>`<button class="trace-button compact" data-iteration="${i}" aria-label="Итерация ${i+1}" aria-pressed="${i===iterationIndex}" ${i>=history.length?'disabled':''}>${i+1}</button>`).join('')}</div>
    ${map(it,cursor,source)}<div class="trace-map-legend"><span><i class="trace-key"></i>агент до</span><span><i class="trace-key after"></i>после шага</span><span><i class="trace-key source"></i>источник числа</span></div></div>
    <div class="trace-navigation-section"><div class="trace-step-readout"><span class="trace-kicker">${update?'ОБНОВЛЕНИЕ ПАРАМЕТРОВ':`ШАГ ${cursor+1} ИЗ ${it.steps.length}`}</span><p><strong>${update?'Из данных — в следующую итерацию':`${point(s.state)} → ${point(s.next)}`}</strong></p><p>${update ? 'Один batch-update критика, затем '+it.config.epochs+' PPO epochs. Следующий rollout использует итоговые параметры.' : `${actionName(s.action)}. ${s.fatal ? 'Попадание в опасность: настоящий terminal.' : s.blocked ? 'Стена или граница: агент остаётся на месте.' : 'Движение в соседнюю клетку.'}`}</p></div>
    <div class="trace-steps" aria-label="Шаги rollout">${it.steps.map((_,t)=>`<button class="trace-step ${source===t?'source':''}" data-step="${t}" aria-pressed="${cursor===t}" aria-label="Шаг t=${t}">t${t}</button>`).join('')}<button class="trace-step" data-step="${it.steps.length}" aria-pressed="${update}" aria-label="Итоговое обновление">Σ update</button></div>
    <div class="trace-playback"><button class="trace-button" data-command="previous-step" aria-label="Предыдущий шаг" ${cursor===0&&iterationIndex===0?'disabled':''}><svg width="14" height="14" viewBox="0 0 16 16" aria-hidden="true"><path d="M13 8H3m5-5L3 8l5 5" fill="none" stroke="currentColor" stroke-width="1.5"/></svg></button><button class="trace-button primary" data-command="next-step" ${update&&iterationIndex+1===it.config.iterations?'disabled':''}>${nextLabel}</button></div>
    <p class="trace-next-help">Команда: <b>${goalName(it.config.command)}</b>.<br>Каждый шаг справа имеет шесть вкладок. Клавиши ← / → переключают шаги.</p></div>
    <div class="trace-small-stats"><div>Награда rollout<strong>${it.reward}</strong></div><div>Окончание<strong>${it.fatal?'Смерть':'Лимит'}</strong></div><div>Teacher β<strong>${format(it.beta,3)}</strong></div></div>`;
}

function actionPanel(context) {
  const { iteration: it, t } = context;
  const step = it.steps[t], teacher = it.teachers[t];
  const rows = ACTIONS.map((a, i) => [actionName(i), number(step.logits[i]), number(Math.exp(step.logits[i])), number(step.policy[i]),
    `${format(i ? step.cdf[i-1] : 0)} ≤ u &lt; ${format(step.cdf[i])}`, i===step.action?'Сэмплировано':'']);
  return note(`Это действие выбрала <b>πold</b>, замороженная на весь rollout. Команда — <b>${goalName(it.config.command)}</b>. Critic teacher пока не выбирает движение: он участвует в обновлении policy после сбора.`)
    + math('ПОЛИТИКА: ТАБЛИЧНЫЕ LOGITS → SOFTMAX', String.raw`\pi_{\rm old}(a\mid s_t,g)=\frac{e^{\theta^{\rm old}_{s_t,g,a}}}{\sum_{b=0}^{3}e^{\theta^{\rm old}_{s_t,g,b}}}`)
    + table(['Действие','θold','exp θold','πold','Интервал CDF','Выбор'], rows, ACTIONS.map((_,a)=>a===step.action?'chosen':''))
    + `<div class="trace-cdf">${step.policy.map((p,a)=>`<span class="${a===step.action?'chosen':''}" style="width:${p*100}%">a${a}</span>`).join('')}<i class="trace-cdf-marker" style="left:${step.u*100}%"><small style="transform:translateX(${step.u<.12?'0':step.u>.88?'-100%':'-50%'})">u = ${format(step.u)}</small></i></div>`
    + math('ПОДСТАНОВКА ДЛЯ ВЫБРАННОГО ДЕЙСТВИЯ',String.raw`\pi_{\rm old}(a_${step.action})=\frac{e^{${n(step.logits[step.action])}}}{${step.logits.map(x=>`e^{${n(x)}}`).join('+')}}\approx ${n(step.policy[step.action])}`)
    + callout(`Seed ${it.config.seed}, итерация ${it.index+1}, случайное число u = ${number(step.u)}. Оно попало в интервал действия <b>${actionName(step.action)}</b>. При возвращении к шагу u сохраняется; это просмотр журнала, а не новый бросок.`)
    + note(`После движения: ${point(step.state)} → ${point(step.next)}${step.blocked?' (координата попытки '+step.attempted.join(', ')+', но проход закрыт)':''}. Награда = ${step.reward}. ${step.fatal?'Поглощение: дальнейших переходов в этом эпизоде нет.':step.truncated?'Лимит: состояние живое, но сбор закончен.':'Среда продолжает работу.'}`)
    + detail('Посмотреть f(sₜ, a, g) критика ДО обучения: все действия и все пять целей', table(['Действие',...GOALS.map(g=>g.id),'f∅'], ACTIONS.map((_,a)=>[actionName(a),...criticForward(it.before,step.state,a,it.config.enabled).scores.map((f,g)=>it.config.enabled[g]?number(f):'выключена'),0])))
    + detail('А что посоветует teacher ПОСЛЕ critic update?', table(['Действие','fnew(sₜ,a,команда)','Teacher'],teacher.scores.map((f,a)=>[actionName(a),it.config.command<0?'—':number(f),a===teacher.action?'argmax':'']))+note('Эти scores появятся после обновления критика на вкладке 4. Они не использовались для выбора уже записанного действия.'));
}

function semanticPanel({iteration:it,t}) {
  const step = it.steps[t], semantic = step.semantic;
  const rows = GOALS.map((g,i)=>[`${g.id} · ${g.name}`,it.config.enabled[i]?'Включена':'Исключена',
    step.fatal?'Поглощение':g.test(step.next)?'Истина':'Ложь',semantic.predicates[i]?1:0,
    semantic.predicates[i]?`1 / ${semantic.count}`:'0',number(semantic.labels[i])]);
  return note(`Проверяем предикаты в <b>следующем состоянии s${t+1} = ${point(step.next)}</b>. Одна observation может выполнять несколько целей. У выключенных целей P = 0; они не участвуют в знаменателе и softmax.`)
    + table(['Предикат','В каталоге','Проверка клетки','Pᵢ','Деление','ℓᵢ'],rows)
    + math('ФОРМУЛА 4 · РАЗДЕЛЕНИЕ ОДНОЙ ЕДИНИЦЫ МАССЫ',String.raw`\ell_g(s)=\begin{cases}P_g(s)/|G^+(s)|,&|G^+(s)|>0\\0,&|G^+(s)|=0\end{cases}`)
    + values([['Выполнено предикатов |G⁺|',semantic.count],['Сумма живых меток',format(sum(semantic.labels))],['Остаток этого state',format(semantic.residual),'orange']])
    + vector([...semantic.labels,semantic.residual]) + distribution([...semantic.labels,semantic.residual])
    + callout(step.fatal?'Смертельная observation не считается живым будущим. Её предикаты принудительно нулевые, даже если координата лежит в правой зоне. У самого fatal anchor target ∅ = 1.':semantic.count===0?'Ни один активный предикат не выполнен. Деления на ноль нет: все ℓ = 0. Эта observation создаст residual mass ∅ в target, хотя агент жив.':`Здесь делим массу поровну между ${semantic.count} выполненными предикатами: ${GOALS.filter((_,g)=>semantic.predicates[g]).map(g=>g.id).join(', ')}. Цели категориальные: метки не являются независимыми вероятностями успеха.`)
    + math('НАГРАДА PPO И МЕТКИ CRL — РАЗНЫЕ ОБЪЕКТЫ',it.config.command<0?String.raw`g=\text{нет команды},\quad r_t=0`:String.raw`r_t=\mathbf1[P_{${GOALS[it.config.command].id}}(s_{t+1})]=${step.reward},\qquad\ell_{${GOALS[it.config.command].id}}(s_{t+1})=${n(semantic.labels[it.config.command])}`)
    + note('Награда проверяет только команду и не делится на число предикатов. CRL reward не использует: он классифицирует всё наблюдаемое семантическое будущее.');
}

function futurePanel({iteration:it,t,source,focusGoal}) {
  const target = it.critic.anchors[t].target, config = it.config;
  const g = focusGoal;
  const terms = target.futures.map(f=>String.raw`${n(f.weight)}\cdot${n(f.semantic.labels[g])}`);
  const sumTerms = terms.join('+') || '0';
  const rows = target.futures.map(f=>[
    `<button data-source-step="${f.t}" aria-label="Подсветить источник: следующий state шага ${f.t}">s${f.t+1} ${point(f.next)}</button>`,
    f.k,number(f.power),number(f.raw),number(f.weight),f.semantic.predicates.map((p,i)=>p?GOALS[i].id:'').filter(Boolean).join(', ')||'нет',
    number(f.semantic.labels[g]),number(f.contribution[g]),number(f.corrected[g]),
  ]);
  return note('Будущее берётся из <b>уже собранной</b> траектории, начиная со следующей observation этого действия. Нажмите на s в таблице — подсветятся клетка и шаг, откуда пришло слагаемое. Reset и observation смерти в сумму не попадают.')
    + `<label class="trace-note">Разложить по слагаемым цель <select class="trace-inline-select" data-focus-goal aria-label="Цель для разложения target">${GOALS.map((goal,i)=>`<option value="${i}" ${i===g?'selected':''}>${goal.id} · ${goal.name}${config.enabled[i]?'':' (выключена, все метки 0)'}</option>`).join('')}</select></label>`
    + values([['Живых будущих K',target.futures.length],['До смерти Δ',target.distance===null?'∞':target.distance],['Нормировка 1−γᴷ',format(target.normalizer)]])
    + math('ФОРМУЛА 6 · ВЕС КАЖДОГО ДОПУСТИМОГО OFFSET',target.futures.length?String.raw`w_k=\frac{(1-\gamma)\gamma^{k-1}}{1-\gamma^K},\quad\gamma=${n(config.gamma)},\quad K=${target.futures.length}`:String.raw`K=0,\quad\bar y_t(g)=0\quad\text{(пустая сумма; веса не вычисляются)}`)
    + (rows.length ? table(['Источник','k','γᵏ⁻¹','raw','wₖ','G⁺','ℓ'+GOALS[g].id,'wₖℓ','После ×m'],rows,target.futures.map(f=>f.t===source?'focused':'')) : callout('Нет живого будущего. Пустая сумма равна 0. Не вычисляем w с нулевым знаменателем.', 'orange'))
    + math('РАО–БЛЭКВЕЛЛ: ТОЧНАЯ СУММА ВНУТРИ ОКНА',String.raw`\bar y_t(${GOALS[g].id})=${sumTerms}\approx ${n(target.mean[g])}`)
    + math('ФОРМУЛА 8 · ПЕРЕНОС КОНЦА',target.distance===null?String.raw`m=1\quad\text{(смерть не наблюдалась)},\quad y_t(${GOALS[g].id})=1\cdot${n(target.mean[g])}`:String.raw`m=1-${n(config.terminalGamma)}^{${target.distance}}\approx${n(target.retained)},\quad y_t(${GOALS[g].id})=m\cdot${n(target.mean[g])}\approx${n(target.y[g])}`)
    + table(['Вектор',...goalHeaders()],[['Среднее по окну',...target.mean.map(number),number(1-sum(target.mean))],['С учётом конца',...target.y.map(number)]])
    + distribution(target.y)
    + math('СОХРАНЕНИЕ МАССЫ',String.raw`y_t(\varnothing)=1-\sum_{g\in G}y_t(g)\approx${n(target.y[5])}`)
    + callout(`Остаток ∅ = ${number(target.terminalMass)} от наблюдаемого конца + ${number(target.missingSemantic)} от пустых семантических состояний. Это не вероятность смерти на следующем шаге.`, 'purple')
    + detail('Почему некоторые переходы исключены?',table(['Переход','Следующая клетка','Используется?','Причина'],it.steps.map((s,i)=>[i,point(s.next),target.futures.some(f=>f.t===i)?'Да':'Нет',i<t?'Раньше anchor':s.fatal?'Observation смерти':i>=t+config.cap?'За Kmax':'Внутри окна'])))
    + note(`При γterm = γ и K = Δ нормировка сокращается до исходной геометрической массы. ${target.distance!==null&&target.futures.length<target.distance?'Сейчас Kmax меньше Δ: расстояние до известной смерти сохраняется, но finite-window bias остаётся.':target.distance===null?'Здесь конец по лимиту: за ненаблюдаемое будущее terminal penalty не добавляется.':'Здесь доступно всё живое будущее до смерти.'}`);
}

function criticPanel({iteration:it,t}) {
  const s=it.steps[t], a=it.critic.anchors[t], forward=a.forward, T=it.steps.length;
  const rows=GOALS.map((g,i)=>[`${g.id}${it.config.enabled[i]?'':' · off'}`,
    number(it.before.psi[i][0]),number(it.before.psi[i][1]),
    `${format(forward.products[i][0])} + ${format(forward.products[i][1])}`,number(forward.scores[i]),
    it.config.enabled[i]?number(Math.exp(forward.scores[i])):'исключён',number(forward.probabilities[i]),number(a.target.y[i]),number(a.terms[i]),number(a.scoreGradient[i])]);
  rows.push(['∅','—','—','фиксирован',0,1,number(forward.probabilities[5]),number(a.target.y[5]),number(a.terms[5]),'не обучается']);
  return note(`Критик читает пару <b>(s${t} = ${point(s.state)}, ${actionName(s.action)})</b>. Её trainable вектор φ = [${forward.phi.map(x => format(x)).join('; ')}]. Векторы ψ общие для всех состояний. Все произведения ниже посчитаны из параметров <b>до обновления</b>.`)
    + math('ФАКТОРИЗОВАННЫЙ CRITIC SCORE',String.raw`f(s_t,a_t,g)=\underbrace{\phi_1}_{${n(forward.phi[0])}}\psi_{g,1}+\underbrace{\phi_2}_{${n(forward.phi[1])}}\psi_{g,2}`)
    + table(['Цель','ψ₁','ψ₂','φ₁ψ₁ + φ₂ψ₂','f','exp f','p','y','−y log p','p−y'],rows)
    + `<h3>Что предсказывается и чего требует будущее</h3>`+note('Верхняя полоса — предсказание p критика, нижняя — target y из траектории. Порядок классов одинаковый; ∅ справа.')+distribution(forward.probabilities)+distribution(a.target.y)
    + math('ФОРМУЛА 7 · КЛАСС ∅ ДОБАВЛЯЕТ 1 В ЗНАМЕНАТЕЛЬ',String.raw`Z=1+\sum_{g\in G}e^{f_g}\approx${n(Math.exp(forward.logZ))},\quad p_g=e^{f_g}/Z,\quad p_\varnothing=1/Z`)
    + math('ФОРМУЛА 3 · SOFT CROSS-ENTROPY',String.raw`L_{C,t}=-\sum_{g\in G\cup\{\varnothing\}}y_t(g)\log p_t(g)\approx${n(a.loss)}`)
    + callout(`Для fatal anchor y∅ = 1: loss = log Z, и ∂L/∂f_g = p_g &gt; 0 для всех активных целей. SGD понижает scores смертельного действия через обновления φ и ψ. ${s.fatal?'Сейчас открыт именно такой anchor.':'Выберите последний шаг первой итерации, чтобы увидеть этот случай.'}`)
    + math('ЦЕПОЧКА ПРОИЗВОДНЫХ · ЛОКАЛЬНЫЙ ВКЛАД ЭТОГО ШАГА',String.raw`\frac{\partial L_{C,t}}{\partial f_g}=p_g-y_g,\quad\frac{\partial L_{C,t}}{\partial\phi_j}=\sum_g(p_g-y_g)\psi_{g,j},\quad\frac{\partial L_{C,t}}{\partial\psi_{g,j}}=(p_g-y_g)\phi_j`)
    + table(['j','Слагаемые ∂Lₜ/∂φⱼ','Локальная сумма','Вклад / T','Batch для этой (s,a)','φ до','φ после'],[0,1].map(j=>[j+1,
      a.scoreGradient.map((v,g)=>`${format(v)}×${format(it.before.psi[g][j])}`).join(' + '),number(a.localPhi[j]),number(a.localPhi[j]/T),
      number(it.critic.phiGradient[s.state][s.action][j]),number(forward.phi[j]),number(it.critic.after.phi[s.state][s.action][j])]))
    + table(['Цель','Локальный ∂ψ₁','Локальный ∂ψ₂','Batch ∂ψ₁','Batch ∂ψ₂','ψnew₁','ψnew₂'],GOALS.map((g,i)=>[g.id,...a.localPsi[i].map(number),...it.critic.psiGradient[i].map(number),...it.critic.after.psi[i].map(number)]))
    + math('ПАКЕТНОЕ ОБНОВЛЕНИЕ, НЕ ПО ОДНОМУ ANCHOR',String.raw`\nabla L_C=\frac1{${T}}\sum_{t=0}^{${T-1}}\nabla L_{C,t},\qquad(\phi,\psi)_{\rm new}=(\phi,\psi)_{\rm old}-${n(it.config.criticLR)}\nabla L_C`)
    + note('При повторном посещении той же пары (s,a) вклады в φ суммируются. ψ получает вклады всех шагов. Обе таблицы обновляются одновременно по старым значениям. Полные суммы и все параметры доступны на вкладке «Обновление».');
}

function gaeTable(it) {
  return table(['t','r','Vold(s)','Vold(s′)','b','c','δ','Aслед','γλcAслед','A','R'],it.steps.map((s,t)=>{
    const g=it.gae[t];return [t,s.reward,number(s.value),number(s.nextValue),g.bootstrap,g.continuation,number(g.delta),number(g.following),number(g.trace),number(g.advantage),number(g.return)];
  }));
}

function ppoPanel({iteration:it,t,epochIndex}) {
  const s=it.steps[t], config=it.config, gae=it.gae[t];
  if(config.command<0) return callout('Все пять целей выключены. В softmax остался один фиксированный класс ∅: p∅ = y∅ = 1, loss и все critic gradients равны 0. PPO/value update пропущен: команды и teacher label нет. Награда, A и R равны 0. Выбор действий равномерный.', 'orange')+gaeTable(it);
  const epoch=it.student.epochs[epochIndex], p=epoch.anchors[t], teachers=it.teachers[t];
  const totalGradient=epoch.thetaGradient[s.state][config.command];
  const teacherRows=ACTIONS.map((_,a)=>[actionName(a),number(criticForward(it.before,s.state,a,config.enabled).scores[config.command]),number(teachers.scores[a]),teachers.action===a?'a*':'']);
  return `<div class="trace-epoch-row"><span class="trace-kicker">PPO EPOCH</span>${it.student.epochs.map((_,i)=>`<button class="trace-button compact" data-epoch="${i}" aria-pressed="${i===epochIndex}">${i+1}</button>`).join('')}</div>`
    + note(`Показан epoch ${epochIndex+1}. πold, Vold, A, R и teacher заморожены для всех epochs. Текущие θ и V берутся из ${epochIndex===0?'начала итерации':'конца предыдущего epoch'}. Выбранный sampled action — <b>${actionName(s.action)}</b>.`)
    + math('СНАЧАЛА GAE · ИЗ НОВОГО REWARD И СТАРОГО VALUE',String.raw`\delta_t=r_t+\gamma b_t V_{\rm old}(s_{t+1},g)-V_{\rm old}(s_t,g)`)
    + math('ПОДСТАНОВКА ДЛЯ ЭТОГО ШАГА',String.raw`\delta_${t}=${s.reward}+${n(config.gamma)}\cdot${gae.bootstrap}\cdot${n(s.nextValue)}-(${n(s.value)})\approx${n(gae.delta)}`)
    + math('ОБРАТНЫЙ ПРОХОД ПО ТРАЕКТОРИИ',String.raw`A_${t}=\delta_${t}+\gamma\lambda c_${t} A_{${t+1}}\approx${n(gae.delta)}+${n(config.gamma)}\cdot${n(config.lambda)}\cdot${gae.continuation}\cdot${n(gae.following)}\approx${n(gae.advantage)},\quad R_${t}=A_${t}+V_{\rm old}(s_t)\approx${n(gae.return)}`)
    + detail('Все GAE-вычисления: читайте таблицу снизу вверх',gaeTable(it),true)
    + callout(`b = ${gae.bootstrap} разрешает bootstrap из final observation. c = ${gae.continuation} разрешает перенос следующего advantage. При смерти b=c=0; при time-limit b=1, c=0. Значения V и A не смешиваются со следующим reset.`)
    + detail('Откуда взялся teacher a*? Все action scores до и после CRL',table(['Действие','fold для команды','fnew для команды','Teacher'],teacherRows)+note('Берём argmax обновлённого f, а не вероятности p. При точном равенстве выбирается действие с меньшим индексом. У teacher нет градиента в PPO.'))
    + math('RATIO — ЭТО НЕ REWARD',String.raw`\rho_t=\frac{\pi_\theta(a_${s.action}\mid s_t,g)}{\pi_{\rm old}(a_${s.action}\mid s_t,g)}=\frac{${n(p.policy[s.action])}}{${n(s.policy[s.action])}}\approx${n(p.ratio)}`)
    + values([['ρA · без clip',format(p.raw)],['clip(ρ)A',format(p.clipped)],['Policy loss −min',format(p.policyLoss),p.clippedActive?'orange':'']])
    + policyComparison(s.policy,p.policy,['πold из rollout','π на выбранном epoch'])
    + math('PPO CLIPPED SURROGATE',String.raw`L_{\pi,t}=-\min(\rho_t A_t,\operatorname{clip}(\rho_t,${n(1-config.clip)},${n(1+config.clip)})A_t)\approx${n(p.policyLoss)}`)
    + callout(p.clippedActive?'Сработала обрезанная ветка min: её derivative по θ равна 0. Но teacher CE и entropy по-прежнему могут менять policy.':`Выбрана обычная ветка: её derivative = −Aρ(1[a=aₜ]−π(a)). При ${gae.advantage>=0?'положительном':'отрицательном'} advantage clip ограничивает ${gae.advantage>=0?'увеличение':'уменьшение'} вероятности sampled action.`)
    + math('ФОРМУЛА 9 · СИЛА TEACHER И ПОЛНЫЙ ЛОКАЛЬНЫЙ LOSS',String.raw`\beta_i=${n(config.beta)}\left(1-\frac{${it.index}}{${config.iterations-1}}\right)=${n(it.beta)},\quad L_t=L_{\pi,t}+${n(config.valueCoefficient)}L_{V,t}-${n(config.entropy)}H_t+${n(it.beta)}L_{BC,t}`)
    + table(['Слагаемое','Вычисление','Без коэффициента','С коэффициентом'],[
      ['Policy','−min(ρA, clip(ρ)A)',number(p.policyLoss),number(p.policyLoss)],
      ['Value',`½(${format(p.value)} − ${format(gae.return)})²`,number(p.valueLoss),number(config.valueCoefficient*p.valueLoss)],
      ['Entropy','−Σ π log π',number(p.entropy),number(-config.entropy*p.entropy)],
      ['Teacher',`−log π(a${teachers.action})`,number(p.bcLoss),number(it.beta*p.bcLoss)],
      ['Итого Lₜ','Сумма правого столбца','',number(p.loss)],
    ])
    + detail('Каждый член entropy и teacher CE',table(['a','π(a)','log π(a)','−π(a)log π(a)','Teacher?'],ACTIONS.map((_,a)=>[actionName(a),number(p.policy[a]),number(p.logPolicy[a]),number(-p.policy[a]*p.logPolicy[a]),a===teachers.action?'−log π входит в BC':'']))+note(`Сумма столбца −π log π = H = ${format(p.entropy)}. Teacher CE = −log π(a${teachers.action}) = ${format(p.bcLoss)}.`))
    + math('ВСЕ ПРОИЗВОДНЫЕ ПО ЧЕТЫРЁМ POLICY LOGITS',String.raw`\nabla_{\theta_a}L_\pi=-A\rho(\mathbf1[a=a_t]-\pi_a)\ \text{или }0;\quad\nabla L_{BC}=\beta(\pi_a-\mathbf1[a=a^*]);\quad\nabla(-c_HH)=c_H\pi_a(\log\pi_a+H)`)
    + table(['a','θ epoch','πold','π epoch','∂policy','β∂BC','−cH∂H','Сумма','Вклад / T','Batch grad','θ после'],ACTIONS.map((_,a)=>[
      actionName(a),number(p.logits[a]),number(s.policy[a]),number(p.policy[a]),number(p.policyGradient[a]),number(p.bcGradient[a]),number(p.entropyGradient[a]),
      number(p.gradient[a]),number(p.gradient[a]/it.steps.length),number(totalGradient[a]),number(epoch.after.theta[s.state][config.command][a]),
    ]))
    + math('UPDATE POLICY И VALUE',String.raw`\theta'_{s,g,a}=\theta_{s,g,a}-${n(config.actorLR)}\nabla L_{s,g,a},\qquad V'_{s,g}=V_{s,g}-${n(config.valueLR)}\cdot\frac{c_V}{T}\sum_{t:s_t=s}(V_{s,g}-R_t)`)
    + table(['Value этого (s,g)','V до','R для этого t','Локальный cV(V−R)','Вклад / T','Batch grad','V после'],[[point(s.state),number(p.value),number(gae.return),number(p.valueGradient),number(p.valueGradient/it.steps.length),number(epoch.valueGradient[s.state][config.command]),number(epoch.after.value[s.state][config.command])]])
    + detail('Все слагаемые batch-gradient для этой строки policy и value',table(['t с тем же s','Локальный ∂θ₀','∂θ₁','∂θ₂','∂θ₃','Локальный ∂V'],it.steps.flatMap((other,i)=>other.state===s.state?[[i,...epoch.anchors[i].gradient.map(number),number(epoch.anchors[i].valueGradient)]]:[]))+note(`Сложите каждый столбец и разделите на T = ${it.steps.length}. Остальные состояния дают этой строке нулевой вклад. Эта сумма и показана выше как Batch grad.`))
    + note('Batch grad суммирует все посещения того же (s,g), поэтому он может отличаться от вклада выбранного шага. Все θ и V обновляются одновременно. Gradients teacher/advantage/return/πold не вычисляются: это зафиксированные данные.');
}

function parameterInspector(it, inspectState, inspectGoal) {
  const config=it.config, s=inspectState, g=inspectGoal;
  return `<div class="trace-inspector"><label>Состояние для проверки<select data-inspect-state class="trace-inline-select">${Array.from({length:12},(_,i)=>i===WALL?'':`<option value="${i}" ${i===s?'selected':''}>${point(i)} · state ${i}</option>`).join('')}</select></label><label>Цель для policy / value<select data-inspect-goal class="trace-inline-select">${GOALS.map((goal,i)=>`<option value="${i}" ${g===i?'selected':''}>${goal.id} · ${goal.name}</option>`).join('')}</select></label></div>`
    + table(['a','φold₁','φold₂','Batch ∂φ₁','Batch ∂φ₂','φnew₁','φnew₂'],ACTIONS.map((_,a)=>[actionName(a),...it.before.phi[s][a].map(number),...it.critic.phiGradient[s][a].map(number),...it.after.phi[s][a].map(number)]))
    + table(['a','θold','θnew','πold','πnew'],ACTIONS.map((_,a)=>[actionName(a),number(it.before.theta[s][g][a]),number(it.after.theta[s][g][a]),number(softmax(it.before.theta[s][g])[a]),number(softmax(it.after.theta[s][g])[a])]))
    + policyComparison(softmax(it.before.theta[s][g]),softmax(it.after.theta[s][g]),['Начало итерации','Конец итерации'])
    + values([['Vold(s,g)',format(it.before.value[s][g])],['Vnew(s,g)',format(it.after.value[s][g])],['Изменение V',format(it.after.value[s][g]-it.before.value[s][g])]])
    + note(`Фактор φ не зависит от команды; ψ зависит только от цели. Таблицы policy и V имеют отдельные строки для каждой команды. Если строка не встречалась в rollout, её градиент равен 0. State стены не посещается. ${config.enabled[g]?'':'Выбранная цель выключена: её ψ и policy/value не обучаются.'}`);
}

function updatePanel({iteration:it, inspectState, inspectGoal}) {
  const T=it.steps.length;
  const allPhi=[];
  for(let s=0;s<12;s++) for(let a=0;a<4;a++) allPhi.push([`${point(s)} / a${a}`,...it.before.phi[s][a].map(number),...it.critic.phiGradient[s][a].map(number),...it.after.phi[s][a].map(number)]);
  const allPolicy=[];
  for(let s=0;s<12;s++) for(let g=0;g<5;g++) allPolicy.push([`${point(s)} / ${GOALS[g].id}`,
    it.before.theta[s][g].map(x => format(x)).join('; '),it.after.theta[s][g].map(x => format(x)).join('; '),
    softmax(it.before.theta[s][g]).map(x => format(x)).join('; '),softmax(it.after.theta[s][g]).map(x => format(x)).join('; '),number(it.before.value[s][g]),number(it.after.value[s][g])]);
  return note(`Итерация ${it.index+1}: сначала собраны ${T} переходов при неизменной πold, затем выполнен один общий critic update и ${it.student.epochs.length} PPO epochs. Ни один градиент не применялся во время сбора. Следующая итерация получит итоговые параметры без переинициализации.`)
    + values([['CRL loss до',format(it.critic.loss)],['CRL loss после',format(it.critic.lossAfter)],['Изменение',format(it.critic.lossAfter-it.critic.loss),'orange']])
    + math('СРЕДНИЙ LOSS КРИТИКА · ВСЕ СЛАГАЕМЫЕ',String.raw`L_C=\frac{${it.critic.anchors.map(a=>n(a.loss)).join('+')}}{${T}}\approx${n(it.critic.loss)}`)
    + table(['t','Состояние / действие','LС до','Локальный ∂φ₁','Локальный ∂φ₂'],it.steps.map((s,t)=>[t,`${point(s.state)} / a${s.action}`,number(it.critic.anchors[t].loss),...it.critic.anchors[t].localPhi.map(number)]))
    + table(['Общий ψ','ψold₁','ψold₂','Средний ∂ψ₁','Средний ∂ψ₂','ψnew₁','ψnew₂'],GOALS.map((g,i)=>[g.id,...it.before.psi[i].map(number),...it.critic.psiGradient[i].map(number),...it.after.psi[i].map(number)]))
    + detail('Разложить каждый batch-градиент ψ на вклады всех шагов',table(['g / j','(∂L₀ + … + ∂Lₜ) / T','Batch gradient'],GOALS.flatMap((g,i)=>[0,1].map(j=>[`${g.id} / ${j+1}`,`(${it.critic.anchors.map(a=>format(a.localPsi[i][j])).join(' + ')}) / ${T}`,number(it.critic.psiGradient[i][j])]))))
    + (it.student.epochs.length?table(['PPO epoch','Средний loss','Policy часть','cV·value','−cH·entropy','β·teacher'],it.student.epochs.map(e=>[e.index+1,number(e.loss),number(sum(e.anchors.map(a=>a.policyLoss))/T),number(it.config.valueCoefficient*sum(e.anchors.map(a=>a.valueLoss))/T),number(-it.config.entropy*sum(e.anchors.map(a=>a.entropy))/T),number(it.beta*sum(e.anchors.map(a=>a.bcLoss))/T)])):callout('PPO пропущен: активных целей нет. Target ∅ = 1 и p∅ = 1, поэтому critic loss и все gradients нулевые.'))
    + callout(`Сравните одну и ту же клетку до и после обучения. На следующей итерации начальные значения совпадут с правым столбцом этой итерации. Выбор действий останется случайным по обновлённой policy; поэтому траектории могут отличаться.`, 'purple')
    + `<h3>Инспектор параметров: до → после</h3>`+parameterInspector(it,inspectState,inspectGoal)
    + detail('Все 48 пар (s,a): φ, batch-градиенты, результат SGD',table(['(s,a)','φold₁','φold₂','grad₁','grad₂','φnew₁','φnew₂'],allPhi))
    + detail('Все 60 пар (s,g): четыре θ и π, а также V до/после',table(['(s,g)','θold [↑,→,↓,←]','θnew','πold','πnew','Vold','Vnew'],allPolicy))
    + note('Полная история каждого PPO epoch с его tensors и градиентами экспортируется кнопкой «Все числа · JSON». В браузере каждый локальный и batch-gradient policy/value доступен на вкладке 5 при выборе соответствующих шага и epoch. Большой learning rate может повысить loss; интерфейс показывает фактический результат.');
}

export function calculations(context) {
  const stage=context.stage;
  const names=['Откуда берётся движение','Как клетка становится метками','Как будущее становится target','Как target изменяет критика','Как обновляются policy и value','Что переносится в следующую итерацию'];
  const panels=[actionPanel,semanticPanel,futurePanel,criticPanel,ppoPanel,updatePanel];
  const it=context.iteration;
  const t=Math.min(context.cursor,it.steps.length-1);
  const body=panels[stage]({...context,t});
  const finished=stage===5&&context.cursor===it.steps.length&&context.iterationIndex+1===it.config.iterations;
  return `<div class="trace-tabs" role="tablist" aria-label="Этапы вычисления">${STAGES.map((name,i)=>`<button class="trace-tab" id="formula-tab-${i}" role="tab" aria-selected="${i===stage}" aria-controls="formula-panel" tabindex="${i===stage?0:-1}" data-stage="${i}"><b>0${i+1}</b>${name}</button>`).join('')}</div>
    <div class="trace-box trace-panel" role="tabpanel" id="formula-panel" aria-labelledby="formula-tab-${stage}"><div class="trace-panel-head"><div><span class="trace-kicker">${stage===5?'ИТОГ ИТЕРАЦИИ':`ШАГ t = ${t} · ${point(it.steps[t].state)} / a${it.steps[t].action}`}</span><h2>${names[stage]}</h2></div><span class="small-badge">${stage+1} / 6</span></div>${body}
    <div class="trace-next-formula"><small>${stage===5?'Итоговые параметры сохранены в журнале.':'Сначала формула, затем её числовая подстановка.'}</small><button class="trace-button primary" data-command="next-formula" ${finished?'disabled':''}>${finished?'Все итерации разобраны':stage===5?'Дальше по шагам':'Следующий расчёт · '+STAGES[stage+1]}</button></div></div>`;
}

export function iterationHistory(history, currentIndex) {
  return `<div class="trace-box trace-history"><h2>Итерации связаны параметрами</h2>${note('Здесь появляются только уже открытые итерации. Loss сравнивается до и после update на одном и том же rollout; между разными rollout это разные задачи классификации, а не кривая качества агента.')}
    ${table(['Итерация','Шагов','Награда','Конец','CRL до → после','β','πold(↑ | старт, команда)'],history.map((it,i)=>[i+1,it.steps.length,it.reward,it.fatal?'Смерть':'Time-limit',`${format(it.critic.loss)} → ${format(it.critic.lossAfter)}`,number(it.beta),number(it.steps[0].policy[0])]),history.map((_,i)=>i===currentIndex?'chosen':''))}</div>`;
}
