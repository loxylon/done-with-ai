import katex from '../vendor/katex.mjs';
import { GOALS, ACTIONS, labels, trajectory, targets, softmax, classProbabilities,
  crossEntropy, criticGradient, rng, drawOffset, conditionalVariance, ppo, clamp, sum, argmax } from './math.js';

const $ = (s, el = document) => el.querySelector(s);
const esc = x => String(x).replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const sub = n => String(n).replace(/\d/g, d => '₀₁₂₃₄₅₆₇₈₉'[d]);
const fmt = (v, digits = 3) => Number.isFinite(v) ? v.toLocaleString('ru-RU', {maximumFractionDigits: digits, minimumFractionDigits: digits}) : '∞';
const pc = (v, digits = 1) => `${fmt(v * 100, digits)}%`;
const mathNumber = value => {
  if(value > 0 && value < .0001){const [mantissa, exponent]=value.toExponential(2).split('e');return String.raw`${mantissa}\times 10^{${Number(exponent)}}`;}
  return fmt(value,4);
};
const tex = (source, cls = '') => `<div class="math ${cls}" data-tex="${esc(source)}"></div>`;
const equation = (label, source) => `<div class="math-box"><div class="math-label">${label}</div>${tex(source)}</div>`;
const callout = (text, orange = false) => `<div class="callout${orange ? ' orange' : ''}">${text}</div>`;
const card = (title, body, tag = '', desc = '') => `<section class="card"><div class="card-header"><div><h2 class="card-title">${title}</h2>${desc ? `<p class="card-desc">${desc}</p>` : ''}</div>${tag ? `<span class="small-badge">${tag}</span>` : ''}</div>${body}</section>`;
const stats = values => `<div class="stats-grid">${values.map(([name, value, unit = '']) => `<div class="stat"><div class="stat-name">${name}</div><div class="stat-value">${value}<small>${unit}</small></div></div>`).join('')}</div>`;
const details = (title, body) => `<details class="details"><summary>${title}</summary>${body}</details>`;
const defaults = () => ({ gamma: .9, term: .9, live: 2, cap: 8, anchor: 0, end: 'death', reset: false,
  duplicate: false, frames: [[0,1],[2],[1,2],[0],[0,2],[1]], semantic: [0,1], sampleGoal: 0,
  criticGoal: 0, criticAction: 2, criticTarget: 'fatal', criticLR: .5,
  logits: [[2,0,0],[3,5,5],[.8,.2,-.3]], phi: [1,2], psi: [2,-1],
  theta: 1.2, old: .35, advantage: 1, clip: .2, beta: .4, progress: 0, teacher: 0,
  teacherLinked: false, actorLR: .1,
});
const limits = { gamma:[.5,.995], term:[0,.995], live:[1,6], cap:[1,8], anchor:[0,8],
  sampleGoal:[0,3], criticGoal:[0,2], criticAction:[0,2], criticLR:[.05,2], theta:[-20,20],
  old:[.05,.95], advantage:[-3,3], clip:[.05,.5], beta:[0,2], progress:[0,100], teacher:[0,2], actorLR:[.01,.5] };
function readState() {
  const state = defaults();
  try {
    const raw = new URLSearchParams(location.search).get('state');
    if (!raw || raw.length > 6000) return state;
    const saved = JSON.parse(raw);
    for (const [key, [lo, hi]] of Object.entries(limits)) {
      if (typeof saved[key] === 'number' && Number.isFinite(saved[key])) state[key] = clamp(saved[key], lo, hi);
    }
    for (const key of ['live','cap','anchor','sampleGoal','criticGoal','criticAction','teacher']) state[key] = Math.round(state[key]);
    for (const key of ['reset','duplicate','teacherLinked']) if (typeof saved[key] === 'boolean') state[key] = saved[key];
    if (['death','success','timeout','window'].includes(saved.end)) state.end = saved.end;
    if (['fatal','live','trajectory'].includes(saved.criticTarget)) state.criticTarget = saved.criticTarget;
    if (Array.isArray(saved.frames)) state.frames = Array.from({length:6}, (_, i) => Array.isArray(saved.frames[i]) ? saved.frames[i].filter(x => [0,1,2].includes(x)) : state.frames[i]);
    if (Array.isArray(saved.semantic)) state.semantic = saved.semantic.filter(x => [0,1,2].includes(x));
    for (const key of ['phi','psi']) if (Array.isArray(saved[key]) && saved[key].length === 2) state[key] = saved[key].map(x => Number.isFinite(x) ? clamp(x,-4,4) : 0);
    if (Array.isArray(saved.logits) && saved.logits.length === 3 && saved.logits.every(row => Array.isArray(row) && row.length === 3)) state.logits = saved.logits.map(row => row.map(x => Number.isFinite(x) ? clamp(x,-100,100) : 0));
    if (state.end === 'window') state.reset = false;
    if (!state.duplicate) state.sampleGoal = Math.min(state.sampleGoal,2);
  } catch { /* A malformed shared URL restores the documented example. */ }
  return state;
}
let state = readState();
let route = 'termination';
let sampleResults = [];
let sampleRandom = rng(2026);
let gradientHistory = [];
let actorHistory = [];
let selectedPreset = 'death';
const pages = {
  semantics: { number:'01', eyebrow:'СЕМАНТИЧЕСКИЕ ЦЕЛИ · ФОРМУЛА 4', title:'Одно состояние.<br><span>Несколько смыслов.</span>', description:'Цель — не конкретная картинка мира, а предикат. Отметьте, что выполняется в состоянии, и проследите, как одна единица массы делится между метками.' },
  future: { number:'02', eyebrow:'RAO–BLACKWELL · ФОРМУЛА 6', title:'Не угадывать будущее.<br><span>Усреднить его.</span>', description:'Один случайный offset даёт шумную метку. Если траектория уже известна, можно вычислить точное ожидание внутри окна. Меняйте γ и смотрите, какие состояния получают вес.' },
  termination: { number:'03', eyebrow:'ПОГЛОЩЕНИЕ · ФОРМУЛЫ 7–8', title:'Будущее обрывается.<br><span>Масса остаётся.</span>', description:'Обычный CRL перераспределяет всё по выжившим состояниям. Но смерть не означает, что цель стала вероятнее. Посмотрите, куда должна попасть недостающая масса.' },
  critic: { number:'04', eyebrow:'КАТЕГОРИАЛЬНЫЙ КРИТИК · ФОРМУЛЫ 3 И 7', title:'Скор — ещё<br><span>не вероятность.</span>', description:'Критик сопоставляет состояние, действие и цель. Меняйте logits, сравнивайте два правила выбора действия и сделайте настоящий шаг градиента для одного anchor.' },
  teacher: { number:'05', eyebrow:'DUAL-CRL-PPO · ФОРМУЛА 9', title:'Учитель подсказывает.<br><span>Политика решает.</span>', description:'PPO учится по собственной награде, а критик добавляет подсказку. Исследуйте, когда клиппинг останавливает update, а когда сильный учитель тянет политику в другую сторону.' },
};

function renderMath(root) {
  root.querySelectorAll('[data-tex]').forEach(el => {
    katex.render(el.dataset.tex, el, { displayMode: !el.classList.contains('inline'), throwOnError: false, strict: 'ignore', trust: false });
  });
}
function optionBlock(label, field, options, help = '') {
  return `<div class="field"><label class="field-head" for="${field}">${label}</label><select id="${field}" data-field="${field}">${options.map(([value,name]) => `<option value="${value}"${String(state[field])===String(value)?' selected':''}>${name}</option>`).join('')}</select>${help ? `<p class="field-help">${help}</p>`:''}</div>`;
}
function slider(label, field, min, max, step, help = '') {
  const value = state[field];
  return `<div class="field"><div class="field-head"><label for="${field}">${label}</label><output for="${field}" data-out="${field}">${displayValue(field,value)}</output></div><input type="range" id="${field}" data-field="${field}" min="${min}" max="${max}" step="${step}" value="${value}"><div class="range-endpoints"><span>${min}</span><span>${max}</span></div>${help?`<p class="field-help">${help}</p>`:''}</div>`;
}
function displayValue(field,value) {
  if (field === 'progress') return `${fmt(value,0)}%`;
  return ['live','cap','sampleGoal','criticAction','criticGoal'].includes(field) ? fmt(value,0) : fmt(value, ['gamma','term'].includes(field)?3:2);
}
const checkbox = (label, field, disabled = false) => `<label class="checkbox-label"><input type="checkbox" data-field="${field}" id="${field}" ${state[field]?'checked':''} ${disabled?'disabled':''}><span>${label}</span></label>`;
const group = (label, body) => `<div class="control-group"><div class="control-heading">${label}</div>${body}</div>`;
function controls() {
  let body = '';
  if (route === 'semantics') {
    body = group('ПРЕДИКАТЫ', `<p class="field-help">Нажимайте на цели в карточке. Они могут выполняться одновременно.</p>${checkbox('Добавить похожий предикат древесины', 'duplicate')}`)
      + group('ПОПРОБУЙТЕ', `<p class="field-help">Включите похожую цель при двух активных предикатах. Вероятности старых меток изменятся, хотя само состояние осталось прежним.</p>`);
  } else if (route === 'future' || route === 'termination') {
    body = group('ДИСКОНТИРОВАНИЕ', slider('Горизонт будущего γ','gamma',.5,.995,.005,'Большое γ придаёт больший вес далёким состояниям.')
      + (route==='termination'?slider('Перенос конца γterm','term',0,.995,.005,'Чем больше значение, тем дальше назад уходит terminal penalty.')+`<button class="link-button" data-action="match-gamma">Приравнять γterm к γ ↗</button>`:''));
    body += group('ТРАЕКТОРИЯ', (route==='termination'?optionBlock('Тип конца','end', [['death','Смерть · настоящий terminal'],['success','Успех · terminal с меткой'],['timeout','Лимит времени · truncation'],['window','Просто конец rollout']]):'')
      + slider('Живых следующих состояний','live',1,6,1)
      + slider('Максимальный offset Kmax','cap',1,8,1,'Короткое окно не делает будущее за ним поглощающим.'));
    if (route==='termination') body += group('КРАЕВОЙ СЛУЧАЙ', checkbox('Показать states после reset','reset',state.end==='window')+`<p class="field-help">Новый эпизод исключается из будущего старого.</p>`);
    if (route==='future') body += group('НАБЛЮДАТЬ ЦЕЛЬ', optionBlock('Столбик статистики','sampleGoal', GOALS.slice(0,state.duplicate?4:3).map((g,i)=>[i,g.name])));
  } else if (route === 'critic') {
    body = group('ПАРА (s, a)', optionBlock('Редактируем действие','criticAction',ACTIONS.map((a,i)=>[i,a]))+optionBlock('Командуемая цель','criticGoal',GOALS.slice(0,3).map((g,i)=>[i,g.name])));
    body += group('ОБУЧАЕМЫЕ LOGITS', GOALS.slice(0,3).map((goal,g) => {
      const value = state.logits[state.criticAction][g];
      return `<div class="field"><div class="field-head"><label for="logit-${g}">${goal.name}</label><output data-logit-out="${g}">${fmt(value,2)}</output></div><input id="logit-${g}" type="range" data-logit="${g}" min="${Math.min(-6,Math.floor(value))}" max="${Math.max(6,Math.ceil(value))}" step=".05" value="${value}"></div>`;
    }).join('')+`<p class="field-help">Logit ∅ фиксирован: 0. Он не участвует в gradient update.</p>`);
    body += group('ОДИН ШАГ ОБУЧЕНИЯ', optionBlock('Target для выбранного action','criticTarget', [['fatal','Смерть: y∅ = 1'],['live','Цель достигнута: yg = 1'],['trajectory','Target из раздела 03']])+slider('Learning rate η','criticLR',.05,2,.05));
    body += group('ОТКУДА БЕРЁТСЯ f', `<details><summary class="field-help">Разложить на φ · ψ</summary><p class="field-help">Два признака — одна гипотеза score. Это не эмбеддинги обученной сети.</p>${['phi','psi'].map(key=>`<label class="field-help">${key==='phi'?'φ(s,a)':'ψ(g)'}</label><div class="number-pair">${state[key].map((v,i)=>`<input aria-label="${key} ${i+1}" type="number" min="-4" max="4" step=".25" value="${v}" data-vector="${key}" data-index="${i}">`).join('')}</div>`).join('')}<button class="link-button" data-action="apply-dot">Подставить φ · ψ в выбранный logit ↗</button></details>`);
  } else {
    body = group('PPO UPDATE', slider('Текущий logit θ для a₀','theta',Math.min(-6,Math.floor(state.theta)),Math.max(6,Math.ceil(state.theta)),.05)
      + slider('Старая вероятность πold(a₀)','old',.05,.95,.01)
      + slider('Advantage A','advantage',-3,3,.1)
      + slider('Ширина clip ε','clip',.05,.5,.01));
    body += group('ПОДСКАЗКА TEACHER', slider('Начальная сила β₀','beta',0,2,.05)
      + slider('Прогресс обучения','progress',0,100,1)
      + checkbox('Взять teacher action из критика','teacherLinked')
      + (!state.teacherLinked?optionBlock('Рекомендация учителя','teacher',ACTIONS.map((a,i)=>[i,a])):`<p class="field-help">Рекомендация — argmax f для цели «${GOALS[state.criticGoal].name}» из раздела 04.</p>`));
    body += group('ОПТИМИЗАЦИЯ', slider('Learning rate η','actorLR',.01,.5,.01));
  }
  return `<div class="controls-head"><h2>Параметры ситуации</h2><span class="controls-actions"><button class="icon-button" data-action="reset" aria-label="Сбросить пример" title="Исходный пример">↺</button><button class="icon-button drawer-close" data-action="close-controls" aria-label="Закрыть параметры">×</button></span></div>${body}<div class="controls-note"><strong>Здесь всё пересчитывается.</strong><br>Числа в диаграммах получены из формул, а не из заранее записанного обучения.</div>`;
}

function massbar(values, valid = true) {
  if (!valid) return `<div class="massbar empty">Нет положительной метки</div>`;
  const G = values.length-1;
  return `<div class="massbar" aria-label="${esc(values.map((v,i)=>`${i===G?'поглощение':GOALS[i].name}: ${pc(v)}`).join('; '))}">${values.map((value,i)=>`<div class="mass-segment" style="width:${value*100}%;background:${i===G?'var(--abs)':GOALS[i].color}" title="${i===G?'∅':GOALS[i].name}: ${pc(value,2)}">${value>.16?pc(value,0):''}</div>`).join('')}</div>`;
}
function legend(values) {
  return `<div class="mass-legend">${values.map((v,i)=>`<div class="legend-row ${v<1e-8?'dim':''}"><span class="swatch" style="background:${i===values.length-1?'var(--abs)':GOALS[i].color}"></span><span>${i===values.length-1?'Поглощение ∅':GOALS[i].name}</span><span class="legend-value">${pc(v)}</span></div>`).join('')}</div>`;
}
function model(method='corrected', override={}) {
  const transitions = trajectory(state);
  state.anchor = clamp(state.anchor,0,transitions.length-1);
  return targets(transitions,state.anchor,{gamma:state.gamma,term:state.term,cap:state.cap,method,duplicate:state.duplicate,...override});
}
function timeline(windowOnly=false) {
  const transitions = windowOnly?state.frames.slice(0,state.live).map((goals,i)=>({goals,name:`s${i+1}`})):trajectory(state);
  const m = windowOnly?targets(transitions,0,{gamma:state.gamma,term:state.term,cap:state.cap,duplicate:state.duplicate}):model();
  const usable = new Set(m.futures.map(f=>f.index));
  return `<div class="timeline">${transitions.map((s,i)=>`<div class="state-column ${i===state.anchor&&!windowOnly?'selected':''} ${!usable.has(i)&&i>=(windowOnly?0:state.anchor)&&!s.fatal?'foreign':''}">${!windowOnly?`<button class="anchor-button ${i===state.anchor?'active':''}" data-anchor="${i}" aria-pressed="${i===state.anchor}">a${sub(i)} · ${i===state.anchor?'считаем отсюда':'выбрать anchor'}</button>`:`<div class="anchor-button">offset k = ${i+1}</div>`}<div class="state-box ${s.fatal?'fatal':''}"><span class="state-symbol">${s.fatal?'∅':esc(s.name.replace(/\d/g,d=>sub(d)))}</span><span class="state-name">${s.fatal?'ПОГЛОЩЕНИЕ':s.reset?'ДРУГОЙ ЭПИЗОД':'СЛЕДУЮЩИЙ STATE'}</span>${s.fatal?'<span class="terminal-label">фатальное действие</span>':`<div class="goal-chips">${GOALS.slice(0,3).map((g,j)=>`<button class="goal-chip ${s.goals.includes(j)?'enabled':''}" style="--goal-color:${g.color}" aria-label="${g.name} в state ${i+1}" aria-pressed="${s.goals.includes(j)}" data-frame="${i}" data-goal="${j}" ${s.reset?'disabled':''}>${g.short}</button>`).join('')}</div>`}</div>${s.terminated&&!s.fatal?'<div class="terminal-label">успех · terminal</div>':s.truncated?'<div class="terminal-label">time-limit</div>':''}</div>`).join('')}</div><div class="timeline-footer"><span>Нажимайте на Д / К / И, чтобы менять выполненные цели.</span><span>${windowOnly?'Считаем для (s₀, a₀)':'Выберите aₜ, чтобы сдвинуть начало отсчёта.'}</span></div>`;
}
function presets() {
  return `<div class="preset-row"><span class="preset-label">СИТУАЦИЯ</span>${[['death','Смерть'],['success','Успех'],['timeout','Time-limit'],['empty','Нет целей'],['reset','Через reset'],['cap','Малое окно']].map(([id,name])=>`<button class="chip-button ${selectedPreset===id?'active':''}" data-preset="${id}">${name}</button>`).join('')}</div>`;
}
function methodPanel(name, method, number, description) {
  const m=model(method); const values=[...m.y,m.absorbing];
  return `<div class="method ${method==='corrected'?'featured':''}"><div class="method-head">${name}<span class="method-number">${number}</span></div><div class="method-sub">${method==='corrected'?'Сохраняем потерянную массу':method==='abs'?'Исправляем fatal anchor':'Условно на живое будущее'}</div>${massbar(values,m.valid)}${legend(values)}<div class="method-caption">${description}</div></div>`;
}
function calculationTable(m) {
  return `<div class="table-wrap"><table class="calculation-table"><thead><tr><th>Будущий state</th><th>k</th><th>(1−γ)γᵏ⁻¹</th><th>Вес wₖ</th><th>|G⁺|</th></tr></thead><tbody>${m.futures.map(f=>`<tr><td>s${sub(f.index+1)}</td><td>${f.k}</td><td>${fmt(f.raw,4)}</td><td>${fmt(f.weight,4)}</td><td>${sum(f.labels)>0?f.labels.filter(x=>x>0).length:0}</td></tr>`).join('')||'<tr><td colspan="5">Живых будущих состояний нет. Вся масса идёт в ∅.</td></tr>'}</tbody></table></div>`;
}
function termination() {
  const m=model();
  const tail=Number.isFinite(m.delta)?state.term**m.delta:0;
  const sourceMass=sum(m.futures.map(f=>f.weight*sum(f.labels)));
  const missingSemantic=m.retained*(1-sourceMass);
  let note = state.end==='timeout'?'Лимит времени — не смерть. <strong>Terminal penalty не добавляется.</strong> Финальная живая observation остаётся в targets.':state.end==='window'?'Мы видим только конец окна. <strong>Неизвестный хвост не объявляем смертью.</strong> Вне окна будущее не наблюдалось.':m.delta===0?'<strong>Именно это действие смертельно.</strong> В обычном CRL оно не имеет положительной метки и пропускается. Класс ∅ даёт ему явный градиент.':state.end==='success'?'<strong>Успех тоже завершает этот эпизод.</strong> Последняя observation сохраняет метку цели: для последнего action Δ = 1, а не 0. Это адаптация к episodic MiniGrid.':`<strong>${pc(tail)} массы находится за наблюдаемым концом.</strong> Перенормировка обычного CRL раздаёт её целям, которых агент успел достичь до смерти.`;
  if(state.reset && !Number.isFinite(m.delta) && trajectory(state).slice(0,state.anchor).some(s=>s.terminated||s.truncated))
    note='<strong>Anchor уже в новом эпизоде.</strong> Прошлый terminal и его цели не входят в это будущее. Конца текущего эпизода мы пока не наблюдали.';
  if (missingSemantic>.0001) note += ` Ещё ${pc(missingSemantic)} остатка связано с states без выполненных целей.`;
  return presets()+card('Постройте будущее одного действия',timeline(),`ANCHOR a${sub(state.anchor)}`)
    +stats([['Доступных будущих Kₜ',m.futures.length,'states'],['До поглощения Δₜ',Number.isFinite(m.delta)?m.delta:'∞',''],['Переносимая масса γterm^Δ',pc(tail,0),'']])
    +card('Одна траектория. Три способа учиться.',`<div class="methods-grid">${methodPanel('Обычный CRL','classic','6','Fatal anchor не участвует в loss. Нулевые метки пропускаются.')}${methodPanel('+ класс ∅','abs','7','Смерть последнего действия видна, но предыдущие targets всё ещё перенормированы.')}${methodPanel('+ перенос конца','corrected','8','Одна и та же доля снимается со всех целей и переходит в ∅.')}</div>${callout(note)}`,'СРАВНЕНИЕ TARGETS')
    +card('Вычисление, которое вы только что увидели',equation('ФОРМУЛА 8 · TERMINATION-DISCOUNTED TARGET',String.raw`y_t(g)=\underbrace{(1-\gamma_{\rm term}^{\Delta_t})}_{${fmt(m.retained,3)}}\;\underbrace{\sum_{k=1}^{K_t}w_k\ell_g(s_{t+k})}_{\text{среднее по будущему}}`)
      +equation('СОХРАНЕНИЕ МАССЫ',String.raw`y_t(\varnothing)=1-\sum_g y_t(g)=${fmt(m.absorbing,4)}`)
      +`<p class="explanation">Класс ∅ получает недостающую массу. Это не отдельный reward за смерть и не вероятность умереть на следующем шаге.</p>`
      +details('Раскрыть веса каждого будущего state',calculationTable(m))
      +details('Когда это точная оценка occupancy?',`<p>Когда γterm = γ и Kₜ = Δₜ, множитель (1−γ^Δ) сокращает нормировку усечённой геометрии. При ограниченном Kmax &lt; Δ или ненаблюдаемом конце окна остаётся finite-window bias. При γterm ≠ γ мы намеренно меняем дальность terminal penalty. Отсутствие семантических меток тоже создаёт residual mass.</p>`),'§ 3.2');
}
function semantics() {
  const y=labels(state.semantic,state.duplicate); const active=y.filter(v=>v>0).length;
  const values=[...y,1-sum(y)];
  const toggles=GOALS.slice(0,3).map((g,i)=>`<button class="semantic-toggle" style="--goal-color:${g.color}" aria-pressed="${state.semantic.includes(i)}" data-sem="${i}"><b>${state.semantic.includes(i)?'✓':'+'}</b>${g.name}</button>`).join('');
  const stage=`<div class="semantic-stage"><div class="state-disc"><strong>s</strong><small>1 единица массы</small></div><span class="semantic-arrow">→</span><div class="semantic-output">${y.map((v,i)=>v>0?`<div class="semantic-goal" style="--goal-color:${GOALS[i].color}"><span>${GOALS[i].short}</span>${GOALS[i].name}<small>1 / ${active}</small></div>`:'').join('')||'<div class="empty-semantic">Ни один предикат не выполнен.<br>У целей масса 0.</div>'}</div></div>`;
  return card('Что выполняется в этом состоянии?',`${toggles?`<div class="semantic-select">${toggles}</div>`:''}${stage}${massbar(values)}${legend(values)}`,'G⁺(s)')
    +card('Равномерное деление, а не независимые вероятности',equation('ФОРМУЛА 4 · СЕМАНТИЧЕСКАЯ МЕТКА',String.raw`\ell_g(s)=\frac{\mathbf{1}[g\in G^+(s)]}{|G^+(s)|}`)
      +stats([['Выполнено предикатов',active,''],['Масса каждой метки',active?fmt(1/active,3):'0',''],['Сумма у целей',fmt(sum(y),1),'']])
      +callout(active?`<strong>Состояние не изменилось, но метки зависят от набора целей.</strong> ${state.duplicate&&state.semantic.includes(0)?'Добавленный похожий предикат получает свою долю. Масса исходных целей стала меньше.':'Если добавить ещё один co-satisfied предикат, каждая активная цель получит меньшую долю.'}`:'<strong>Пустой набор G⁺ — важный крайний случай.</strong> Выражение 0/0 не вычисляем: ℓ(s) = 0. В модели с поглощением недостающая масса уходит в ∅, даже если агент жив.',!active)
      +details('Почему не поставить каждой цели вероятность 1?',`<p>Это была бы multi-label задача с независимым BCE. В статье используется categorical softmax: одна единица массы разделена между всеми выполненными целями. Именно эту нормированную semantic occupancy оценивает critic. Она не равна вероятности «достигнуть цель хотя бы раз» и не является исходным Q для бинарного reward.</p>`),'§ 3.1')
    +`<div class="lesson-grid">${card('Предикат — смысл, а не точный state',`<p class="lesson-text">«Есть древесина» может выполняться в тысячах разных состояний. «Есть кирка» и «рядом с деревом» могут быть истинны одновременно.</p>`)}${card('Что дальше?',`<p class="lesson-text">Следующий раздел усредняет такие метки во времени — с геометрическими весами. <a class="link-button" href="#future">Перейти к будущему →</a></p>`)}</div>`;
}
function svgPlot(xs, ys, marker=null, secondary=null, axisLabel='') {
  const W=560,H=205,L=36,R=16,T=16,B=32;
  const xmin=Math.min(...xs),xmax=Math.max(...xs);
  const all=secondary?[...ys,...secondary]:ys;
  let ymin=Math.min(0,...all),ymax=Math.max(.1,...all);
  if(ymin===ymax) ymax=ymin+1;
  const pad=(ymax-ymin)*.08;ymin-=pad;ymax+=pad;
  const px=x=>L+(x-xmin)/(xmax-xmin||1)*(W-L-R);
  const py=y=>T+(ymax-y)/(ymax-ymin)*(H-T-B);
  const path=v=>v.map((y,i)=>`${i?'L':'M'}${px(xs[i]).toFixed(2)},${py(y).toFixed(2)}`).join(' ');
  return `<svg class="plot" viewBox="0 0 ${W} ${H}" role="img" aria-label="${esc(axisLabel||'График вычисляемой функции')}">${[0,.5,1].map(f=>{const y=ymin+(ymax-ymin)*f;return `<line class="grid" x1="${L}" x2="${W-R}" y1="${py(y)}" y2="${py(y)}"/><text x="${L-7}" y="${py(y)+3}" text-anchor="end">${fmt(y,1)}</text>`;}).join('')}<line class="axis" x1="${L}" x2="${W-R}" y1="${H-B}" y2="${H-B}"/>${secondary?`<path d="${path(secondary)}" fill="none" stroke="#9aa8bb" stroke-width="1.5" stroke-dasharray="4 4"/>`:''}<path class="curve" d="${path(ys)}"/>${marker?`<line x1="${px(marker[0])}" x2="${px(marker[0])}" y1="${T}" y2="${H-B}" stroke="#d3654640" stroke-dasharray="3 4"/><circle class="marker" cx="${px(marker[0])}" cy="${py(marker[1])}" r="4.5"/>`:''}<text x="${L}" y="${H-11}">${fmt(xmin,1)}</text><text x="${W-R}" y="${H-11}" text-anchor="end">${fmt(xmax,1)}</text><text x="${W/2}" y="${H-3}" text-anchor="middle">${esc(axisLabel)}</text></svg>`;
}
function futureModel() {
  const transitions=state.frames.slice(0,state.live).map(goals=>({goals}));
  return targets(transitions,0,{gamma:state.gamma,term:state.term,cap:state.cap,duplicate:state.duplicate,method:'corrected'});
}
function future() {
  const m=futureModel(); const g=Math.min(state.sampleGoal,m.y.length-1);
  const average=sampleResults.length?sum(sampleResults.map(s=>s.labels[g]))/sampleResults.length:0;
  const std=Math.sqrt(conditionalVariance(m.futures,g));
  const xs=m.futures.map(f=>f.k), ys=m.futures.map(f=>f.weight);
  const last=sampleResults.at(-1);
  return card('Уже наблюдённая траектория',timeline(true),'ОДНО ОКНО')
    +card('Сколько весит каждое будущее?',svgPlot(xs,ys,last?[last.k,last.weight]:null,null,'Смещение k')+equation('НОРМИРОВАННЫЙ ГЕОМЕТРИЧЕСКИЙ ВЕС',String.raw`w_k=\frac{(1-\gamma)\gamma^{k-1}}{1-\gamma^{K_t}}`)
      +details('Развернуть конкретные коэффициенты',calculationTable(m)),'k ∼ Geom(γ)')
    +card('Случайная метка против точного ожидания',`<div class="action-row"><button class="primary-button" data-sample="1">＋ Одно будущее</button><button class="secondary-button" data-sample="100">＋ 100</button><button class="secondary-button" data-sample="1000">＋ 1000</button><button class="text-button" data-action="clear-samples">Сбросить</button><span class="action-note">PRNG seed · 2026</span></div><div class="sample-counter"><strong>${sampleResults.length.toLocaleString('ru-RU')}</strong> offsets выбрано${last?` · последний k = ${last.k}`:''}</div><div class="sample-dots" aria-label="Последние выбранные offsets">${sampleResults.slice(-70).map(s=>`<span class="sample-dot" style="--goal-color:${s.labels[g]>0?GOALS[g].color:'#c7cebc'}" title="k=${s.k}, метка=${fmt(s.labels[g])}"></span>`).join('')}</div><div class="comparison-table">${[['Среднее MC',average],['Точное RB',m.y[g]]].map(([label,v])=>`<div class="comparison-row"><span>${label}</span><div class="comparison-track"><span class="comparison-fill" style="--goal-color:${GOALS[g].color};width:${v*100}%"></span><span class="comparison-target" style="left:${m.y[g]*100}%"></span></div><span>${fmt(v)}</span></div>`).join('')}</div><div class="numeric-cards"><div class="numeric-card"><span>STD одного sampled target</span><strong>${fmt(std,4)}</strong></div><div class="numeric-card"><span>STD RB при той же траектории</span><strong>0</strong></div></div>${callout('<strong>Меньше шума — та же условная средняя.</strong> RB интегрирует выбор offset. Шум самой траектории, среды, политики и SGD остаётся.')}`,'ФОРМУЛА 6')
    +card('Точное ожидание внутри окна ≠ бесконечное будущее',equation('RAO–BLACKWELL TARGET',String.raw`\bar y_t(g)=\sum_{k=1}^{K_t}w_k\ell_g(s_{t+k})`)+`<p class="explanation">Для цели «${GOALS[g].name}»: <strong>y = ${fmt(m.y[g],4)}</strong>. Внутри наблюдённого окна распределение известно, но масса геометрического хвоста за ним равна γᴷ = <strong>${pc(state.gamma**m.futures.length)}</strong>. Просто конец rollout не позволяет назвать этот хвост смертью.</p>${callout('В обоих вариантах semantic label уже распределена по всем выполненным предикатам. Здесь семплируется только время, а не сама цель.',true)}`,'КОНЕЧНОЕ ОКНО');
}
function criticTarget() {
  if(state.criticTarget==='fatal') return [0,0,0,1];
  if(state.criticTarget==='live') return [0,0,0,0].map((_,i)=>Number(i===state.criticGoal));
  const m=model('corrected',{duplicate:false});return [...m.y,m.absorbing];
}
function critic() {
  const probs=state.logits.map(classProbabilities),g=state.criticGoal,a=state.criticAction;
  const fWinner=argmax(state.logits.map(row=>row[g]));
  const pWinner=argmax(probs.map(row=>row[g]));
  const target=criticTarget(); const loss=crossEntropy(state.logits[a],target);
  const grad=criticGradient(state.logits[a],target);
  const phiDot=sum(state.phi.map((v,i)=>v*state.psi[i]));
  return card('Три действия. Два правила выбора.',`<div class="critic-actions">${ACTIONS.map((name,i)=>`<div class="critic-action"><div class="action-name">${name}<small>f(g) = ${fmt(state.logits[i][g],2)}</small></div><div>${massbar(probs[i])}${fWinner===i||pWinner===i?`<div class="winner">${fWinner===i?'↳ argmax f':''} ${pWinner===i?'↳ argmax p(g)':''}</div>`:''}</div><div class="critic-value">${pc(probs[i][g],1)}<small class="screen-reader"> вероятность цели</small></div></div>`).join('')}</div><div class="pills">${GOALS.slice(0,3).map(goal=>`<span class="pill" style="color:${goal.color}">● ${goal.name}</span>`).join('')}<span class="pill">● ∅</span></div>${callout(fWinner!==pWinner?`<strong>Решения различаются.</strong> Статья выбирает «${ACTIONS[fWinner]}» по logit. Нормированная вероятность цели выше у «${ACTIONS[pWinner]}». Маленький p∅ меняет отношение p(g)/p∅.`:'<strong>Сейчас оба правила согласны.</strong> Измените logits других целей: denominator softmax зависит и от них, хотя командуемая цель остаётся прежней.',fWinner!==pWinner)}`,'ВЫБРАННАЯ ЦЕЛЬ: '+GOALS[g].name)
    +card('Общий нулевой logit задаёт шкалу',equation('ФОРМУЛА 7',String.raw`p_g=\frac{e^{f_g}}{1+\sum_{g'}e^{f_{g'}}},\qquad p_{\varnothing}=\frac{1}{1+\sum_{g'}e^{f_{g'}}}`)+equation('ТОЧНОЕ ОТНОШЕНИЕ ДЛЯ ВЫБРАННОГО ACTION',String.raw`f_g=\log\frac{p_g}{p_\varnothing}\approx\log\frac{${mathNumber(probs[a][g])}}{${mathNumber(probs[a].at(-1))}}\approx ${fmt(state.logits[a][g],3)}`)+`<p class="explanation">p∅ — residual discounted mass, не одноступенчатая вероятность смерти. Логит тоже не вероятность. Совместно обученная нейросеть ограничивает допустимые scores через φ(s,a) · ψ(g).</p>`,'LOGIT ∅ ≡ 0')
    +card('Пусть это действие получило такую метку',`${massbar(target)}${legend(target)}${equation('CROSS-ENTROPY И ЕЁ ГРАДИЕНТ',String.raw`\mathcal L=-\sum_g y_g\log p_g,\qquad \frac{\partial\mathcal L}{\partial f_g}=p_g-y_g`)}<div class="gradient-row">${grad.map((v,i)=>`<div class="gradient-item ${v<0?'negative':''}">${GOALS[i].name}<strong>${v>0?'+':''}${fmt(v,4)}</strong>logit ${v>0?'будет снижаться':'будет расти'}</div>`).join('')}</div><div class="action-row"><button class="primary-button" data-action="critic-step">Один шаг градиента →</button><span class="action-note">f ← f − η(p − y)</span></div><div class="numeric-cards"><div class="numeric-card"><span>Текущая cross-entropy</span><strong>${fmt(loss,4)}</strong></div><div class="numeric-card"><span>После одного шага</span><strong>${fmt(crossEntropy(state.logits[a].map((x,i)=>x-state.criticLR*grad[i]),target),4)}</strong></div></div>${gradientHistory.length?`<p class="field-help">Уже сделано ${gradientHistory.length} updates в этом примере. Изменение logits/target начинает новую историю.</p>`:''}${callout('<strong>Это локальная модель update.</strong> Обновляем три независимых logits одного action. В реальной φ·ψ-сети параметры разделяются между состояниями и целями; напрямую независимые logits не оптимизируются.',true)}`,'ACTION: '+ACTIONS[a])
    +card('Откуда вообще берётся score?',`<div class="dot-product"><div class="vector">${state.phi.map(x=>`<span>${fmt(x,2)}</span>`).join('')}</div><span class="vector-operator">·</span><div class="vector">${state.psi.map(x=>`<span>${fmt(x,2)}</span>`).join('')}</div><span class="vector-operator">=</span><span class="stat-value">${fmt(phiDot,2)}</span></div><p class="explanation">Слева — φ(s,a), справа — ψ(g). В «Параметрах» раскройте блок φ · ψ, измените компоненты и подставьте результат в выбранный logit.</p>`,'φ(s,a) · ψ(g)');
}
function actorOptions(theta=state.theta) {
  const teacher=state.teacherLinked?argmax(state.logits.map(row=>row[state.criticGoal])):state.teacher;
  return {theta,old:state.old,advantage:state.advantage,clip:state.clip,beta:state.beta,progress:state.progress,teacher};
}
function teacher() {
  const options=actorOptions(), m=ppo(options);
  const xmin=Math.min(-6,Math.floor(state.theta-1)), xmax=Math.max(6,Math.ceil(state.theta+1));
  const xs=Array.from({length:121},(_,i)=>xmin+i*(xmax-xmin)/120), losses=xs.map(x=>ppo(actorOptions(x)).loss), ppoOnly=xs.map(x=>ppo(actorOptions(x)).policyLoss);
  const ratioMax=Math.max(2.6,Math.ceil(m.ratio+.2));
  const ratioXs=Array.from({length:101},(_,i)=>i*ratioMax/100);
  const surrogate=ratioXs.map(r=>-Math.min(r*state.advantage,clamp(r,1-state.clip,1+state.clip)*state.advantage));
  return card('Одна policy. Два источника обучения.',`<div class="teacher-policies"><div class="teacher-block"><small>CRL TEACHER</small><strong>${ACTIONS[options.teacher]}</strong><p class="field-help">${state.teacherLinked?'argmax f из раздела 04':'Заданная вами рекомендация'}</p></div><span class="teacher-arrow">→</span><div class="teacher-block"><small>PPO STUDENT</small>${m.policy.map((p,i)=>`<div class="policy-line"><span>a${sub(i)} · ${ACTIONS[i].split(' ')[0]}</span><b>${pc(p,1)}</b></div>`).join('')}</div></div>${equation('ФОРМУЛА 9 · ПОКАЗАНА POLICY-ЧАСТЬ',String.raw`\mathcal L=\mathcal L_{\rm PPO}+\underbrace{\beta_0(1-\mathrm{progress})}_{${fmt(m.effectiveBeta,3)}}\,[-\log\pi(a^*|s,g)]`)}<p class="explanation">В этом изолированном примере текущая policy = softmax(θ, 0, 0). Рассматриваем sampled action a₀. Value loss и entropy не показаны, чтобы отделить clipping от teacher CE.</p><div class="pills"><span class="pill ${m.effectiveBeta===0?'orange':''}">β сейчас: ${fmt(m.effectiveBeta,3)}</span><span class="pill">Sampled action: a₀</span><span class="pill">Teacher: a${sub(options.teacher)}</span></div>`,'PPO + BEHAVIORAL CLONING')
    +card('Клиппинг зависит от знака advantage',stats([['Ratio πnew(a₀) / πold(a₀)',fmt(m.ratio,3)],['Clipped ratio',fmt(m.bounded,3)],['Lπ = −min(…)',fmt(m.policyLoss,3)]])
      +svgPlot(ratioXs,surrogate,[m.ratio,m.policyLoss],null,'Ratio (по оси X)')
      +equation('CLIPPED SURROGATE',String.raw`\mathcal L_\pi=-\min\left(rA,\;\mathrm{clip}(r,1-\epsilon,1+\epsilon)A\right)`)
      +callout(m.clippedActive?'<strong>Clipped-ветка активна.</strong> По этой координате PPO gradient сейчас равен нулю. Подсказка учителя всё ещё может менять policy.':'<strong>PPO gradient не выключен.</strong> Ratio вне диапазона сам по себе не означает clip: minimum выбирает ветку с учётом знака advantage.')
      +details('Частая ошибка: «PPO всегда обрезает ratio»',`<p>При A &gt; 0 обрезается чрезмерное повышение вероятности полезного action. При A &lt; 0 — чрезмерное снижение вероятности неудачного. Движение в плохую сторону остаётся штрафуемым без такого ограничения. На границах clip функция недифференцируема; здесь показана выбранная кусочная ветка.</p>`),'МЕНЯЙТЕ A ↔ −A')
    +card('Что произойдёт с θ?',svgPlot(xs,losses,[state.theta,m.loss],ppoOnly,'Logit θ для a₀ · сплошная: total; пунктир: PPO')
      +`<div class="gradient-row"><div class="gradient-item ${m.policyGradient<0?'negative':''}">PPO gradient<strong>${fmt(m.policyGradient,4)}</strong></div><div class="gradient-item ${m.bcGradient<0?'negative':''}">Teacher gradient × β<strong>${fmt(m.effectiveBeta*m.bcGradient,4)}</strong></div><div class="gradient-item ${m.gradient<0?'negative':''}">Суммарный gradient<strong>${fmt(m.gradient,4)}</strong></div></div><div class="action-row"><button class="primary-button" data-action="actor-step">Один шаг policy →</button><span class="action-note">θ ← θ − η ∂L/∂θ</span></div><div class="numeric-cards"><div class="numeric-card"><span>Total policy loss сейчас</span><strong>${fmt(m.loss,4)}</strong></div><div class="numeric-card"><span>θ после update</span><strong>${fmt(state.theta-state.actorLR*m.gradient,4)}</strong></div></div>${actorHistory.length?`<p class="field-help">Выполнено updates: ${actorHistory.length}. Старые вероятности фиксированы, как внутри PPO update.</p>`:''}${callout(m.effectiveBeta===0?'<strong>Учитель отключён.</strong> При progress = 100% или β₀ = 0 остаётся только policy clipping. Поведение teacher больше не влияет на loss.':Math.sign(m.policyGradient)!==Math.sign(m.bcGradient)&&Math.abs(m.policyGradient)>.00001?'<strong>Сигналы могут спорить.</strong> Увеличивая β, можно заставить student следовать даже ошибочному teacher. Аннилинг β до нуля постепенно снимает это ограничение.':'<strong>Teacher CE направляет, но не заменяет PPO.</strong> Низкий BC loss означает согласие с учителем, а не качество его рекомендаций.',true)}`,'АНАЛИТИЧЕСКИЙ ГРАДИЕНТ');
}

function renderContent() {
  const active=document.activeElement;
  const focus=active?.id;
  const token=['sem','anchor','action','sample'].find(key=>active?.dataset?.[key]!==undefined);
  const selector=token?`button[data-${token}="${active.dataset[token]}"]`:null;
  $('#workspace').innerHTML=({semantics,future,termination,critic,teacher})[route]();
  renderMath($('#workspace'));
  if(focus) document.getElementById(focus)?.focus({preventScroll:true});
  else if(selector) $(selector,$("#workspace"))?.focus({preventScroll:true});
}
function renderControls() { $('#controls').innerHTML=controls(); }
function renderPage() {
  const p=pages[route];
  $('#eyebrow').textContent=p.eyebrow;$('#title').innerHTML=p.title;$('#description').textContent=p.description;
  document.title=`${p.number} · ${route==='termination'?'Поглощение':route==='future'?'Будущее':route==='critic'?'Критик':route==='teacher'?'PPO':'Семантика'} — CRL Lab`;
  document.querySelectorAll('[data-route]').forEach(a=>{if(a.dataset.route===route)a.setAttribute('aria-current','page');else a.removeAttribute('aria-current');});
  renderControls();renderContent();
}
function invalidateSamples() { sampleResults=[];sampleRandom=rng(2026); }
function setRoute() {
  const requested=location.hash.slice(1);
  route=pages[requested]?requested:'termination';
  document.body.classList.remove('controls-open');$('#mobile-controls').setAttribute('aria-expanded','false');
  renderPage();
  window.scrollTo({top:0,behavior:'instant'});
}
function preset(id) {
  state.frames=defaults().frames;state.live=2;state.anchor=0;state.cap=8;state.reset=false;state.end='death';state.gamma=.9;state.term=.9;
  if(['success','timeout'].includes(id))state.end=id;
  if(id==='empty'){state.end='window';state.frames=[[],[],...defaults().frames.slice(2)];}
  if(id==='reset')state.reset=true;
  if(id==='cap')state.cap=1;
  selectedPreset=id;invalidateSamples();renderControls();renderContent();
}
function toast(text) { $('#toast').textContent=text;$('#toast').classList.add('visible');setTimeout(()=>$('#toast').classList.remove('visible'),2600); }

$('#controls').addEventListener('input', event=> {
  const el=event.target;
  if(el.matches('select'))return;
  if(el.dataset.logit!==undefined){state.logits[state.criticAction][+el.dataset.logit]=Number(el.value);$(`[data-logit-out="${el.dataset.logit}"]`).textContent=fmt(Number(el.value),2);gradientHistory=[];renderContent();return;}
  if(el.dataset.vector){state[el.dataset.vector][+el.dataset.index]=clamp(Number(el.value)||0,-4,4);renderContent();return;}
  const key=el.dataset.field;if(!key)return;
  state[key]=el.type==='checkbox'?el.checked:Number(el.value);
  if(key==='duplicate'&&!state.duplicate)state.sampleGoal=Math.min(state.sampleGoal,2);
  const out=$(`[data-out="${key}"]`);if(out)out.textContent=displayValue(key,state[key]);
  if(['gamma','live','cap','duplicate'].includes(key))invalidateSamples();
  gradientHistory=[];actorHistory=[];
  if(key==='teacherLinked')renderControls();
  renderContent();
});
$('#controls').addEventListener('change',event=>{
  const el=event.target;if(!el.matches('select'))return;
  const key=el.dataset.field;
  state[key]=['end','criticTarget'].includes(key)?el.value:Number(el.value);
  if(state.end==='window')state.reset=false;
  state.anchor=clamp(state.anchor,0,trajectory(state).length-1);
  gradientHistory=[];actorHistory=[];invalidateSamples();
  renderControls();renderContent();document.getElementById(key)?.focus({preventScroll:true});
});
document.addEventListener('click',event=>{
  const el=event.target.closest('button');if(!el)return;
  if(el.dataset.openAbout!==undefined){$('#about').showModal();return;}
  if(el.dataset.preset){preset(el.dataset.preset);return;}
  if(el.dataset.anchor!==undefined){state.anchor=Number(el.dataset.anchor);renderContent();return;}
  if(el.dataset.sem!==undefined){const g=+el.dataset.sem;state.semantic=state.semantic.includes(g)?state.semantic.filter(x=>x!==g):[...state.semantic,g];renderContent();return;}
  if(el.dataset.goal!==undefined){const g=+el.dataset.goal,i=+el.dataset.frame;state.frames[i]=state.frames[i].includes(g)?state.frames[i].filter(x=>x!==g):[...state.frames[i],g];invalidateSamples();renderContent();return;}
  if(el.dataset.sample){const m=futureModel();for(let i=0;i<+el.dataset.sample;i++)sampleResults.push(m.futures[drawOffset(m.futures,sampleRandom)]);renderContent();return;}
  const action=el.dataset.action;
  if(action==='close-controls'){closeControls();return;}
  if(action==='reset'){state=defaults();invalidateSamples();gradientHistory=[];actorHistory=[];selectedPreset='death';renderPage();toast('Исходный пример восстановлен');}
  if(action==='match-gamma'){state.term=state.gamma;renderControls();renderContent();}
  if(action==='clear-samples'){invalidateSamples();renderContent();}
  if(action==='apply-dot'){state.logits[state.criticAction][state.criticGoal]=sum(state.phi.map((x,i)=>x*state.psi[i]));gradientHistory=[];renderControls();renderContent();toast('Скалярное произведение подставлено в logit');}
  if(action==='critic-step'){const a=state.criticAction,t=criticTarget();const grad=criticGradient(state.logits[a],t);gradientHistory.push(crossEntropy(state.logits[a],t));state.logits[a]=state.logits[a].map((x,i)=>x-state.criticLR*grad[i]);renderControls();renderContent();}
  if(action==='actor-step'){const m=ppo(actorOptions());actorHistory.push(m.loss);state.theta=state.theta-state.actorLR*m.gradient;renderControls();renderContent();}
});
$('#share').addEventListener('click',async()=>{
  const url=new URL(location.href);url.search='';url.searchParams.set('state',JSON.stringify(state));url.hash=route;
  try{await navigator.clipboard.writeText(url.href);history.replaceState(null,'',url);toast('Ссылка на эту ситуацию скопирована');}
  catch{history.replaceState(null,'',url);toast('Ситуация сохранена в адресной строке — скопируйте ссылку');}
});
$('#about-open').addEventListener('click',()=>$('#about').showModal());
$('.dialog-close').addEventListener('click',()=>$('#about').close());
$('#about').addEventListener('click',e=>{if(e.target===$('#about')){const r=e.target.getBoundingClientRect();if(e.clientX<r.left||e.clientX>r.right||e.clientY<r.top||e.clientY>r.bottom)e.target.close();}});
function closeControls(){document.body.classList.remove('controls-open');$('#mobile-controls').setAttribute('aria-expanded','false');$('#mobile-controls').focus({preventScroll:true});}
$('#mobile-controls').addEventListener('click',()=>{
  const open=document.body.classList.toggle('controls-open');
  $('#mobile-controls').setAttribute('aria-expanded',String(open));
  if(open) $('#controls input, #controls select')?.focus({preventScroll:true});
});
document.addEventListener('keydown',event=>{if(event.key==='Escape'&&!$('#about').open&&document.body.classList.contains('controls-open'))closeControls();});
$('.skip').addEventListener('click',event=>{event.preventDefault();$('#workspace').focus();$('#workspace').scrollIntoView({block:'start'});});
window.addEventListener('hashchange',setRoute);
setRoute();
