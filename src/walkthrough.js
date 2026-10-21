import katex from '../vendor/katex.mjs';
import { normalizeConfig, runExperiment, runIteration, DEFAULT_CONFIG, WALL } from './walkthrough-engine.js';
import { scenario, world, calculations, iterationHistory, STAGES } from './walkthrough-view.js';

const $ = selector => document.querySelector(selector);
const boundedInteger = (value, minimum, maximum, fallback = minimum) => Number.isInteger(value)
  ? Math.max(minimum, Math.min(maximum, value)) : fallback;

function restore() {
  try {
    const raw = new URLSearchParams(location.search).get('experiment');
    if (!raw || raw.length > 5000) return {};
    const saved = JSON.parse(raw);
    return saved && typeof saved === 'object' ? saved : {};
  } catch { return {}; }
}

const saved = restore();
let config = normalizeConfig(saved.config);
let iterationIndex = boundedInteger(saved.iteration, 0, config.iterations - 1);
let history = runExperiment(config, iterationIndex + 1).iterations;
let cursor = boundedInteger(saved.step, 0, history[iterationIndex].steps.length);
let stage = boundedInteger(saved.stage, 0, STAGES.length - 1);
let epochIndex = boundedInteger(saved.epoch, 0, config.epochs - 1);
let focusGoal = boundedInteger(saved.goal, 0, 4);
let inspectState = boundedInteger(saved.inspectState, 0, 11, 4);
if (inspectState === WALL) inspectState = 4;
let inspectGoal = boundedInteger(saved.inspectGoal, 0, 4);
let source = null;

function status(text) { $('#experiment-status').textContent = text; }
function render() {
  const focused = document.activeElement;
  const focusKey = focused?.dataset && ['stage','step','iteration','epoch','sourceStep','command']
    .find(key => focused.dataset[key] !== undefined);
  const activeId = focused?.id;
  const context = { iteration: history[iterationIndex], history, iterationIndex, cursor, stage, epochIndex,
    focusGoal, inspectState, inspectGoal, source };
  $('#world-panel').innerHTML = world(context);
  $('#calculations').innerHTML = calculations(context);
  $('#iteration-history').innerHTML = iterationHistory(history, iterationIndex);
  $('#calculations').querySelectorAll('[data-tex]').forEach(el => {
    katex.render(el.dataset.tex, el, { displayMode: true, throwOnError: false, strict: 'ignore', trust: false });
  });
  if (focusKey) {
    const attr = focusKey.replace(/[A-Z]/g, c => '-'+c.toLowerCase());
    document.querySelector(`[data-${attr}="${focused.dataset[focusKey]}"]`)?.focus({preventScroll:true});
  } else if (activeId) document.getElementById(activeId)?.focus({preventScroll:true});
}

function restart(input, message) {
  config = normalizeConfig(input);
  history = runExperiment(config, 1).iterations;
  iterationIndex = 0; cursor = 0; stage = 0; epochIndex = 0; source = null;
  $('#scenario').innerHTML = scenario(config);
  render();
  status(message || `Опыт пересчитан с начала. Seed ${config.seed}; параметры сбора заморожены внутри каждой итерации.`);
}

function nextStep() {
  const iteration = history[iterationIndex];
  source = null;
  if (cursor < iteration.steps.length) {
    cursor++;
    if (cursor === iteration.steps.length) stage = 5;
    else if (stage === 5) stage = 0;
  } else if (iterationIndex + 1 < config.iterations) {
    if (!history[iterationIndex + 1]) history.push(runIteration(iteration.after, config, iterationIndex + 1));
    iterationIndex++; cursor = 0; stage = 0; epochIndex = 0;
    status(`Открыта итерация ${iterationIndex+1}. Её начальные φ, ψ, θ и V точно совпадают с итогом итерации ${iterationIndex}. Новый rollout использует обновлённую policy.`);
  }
  render();
}

function previousStep() {
  source = null;
  if (cursor > 0) cursor--;
  else if (iterationIndex > 0) { iterationIndex--; cursor = history[iterationIndex].steps.length; }
  stage = cursor === history[iterationIndex].steps.length ? 5 : Math.min(stage,4);
  render();
}

document.addEventListener('click', event => {
  const button = event.target.closest('button');
  if (!button || button.disabled) return;
  if (button.dataset.stage !== undefined) { stage = +button.dataset.stage; render(); return; }
  if (button.dataset.step !== undefined) {
    cursor = +button.dataset.step; source = null;
    if (cursor === history[iterationIndex].steps.length) stage = 5;
    else if (stage === 5) stage = 0;
    render(); return;
  }
  if (button.dataset.iteration !== undefined) {
    iterationIndex = +button.dataset.iteration; cursor = 0; stage = 0; source = null; epochIndex = 0;
    render(); return;
  }
  if (button.dataset.epoch !== undefined) { epochIndex = +button.dataset.epoch; render(); return; }
  if (button.dataset.sourceStep !== undefined) {
    source = source === +button.dataset.sourceStep ? null : +button.dataset.sourceStep;
    render();
    status(source === null ? 'Подсветка источника снята.' : `Источник слагаемого — observation после шага t=${source}. На карте и ленте она выделена фиолетовым.`);
    return;
  }
  if (button.dataset.command === 'next-step') nextStep();
  if (button.dataset.command === 'previous-step') previousStep();
  if (button.dataset.command === 'next-formula') {
    if (stage < STAGES.length-1) { stage++; render(); } else { stage=0; nextStep(); }
    $('#calculations').scrollIntoView({block:'start',behavior:'smooth'});
  }
  if (button.dataset.command === 'default') restart(DEFAULT_CONFIG, 'Восстановлен исходный опыт: пять целей, seed 1724, первый rollout заканчивается смертью.');
});

document.addEventListener('change', event => {
  const el = event.target;
  if (el.dataset.goalToggle !== undefined) {
    const enabled = config.enabled.slice(); enabled[+el.dataset.goalToggle] = el.checked;
    restart({...config, enabled}, 'Набор целей изменён. Опыт начат заново с тем же seed, чтобы старые и новые метки не смешивались.');
    $(`[data-goal-toggle="${el.dataset.goalToggle}"]`)?.focus({preventScroll:true});
  }
  if (el.hasAttribute('data-focus-goal')) { focusGoal = +el.value; render(); $('[data-focus-goal]').focus({preventScroll:true}); }
  if (el.hasAttribute('data-inspect-state')) { inspectState = +el.value; render(); $('[data-inspect-state]').focus({preventScroll:true}); }
  if (el.hasAttribute('data-inspect-goal')) { inspectGoal = +el.value; render(); $('[data-inspect-goal]').focus({preventScroll:true}); }
});

document.addEventListener('submit', event => {
  if (event.target.id !== 'trace-config-form') return;
  event.preventDefault();
  const data = new FormData(event.target);
  const input = {...config};
  for (const [key,value] of data) input[key] = key === 'danger' ? value === 'true' : Number(value);
  restart(input);
});

document.addEventListener('keydown', event => {
  if (event.altKey || event.ctrlKey || event.metaKey || event.shiftKey) return;
  if (event.target.closest('.trace-tabs') && ['ArrowLeft','ArrowRight','Home','End'].includes(event.key)) {
    event.preventDefault();
    stage = event.key === 'Home' ? 0 : event.key === 'End' ? STAGES.length-1 : (stage + (event.key === 'ArrowRight' ? 1 : STAGES.length-1)) % STAGES.length;
    render(); $(`#formula-tab-${stage}`).focus(); return;
  }
  if (event.target.closest('input,select,textarea,[contenteditable="true"],summary,details[open]')) return;
  if (event.key === 'ArrowRight') { event.preventDefault(); nextStep(); }
  if (event.key === 'ArrowLeft') { event.preventDefault(); previousStep(); }
});

$('#share-experiment').addEventListener('click', async () => {
  const url = new URL(location.href); url.search=''; url.hash='';
  url.searchParams.set('experiment', JSON.stringify({config, iteration:iterationIndex, step:cursor, stage,
    epoch:epochIndex, goal:focusGoal, inspectState, inspectGoal}));
  historyReplace(url);
  try { await navigator.clipboard.writeText(url.href); status('Ссылка скопирована: цели, параметры, итерация, шаг и вкладка восстановятся точно.'); }
  catch { status('Опыт сохранён в адресной строке. Скопируйте её, чтобы вернуться к тем же числам.'); }
});

function historyReplace(url) { window.history.replaceState(null,'',url); }

$('#export-experiment').addEventListener('click', () => {
  const data = { schema:1, description:'CRL Lab: табличный Dual-CRL-PPO, analytic SGD, float64 без округления.',
    config, openIterations:history.length, selected:{iteration:iterationIndex,step:cursor,stage,epoch:epochIndex}, iterations:history };
  const url = URL.createObjectURL(new Blob([JSON.stringify(data,null,2)],{type:'application/json'}));
  const link = document.createElement('a'); link.href=url; link.download=`crl-step-by-step-seed-${config.seed}.json`;
  link.click(); setTimeout(()=>URL.revokeObjectURL(url),1000);
  status(`В JSON включены ${history.length} открытых итераций: параметры до/после, все logits, targets, losses и gradients каждого epoch.`);
});

$('#scenario').innerHTML = scenario(config);
render();
status(`Готов журнал итерации ${iterationIndex+1}. Действия сэмплированы из πold; seed ${config.seed}. Будущее используется только после сбора.`);
