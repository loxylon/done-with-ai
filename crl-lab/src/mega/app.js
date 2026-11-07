import katex from '../../vendor/katex.mjs';
import {METHODS,GOALS,DEFAULT_COUNTS,DEFAULT_Q,normalizeLab,trainConfig,createModel,trainEpisode,
  conditionalPreset,normalize,categorical,achievedGoal} from './engine.js';
import {rng} from '../math.js';
import {PAGES,page} from './view.js';

const $=selector=>document.querySelector(selector);
const integer=(value,minimum,maximum,fallback=minimum)=>Number.isInteger(value)?Math.max(minimum,Math.min(maximum,value)):fallback;
function savedState() {
  try {const raw=new URLSearchParams(location.search).get('scenario');if(!raw||raw.length>6000)return {};
    const parsed=JSON.parse(raw);return parsed&&typeof parsed==='object'?parsed:{};
  } catch {return {};}
}
const saved=savedState();
let lab=normalizeLab(saved.lab), train=trainConfig(saved.train||{});
let method=METHODS.includes(saved.method)?saved.method:'MEGA';
let trainMethod=METHODS.includes(saved.trainMethod)?saved.trainMethod:'MEGA';
let models=Object.fromEntries(METHODS.map(m=>[m,createModel(m,train)]));
const count=integer(saved.count,0,300);
for(let i=0;i<count;i++)for(const m of METHODS)trainEpisode(models[m]);
let episodeIndex=integer(saved.episodeIndex,0,Math.max(0,count-1));
let stepIndex=integer(saved.stepIndex,0,train.horizon-1),route='problem',busy=false;

function status(message) {$('#mega-status').textContent=message;}
const episodesWord=n=>n%10===1&&n%100!==11?'эпизод':n%10>=2&&n%10<=4&&!(n%100>=12&&n%100<=14)?'эпизода':'эпизодов';
function render() {
  const active=document.activeElement;
  const key=active?.dataset&&['lab','outcome','focus','episodeIndex','stepIndex','action'].find(k=>active.dataset[k]!==undefined);
  const context={lab,method,models,train,trainMethod,episodeIndex,stepIndex,route,busy};
  $('#mega-content').innerHTML=page(context);
  $('#mega-content').querySelectorAll('[data-tex]').forEach(el=>katex.render(el.dataset.tex,el,
    {displayMode:true,throwOnError:false,strict:'ignore',trust:false}));
  if(key){
    const attribute=key.replace(/[A-Z]/g,c=>'-'+c.toLowerCase());
    let selector=`[data-${attribute}="${active.dataset[key]}"]`;
    if(active.dataset.episodes)selector+=`[data-episodes="${active.dataset.episodes}"]`;
    document.querySelector(selector)?.focus({preventScroll:true});
  }
}
function changeRoute() {
  const requested=location.hash.slice(1);
  route=PAGES[requested]?requested:'problem';
  const p=PAGES[route];
  $('#mega-eyebrow').textContent=p.eyebrow;$('#mega-title').innerHTML=p.title;$('#mega-description').textContent=p.description;
  document.title=`${p.title.replace(/<[^>]*>/g,' ').replace(/\s+/g,' ')} — MEGA Lab`;
  document.querySelectorAll('[data-pane]').forEach(el=>{if(el.dataset.pane===route)el.setAttribute('aria-current','page');else el.removeAttribute('aria-current');});
  render();window.scrollTo({top:0,behavior:'instant'});
}
function preset(id) {
  lab=normalizeLab({...lab,counts:DEFAULT_COUNTS.slice(),values:DEFAULT_Q.slice(),cutoff:-20,desired:[0,0,0,0,1]});
  if(id==='rare'){lab.counts=[80,18,2,0,1];lab.values[4]=-25;lab.cutoff=-10;}
  if(id==='empty')lab.counts=[0,0,0,0,0];
  if(id==='balanced')lab.counts=[20,20,20,20,20];
  if(id==='contact')lab.counts=[800,180,20,0,1];
  if(id==='covered')lab.counts=[15,10,10,15,50];
  render();status('Сценарий буфера изменён. Опыт среды независим и сохранён.');
}
async function runEpisodes(requested) {
  if(busy)return;
  const n=Math.min(requested,300-models.MEGA.history.length);
  busy=true;render();
  try {
    for(let i=0;i<n;i++){
      for(const m of METHODS)trainEpisode(models[m]);
      if(i%10===0){status(`Обучение: ${models.MEGA.history.length} эпизодов у каждого метода…`);await new Promise(resolve=>setTimeout(resolve,0));}
    }
    episodeIndex=Math.max(0,models.MEGA.history.length-1);stepIndex=0;
    const total=models.MEGA.history.length;
    status(`Готово: ${total} ${episodesWord(total)} у каждого метода. Можно открыть любой эпизод и шаг, сравнить буферы и посмотреть реальные HER / Q updates.`);
  } finally {busy=false;render();}
}
document.addEventListener('click',event=>{
  const el=event.target.closest('[data-action]');if(!el||el.disabled)return;
  const action=el.dataset.action;
  if(action==='preset'){preset(el.dataset.preset);return;}
  if(action==='resample'){lab.seed=(lab.seed+1)%1000000;render();status('Новая выборка кандидатов. Гистограмма и оценки достижимости не менялись.');}
  if(action==='method'){method=el.dataset.method;render();}
  if(action==='train-method'){trainMethod=el.dataset.method;render();}
  if(action==='conditional-preset'){
    lab.conditional=conditionalPreset(el.dataset.preset);lab.focus=2;
    if(el.dataset.preset==='broken'){lab.counts=DEFAULT_COUNTS.slice();lab.values=DEFAULT_Q.slice();lab.cutoff=-20;lab.seed=2020;}
    render();status('Изменена учебная модель q. Практический MEGA её не знает; oracle использует её только для объяснения.');
  }
  if(action==='add-outcome'){
    const outcome=categorical(normalize(lab.conditional[lab.focus]),rng(lab.seed));
    lab.seed=(lab.seed+1)%1000000;
    if(outcome!==null&&lab.counts[outcome]<50000){lab.counts[outcome]++;render();status(`Добавлено одно наблюдение ${GOALS[outcome]} согласно q. Буфер нормирован заново.`);}
    else status('Для выбранного исхода достигнут предел 50 000 наблюдений. Уменьшите счётчик в инспекторе.');
  }
  if(action==='train')void runEpisodes(+el.dataset.episodes);
  if(action==='reset-training'&&!busy){models=Object.fromEntries(METHODS.map(m=>[m,createModel(m,train)]));episodeIndex=0;stepIndex=0;render();status('Все пять агентов сброшены. Разогрев одинаковый; seed сохранён.');}
  const record=models[trainMethod].history[episodeIndex];
  if(action==='previous-step'&&record){stepIndex=Math.max(0,stepIndex-1);render();}
  if(action==='next-step'&&record){stepIndex=Math.min(record.steps.length-1,stepIndex+1);render();}
  if(action==='next-record'&&record){episodeIndex=Math.min(models[trainMethod].history.length-1,episodeIndex+1);stepIndex=0;render();}
  if(action==='copy-buffer'){
    const m=models[trainMethod];lab.counts=(record?.after||m.counts).slice();
    lab.values=GOALS.map((_,g)=>Math.max(...[0,1,2].map(a=>m.q[(a*5)+g])));
    status(`Гистограмма ${trainMethod} скопирована в инспектор. Q-оценки взяты из текущей таблицы агента; история среды сохранена.`);
    location.hash='selector';if(route==='selector')render();
  }
});

function updateLab(el) {
  const key=el.dataset.lab,value=Number(el.value);if(!Number.isFinite(value))return;
  if(key.startsWith('count-'))lab.counts[+key.slice(6)]=value;
  else if(key.startsWith('value-'))lab.values[+key.slice(6)]=value;
  else if(key.startsWith('desired-'))lab.desired[+key.slice(8)]=value;
  else if(['cutoff','candidates','seed','bias'].includes(key))lab[key]=value;
  lab=normalizeLab(lab);render();
}
document.addEventListener('change',event=>{
  const el=event.target;
  if(el.dataset.lab!==undefined&&el.type!=='range')updateLab(el);
  if(el.dataset.outcome!==undefined){lab.conditional[lab.focus][+el.dataset.outcome]=Number(el.value);lab=normalizeLab(lab);render();}
  if(el.hasAttribute('data-focus')){lab.focus=+el.value;render();}
  if(el.hasAttribute('data-episode-index')){episodeIndex=+el.value;stepIndex=0;render();}
  if(el.hasAttribute('data-step-index')){stepIndex=+el.value;render();}
});
document.addEventListener('input',event=>{if(event.target.dataset.lab==='bias')updateLab(event.target);});
document.addEventListener('submit',event=>{
  if(event.target.id!=='mega-train-form')return;event.preventDefault();if(busy)return;
  const values={};for(const [key,value] of new FormData(event.target))values[key]=key==='blocked'?value==='true':Number(value);
  train=trainConfig(values);models=Object.fromEntries(METHODS.map(m=>[m,createModel(m,train)]));episodeIndex=0;stepIndex=0;
  render();status('Параметры применены ко всем пяти методам. Обучение начато с одинакового разогрева.');
});
$('#mega-share').addEventListener('click',async()=>{
  const url=new URL(location.href);url.search='';url.searchParams.set('scenario',JSON.stringify({lab,train,method,trainMethod,
    count:models.MEGA.history.length,episodeIndex,stepIndex}));url.hash=route;
  window.history.replaceState(null,'',url);
  try{await navigator.clipboard.writeText(url.href);status('Ссылка скопирована: восстановятся буфер, параметры, все эпизоды и выбранный шаг.');}
  catch{status('Сценарий сохранён в адресной строке: скопируйте ссылку.');}
});
$('.skip').addEventListener('click',event=>{event.preventDefault();$('#mega-content').focus();$('#mega-content').scrollIntoView({block:'start'});});
window.addEventListener('hashchange',changeRoute);
changeRoute();
status('Начните с постановки задачи, затем выберите цель в инспекторе или запустите обучение на вкладке 03.');
