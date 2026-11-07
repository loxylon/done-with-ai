/** MEGA / OMEGA: точные дискретные распределения и маленький off-policy опыт.
 * Источник: Pitis et al., ICML 2020, §2–3, Algorithm 2; supplement B–C.
 * Гистограмма заменяет KDE, табличный Q-learning + future HER заменяет DDPG.
 */
import { rng, sum, clamp } from '../math.js';

export const GOALS = ['A','B','C','D','E'];
export const COLORS = ['#b79a65','#72a691','#528c96','#8275a4','#d47551'];
export const METHODS = ['HER','Achieved','Diverse','MEGA','OMEGA'];
export const DEFAULT_COUNTS = [80,18,2,0,0];
export const DEFAULT_Q = [-1,-3,-5,-12,-16];
export const DEFAULT_TRAIN = {seed:17,edge:3,horizon:24,candidates:100,updates:180,
  epsilon:.1,bonus:.1,learningRate:.3,gamma:.95,bias:-3,blocked:false};
const zeros = () => Array(5).fill(0);
export const normalize = weights => {
  const total=sum(weights);
  return total>0 ? weights.map(x=>x/total) : zeros();
};
const xlogx = x => x===0 ? 0 : x*Math.log(x);
export const entropy = probabilities => -sum(probabilities.map(xlogx));
export function kl(desired, achieved) {
  let result=0;
  for(let g=0;g<5;g++) {
    if(desired[g]===0)continue;
    if(achieved[g]===0)return Infinity;
    result+=desired[g]*Math.log(desired[g]/achieved[g]);
  }
  return Math.max(0,result);
}
export function omega(counts, desired, bias=-3) {
  const p=normalize(counts), valid=sum(counts)>0&&sum(desired)>0;
  const divergence=valid?kl(desired,p):Infinity;
  const alpha=Number.isFinite(divergence)?1/Math.max(bias+divergence,1):0;
  const support=counts.filter(x=>x>0).length;
  const uniform=counts.map(x=>x>0?1/support:0);
  const mixture=desired.map((x,g)=>alpha*x+(1-alpha)*uniform[g]);
  return {valid,p,divergence,alpha,uniform,mixture,objective:valid?kl(mixture,p):null};
}
export function categorical(probabilities, random) {
  const total=sum(probabilities); if(total<=0)return null;
  const u=random()*total;
  let cumulative=0;
  for(let g=0;g<probabilities.length;g++){cumulative+=probabilities[g];if(u<cumulative)return g;}
  return probabilities.findLastIndex(x=>x>0);
}
export function pool(counts, candidates, random) {
  const p=normalize(counts);
  return sum(counts)>0?Array.from({length:candidates},()=>categorical(p,random)):[];
}

export function inspectPool(counts, candidateGoals, values=DEFAULT_Q, cutoff=-20) {
  const p=normalize(counts);
  const eligible=candidateGoals.filter(g=>values[g]>=cutoff);
  const occurrences=zeros(), weights=zeros();
  for(const g of eligible){occurrences[g]++;weights[g]+=1/p[g];}
  const winner=eligible.length?eligible.reduce((best,g)=>p[g]<p[best]?g:best):null;
  return {p,eligible,occurrences,achieved:normalize(occurrences),diverse:normalize(weights),winner,
    mega:GOALS.map((_,g)=>Number(g===winner))};
}

/** Вероятность победы по многим повторным выборкам N кандидатов, до наблюдения pool.
 * При равной density побеждает первый встретившийся кандидат; отвергнутые кандидаты
 * не маскируются в генераторе: они отсеиваются ПОСЛЕ выборки, как в Algorithm 2.
 */
export function minimumDensityLaw(counts,candidates,values=DEFAULT_Q,cutoff=-20) {
  const p=normalize(counts), eligible=counts.map((n,g)=>n>0&&values[g]>=cutoff);
  const probabilities=p.map((density,g)=>{
    if(!eligible[g])return 0;
    const lower=sum(p.filter((d,j)=>eligible[j]&&d<density));
    const tied=sum(p.filter((d,j)=>eligible[j]&&d===density));
    return density/tied*((1-lower)**candidates-Math.max(0,1-lower-tied)**candidates);
  });
  const failure=Math.max(0,1-sum(probabilities));
  return {probabilities,failure};
}

export function selectionLab({counts=DEFAULT_COUNTS,desired=[0,0,0,0,1],values=DEFAULT_Q,
  cutoff=-20,candidates=100,seed=2020,bias=-3}={}) {
  const random=rng(seed), candidateGoals=pool(counts,candidates,random);
  const inspected=inspectPool(counts,candidateGoals,values,cutoff);
  const mix=omega(counts,normalize(desired),bias);
  const u=random();
  const probabilities={HER:normalize(desired),Achieved:inspected.achieved,Diverse:inspected.diverse,
    MEGA:inspected.mega,OMEGA:desired.map((_,g)=>mix.alpha*normalize(desired)[g]+(1-mix.alpha)*inspected.mega[g])};
  const law=minimumDensityLaw(counts,candidates,values,cutoff);
  return {...inspected,candidateGoals,mix,u,external:u<mix.alpha,probabilities,law};
}

export function outcomeAnalysis(counts,conditional) {
  const size=sum(counts); if(size===0)return null;
  const p=normalize(counts), current=entropy(p), eta=1/size;
  const outcomes=GOALS.map((_,g)=>{
    const next=counts.slice();next[g]++;
    const after=entropy(normalize(next));
    // Eq.5 — score с отброшенными общими константами; это не буквально ΔH.
    const score=xlogx(p[g])-xlogx(p[g]+eta);
    return {after,delta:after-current,score,next:normalize(next)};
  });
  const candidates=conditional.map(row=>{
    const q=normalize(row);
    return {q,valid:sum(row)>0,expected:sum(q.map((v,g)=>v*outcomes[g].after)),
      gain:sum(q.map((v,g)=>v*outcomes[g].delta)),score:sum(q.map((v,g)=>v*outcomes[g].score))};
  });
  return {p,size,eta,current,outcomes,candidates};
}
export function conditionalPreset(kind='perfect') {
  const q=GOALS.map((_,g)=>GOALS.map((_,j)=>Number(g===j)));
  if(kind==='broken')q[2]=[1,0,0,0,0];
  if(kind==='frontier')for(let g=0;g<4;g++){q[g][g]=.7;q[g][g+1]=.3;}
  return q;
}

export function normalizeLab(input={}) {
  if(!input||typeof input!=='object')input={};
  const lab={counts:DEFAULT_COUNTS.slice(),desired:[0,0,0,0,1],values:DEFAULT_Q.slice(),cutoff:-20,
    candidates:100,seed:2020,bias:-3,conditional:conditionalPreset(),focus:2};
  for(const [key,lo,hi] of [['counts',0,50000],['desired',0,100],['values',-30,0]]) {
    if(Array.isArray(input[key])&&input[key].length===5&&input[key].every(Number.isFinite))
      lab[key]=input[key].map(x=>key==='counts'?Math.round(clamp(x,lo,hi)):clamp(x,lo,hi));
  }
  for(const [key,lo,hi] of [['cutoff',-30,0],['candidates',1,100],['seed',0,999999],['bias',-3,1],['focus',0,4]])
    if(Number.isFinite(input[key]))lab[key]=clamp(input[key],lo,hi);
  for(const key of ['candidates','seed','focus'])lab[key]=Math.round(lab[key]);
  if(Array.isArray(input.conditional)&&input.conditional.length===5&&input.conditional.every(row=>Array.isArray(row)&&row.length===5&&row.every(Number.isFinite)))
    lab.conditional=input.conditional.map(row=>row.map(x=>clamp(x,0,100)));
  return lab;
}

export function trainConfig(input={}) {
  const config={...DEFAULT_TRAIN};
  const bounds={seed:[0,999999],edge:[1,4],horizon:[8,40],candidates:[1,100],updates:[20,300],
    epsilon:[0,1],bonus:[0,.5],learningRate:[.01,1],gamma:[.5,.99],bias:[-3,1]};
  for(const [key,[lo,hi]] of Object.entries(bounds))if(Number.isFinite(input[key]))config[key]=clamp(input[key],lo,hi);
  for(const key of ['seed','edge','horizon','candidates','updates'])config[key]=Math.round(config[key]);
  if(typeof input.blocked==='boolean')config.blocked=input.blocked;
  return config;
}
export const achievedGoal=(state,config)=>Math.min(4,Math.floor(state/config.edge));
export function transition(state,action,config) {
  const end=config.edge*4-(config.blocked?1:0);
  return clamp(state+[ -1,1,0 ][action],0,end);
}
const qIndex=(s,a,g)=>(s*3+a)*5+g;
export const qValues=(model,state,goal)=>[0,1,2].map(a=>model.q[qIndex(state,a,goal)]);
const maximizing=values=>values.map((v,a)=>v===Math.max(...values)?a:-1).filter(a=>a>=0);

function optimize(model,random) {
  const logs=[], c=model.config;
  for(let update=0;update<c.updates;update++) {
    const item=model.replay[Math.floor(random()*model.replay.length)];
    const relabel=random()<.8;
    const future=relabel?item.episode[item.t+Math.floor(random()*(item.episode.length-item.t))]:null;
    const goal=relabel?achievedGoal(future.next,c):item.goal;
    const reward=achievedGoal(item.next,c)===goal?0:-1;
    const index=qIndex(item.state,item.action,goal), before=model.q[index];
    const bootstrap=reward===0?0:Math.max(...qValues(model,item.next,goal));
    const target=reward+c.gamma*bootstrap;
    model.q[index]=before+c.learningRate*(target-before);
    if(logs.length<6)logs.push({state:item.state,action:item.action,next:item.next,goal,originalGoal:item.goal,
      episodeIndex:item.episode[0].episodeIndex??-1,sourceT:item.t,
      relabel,futureT:future?.t??null,futureNext:future?.next??null,reward,before,bootstrap,target,after:model.q[index]});
  }
  return logs;
}
function appendEpisode(model,steps) {
  for(const s of steps)model.replay.push({...s,episode:steps});
}
export function createModel(method,input={}) {
  const config=trainConfig(input), end=config.edge*4;
  const model={method,config,q:Array((end+1)*3*5).fill(-1/(1-config.gamma)),counts:[1,0,0,0,0],
    replay:[],history:[],totalSteps:0,firstDesired:null};
  const random=rng(config.seed+2003), warmup=[];let state=0;
  for(let t=0;t<6;t++){
    const action=Math.floor(random()*3),next=transition(state,action,config);
    warmup.push({state,action,next,goal:0,t});model.counts[achievedGoal(next,config)]++;
    if(achievedGoal(next,config)===4)model.firstDesired=0;
    state=next;
  }
  appendEpisode(model,warmup);optimize(model,rng(config.seed+4001));
  model.totalSteps=6;
  return model;
}

export function evaluate(model) {
  let state=0;const path=[0];
  for(let t=0;t<model.config.horizon;t++) {
    const choices=maximizing(qValues(model,state,4));
    state=transition(state,choices[0],model.config);path.push(state);
    if(achievedGoal(state,model.config)===4)return {success:true,path};
  }
  return {success:false,path};
}

export function trainEpisode(model) {
  const c=model.config,index=model.history.length;
  const random=rng(c.seed+index*10007+71);
  const p=normalize(model.counts), desired=[0,0,0,0,1];
  const mix=omega(model.counts,desired,c.bias);
  const candidates=pool(model.counts,c.candidates,random);
  // Учебный фильтр использует точную проходимость. Динамический Q cutoff статьи
  // отдельно разобран в инспекторе; он не подменяется «идеальной» Q-оценкой здесь.
  const values=GOALS.map((_,g)=>g===4&&c.blocked?-100:0);
  const inspected=inspectPool(model.counts,candidates,values,-50);
  let goal;
  if(model.method==='HER')goal=4;
  if(model.method==='Achieved')goal=categorical(inspected.achieved,random);
  if(model.method==='Diverse')goal=categorical(inspected.diverse,random);
  if(model.method==='MEGA')goal=inspected.winner;
  if(model.method==='OMEGA')goal=random()<mix.alpha?4:inspected.winner;
  if(goal===null)goal=0; // В учебной среде стартовая область всегда доступна.
  const actionRandom=rng(c.seed+index*10007+1709), steps=[];
  const before=model.counts.slice();
  model.counts[0]++;let state=0,epsilon=c.epsilon,hits=0;
  for(let t=0;t<c.horizon;t++){
    const values=qValues(model,state,goal), choices=maximizing(values);
    // Одинаковое число random draws на шаг у всех методов: общий seed для сравнения.
    const u=actionRandom(), explore=u<epsilon;
    const action=explore?Math.floor(actionRandom()*3):choices[Math.floor(actionRandom()*choices.length)];
    const next=transition(state,action,c), achieved=achievedGoal(next,c);
    const reward=achieved===goal?0:-1;
    steps.push({t,episodeIndex:index,state,action,next,goal,achieved,reward,epsilon,explore,u,values});
    model.counts[achieved]++; state=next;
    if(reward===0){hits++;epsilon=Math.min(1,epsilon+c.bonus);}
    if(achieved===4&&model.firstDesired===null)model.firstDesired=index+1;
  }
  appendEpisode(model,steps);
  const updates=optimize(model,rng(c.seed+index*10007+3109));
  model.totalSteps+=steps.length;
  const record={index,goal,alpha:mix.alpha,divergence:mix.divergence,pBefore:p,before,after:model.counts.slice(),
    entropy:entropy(normalize(model.counts)),coverage:model.counts.filter(n=>n>0).length,
    hits,steps,updates,evaluation:evaluate(model)};
  model.history.push(record);
  return record;
}
