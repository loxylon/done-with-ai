import { test } from 'node:test';
import assert from 'node:assert/strict';
import { normalizeConfig, move, semantics, initialize, runExperiment, runIteration,
  criticForward, criticUpdate, buildTarget, advantages, policyTerms } from '../src/walkthrough-engine.js';
import { sum, softmax } from '../src/math.js';

const close = (a,b,tolerance=1e-9) => assert.ok(Math.abs(a-b)<tolerance,`${a} != ${b}`);
const vector = (a,b,tolerance) => { assert.equal(a.length,b.length); a.forEach((v,i)=>close(v,b[i],tolerance)); };
const copy = value => structuredClone(value);
const derivative = (fn, x) => { const h=1e-6; return (fn(x+h)-fn(x-h))/(2*h); };

test('default rollout has visible overlap, reward and fatal action; target recovers geometric mass',()=>{
  const it=runExperiment({}).iterations[0];
  assert.deepEqual(it.steps.map(s=>s.action),[0,1,1,1,2]);
  assert.deepEqual(it.steps.map(s=>s.next),[0,1,2,3,7]);
  assert.equal(it.reward,1); assert.equal(it.fatal,true);
  vector(it.steps[2].semantic.labels,[1/3,0,0,1/3,1/3]);
  assert.equal(it.steps[2].reward,1);
  const target=it.critic.anchors[0].target;
  assert.equal(target.distance,4);
  vector(target.y,[.027,0,0,.25345,.06345,.6561]);
  vector(it.critic.anchors[4].target.y,[0,0,0,0,0,1]);
  assert.equal(it.critic.anchors[4].target.futures.length,0);
  assert.ok(it.critic.anchors[4].scoreGradient.every(x=>x>0));
  const capped=buildTarget(it.steps,0,{...it.config,cap:1});
  assert.equal(capped.distance,4); close(capped.y[3],1-.9**4);
  vector(semantics(2,[false,true,true,true,true]).labels,[0,0,0,.5,.5]);
});

test('boundaries, empty predicates, truncation and all goal subsets conserve target mass',()=>{
  assert.equal(move(4,1).blocked,true); assert.equal(move(0,0).next,0);
  assert.equal(move(3,2,true).fatal,true); assert.equal(move(3,2,false).fatal,false);
  for(let bits=0;bits<32;bits++) for(const danger of [false,true]){
    const enabled=Array.from({length:5},(_,i)=>Boolean(bits&(1<<i)));
    const config=normalizeConfig({enabled,danger,cap:2});
    const it=runExperiment(config).iterations[0];
    it.critic.anchors.forEach(a=>{
      close(sum(a.target.y),1); assert.ok(a.target.y.every(y=>y>=-1e-14));
      a.target.y.slice(0,5).forEach((y,g)=>{if(!enabled[g])close(y,0);});
    });
    if(!danger){assert.equal(it.fatal,false);assert.equal(it.steps.at(-1).truncated,true);assert.equal(it.critic.anchors[0].target.distance,null);}
  }
});

test('factorized critic batch-gradients equal numerical derivatives, including shared psi and repeated (s,a)',()=>{
  const config=normalizeConfig({seed:17,danger:false,enabled:[true,false,true,true,true]});
  const before=initialize(config);
  const it=runIteration(before,config);
  const update=it.critic;
  const loss=params=>sum(it.steps.map((s,t)=>{
    const forward=criticForward(params,s.state,s.action,config.enabled);
    return -sum(update.anchors[t].target.y.map((y,g)=>y*forward.logProbabilities[g]));
  }))/it.steps.length;
  const pairs=new Set(it.steps.map(s=>`${s.state},${s.action}`));
  for(const pair of pairs){
    const [s,a]=pair.split(',').map(Number);
    for(let j=0;j<2;j++){
      const numerical=derivative(x=>{const params=copy(before);params.phi[s][a][j]=x;return loss(params);},before.phi[s][a][j]);
      close(update.phiGradient[s][a][j],numerical,2e-9);
      close(update.after.phi[s][a][j],before.phi[s][a][j]-config.criticLR*numerical,2e-9);
    }
  }
  for(let g=0;g<5;g++)for(let j=0;j<2;j++){
    const numerical=derivative(x=>{const params=copy(before);params.psi[g][j]=x;return loss(params);},before.psi[g][j]);
    close(update.psiGradient[g][j],numerical,2e-9);
  }
  vector(update.psiGradient[1],[0,0]);
  assert.deepEqual(before,initialize(config),'input snapshot mutated');
});

test('GAE separates death, time-limit bootstrap and recurrence with nonzero value',()=>{
  const config=normalizeConfig({gamma:.9,lambda:.8});
  const steps=[
    {reward:0,value:.3,nextValue:.5,fatal:false,truncated:false},
    {reward:1,value:.5,nextValue:2,fatal:false,truncated:true},
  ];
  const gae=advantages(steps,config);
  close(gae[1].delta,1+.9*2-.5); assert.equal(gae[1].bootstrap,1); assert.equal(gae[1].continuation,0);
  close(gae[0].advantage,.9*.5-.3+.9*.8*gae[1].advantage);
  close(gae[0].return,gae[0].advantage+.3);
  const fatal=advantages([{...steps[1],fatal:true,truncated:false}],config)[0];
  close(fatal.advantage,.5);assert.equal(fatal.bootstrap,0);assert.equal(fatal.continuation,0);
});

test('all four PPO logit gradients and value gradient match finite differences on both clip branches',()=>{
  const config=normalizeConfig({entropy:.03});
  for(const logits of [[1,-.4,.2,-.5],[-2,.4,.1,.2]])for(const advantage of [-1.3,.7]){
    const step={action:0,policy:[.25,.25,.25,.25]};
    const gae={advantage,return:.8};
    const value=.3,beta=.4,teacher=2;
    const current=policyTerms(logits,value,step,gae,teacher,beta,config);
    logits.forEach((x,a)=>{
      const numerical=derivative(v=>{const z=logits.slice();z[a]=v;return policyTerms(z,value,step,gae,teacher,beta,config).loss;},x);
      close(current.gradient[a],numerical,2e-9);
    });
    close(current.valueGradient,derivative(v=>policyTerms(logits,v,step,gae,teacher,beta,config).loss,value),2e-9);
    close(sum(current.gradient),0);
    const expectedClip=(advantage>0&&current.ratio>1+config.clip)||(advantage<0&&current.ratio<1-config.clip);
    assert.equal(current.clippedActive,expectedClip);
    if(expectedClip) vector(current.policyGradient,[0,0,0,0]);
  }
});

test('PPO accumulates repeated visits before a synchronous step; old policy stays frozen across epochs',()=>{
  const it=runExperiment({seed:17,danger:false}).iterations[0];
  const config=it.config;
  for(const epoch of it.student.epochs){
    for(let s=0;s<12;s++)for(let a=0;a<4;a++){
      const expected=sum(it.steps.map((step,t)=>step.state===s?epoch.anchors[t].gradient[a]/it.steps.length:0));
      close(epoch.thetaGradient[s][config.command][a],expected);
      close(epoch.after.theta[s][config.command][a],epoch.before.theta[s][config.command][a]-config.actorLR*expected);
    }
    it.steps.forEach((step,t)=>{
      vector(step.policy,softmax(it.before.theta[step.state][config.command]));
      close(epoch.anchors[t].ratio,epoch.anchors[t].policy[step.action]/step.policy[step.action]);
    });
  }
  for(let i=1;i<it.student.epochs.length;i++)assert.deepEqual(it.student.epochs[i].before,it.student.epochs[i-1].after);
});

test('iterations carry parameters exactly; history is immutable; zero goals have a well-defined no-op update',()=>{
  const result=runExperiment({},5);
  for(let i=1;i<result.iterations.length;i++)assert.deepEqual(result.iterations[i].before,result.iterations[i-1].after);
  close(result.iterations.at(-1).beta,0);
  assert.notDeepEqual(result.iterations[1].steps[0].policy,result.iterations[0].steps[0].policy);
  const first=JSON.stringify(result.iterations[0]);
  runIteration(result.iterations[0].after,result.config,1);
  assert.equal(JSON.stringify(result.iterations[0]),first);
  const empty=runExperiment({enabled:[false,false,false,false,false]}).iterations[0];
  assert.equal(empty.config.command,-1); assert.equal(empty.student.epochs.length,0);
  close(empty.critic.loss,0); assert.deepEqual(empty.after,empty.before);
  empty.critic.anchors.forEach(a=>vector(a.forward.probabilities,[0,0,0,0,0,1]));
  assert.equal(normalizeConfig(null).seed,1724);
  assert.equal(normalizeConfig({horizon:999,seed:Infinity}).horizon,10);
});
