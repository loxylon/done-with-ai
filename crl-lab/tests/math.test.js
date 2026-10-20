import { test } from 'node:test';
import assert from 'node:assert/strict';
import { labels, trajectory, targets, sum, rng, drawOffset, conditionalVariance,
  crossEntropy, criticGradient, classProbabilities, argmax, ppo } from '../src/math.js';
const close = (a,b,e=1e-10) => assert.ok(Math.abs(a-b)<e, `${a} != ${b}`);
const vector = (a,b) => { assert.equal(a.length,b.length);a.forEach((v,i)=>close(v,b[i])); };
const state = {live:2,frames:[[0,1],[2]],end:'death',reset:false};
const opt = {gamma:.9,term:.9,cap:8};

test('co-satisfied predicates, duplication, and empty semantic set',()=>{
 vector(labels([0,1]),[.5,.5,0]);vector(labels([0,1],true),[1/3,1/3,0,1/3]);vector(labels([]),[0,0,0]);
});
test('exact JAX fixture: death, Δ=[2,1,0], conserved mass',()=>{
 const t=trajectory(state);
 const a=targets(t,0,opt);vector([...a.y,a.absorbing],[.05,.05,.09,.81]);assert.equal(a.delta,2);
 const b=targets(t,1,opt);vector([...b.y,b.absorbing],[0,0,.1,.9]);
 const c=targets(t,2,opt);vector([...c.y,c.absorbing],[0,0,0,1]);assert.equal(c.delta,0);
 assert.equal(targets(t,2,{...opt,method:'classic'}).valid,false);
});
test('absorbing-only changes final action but not previous normalized goals',()=>{
 const t=trajectory(state);const a=targets(t,0,{...opt,method:'abs'});
 vector(a.y,[.5/1.9,.5/1.9,.9/1.9]);close(a.absorbing,0);
 close(targets(t,2,{...opt,method:'abs'}).absorbing,1);
});
test('success terminal includes immediate label; time-limit and window do not penalize',()=>{
 for(const end of ['success','timeout','window']){
  const t=trajectory({...state,live:1,frames:[[2]],end});const m=targets(t,0,opt);
  if(end==='success'){vector([...m.y,m.absorbing],[0,0,.1,.9]);assert.equal(m.delta,1);}
  else{vector([...m.y,m.absorbing],[0,0,1,0]);assert.equal(m.delta,Infinity);}
 }
});
test('a reset never leaks a new episode into the preceding future',()=>{
 const a=targets(trajectory(state),0,opt), b=targets(trajectory({...state,reset:true}),0,opt);
 vector([...a.y,a.absorbing],[...b.y,b.absorbing]);assert.equal(b.futures.length,2);
 const second=targets(trajectory({...state,reset:true}),3,opt);vector(second.y,[0,1,0]);
});
test('offset cap leaves terminal distance independent; semantic missing mass is residual',()=>{
 const a=targets(trajectory(state),0,{...opt,cap:1});assert.equal(a.delta,2);
 vector([...a.y,a.absorbing],[.095,.095,0,.81]);
 const m=targets(trajectory({...state,end:'window',frames:[[],[]]}),0,opt);close(m.absorbing,1);
 assert.equal(targets(trajectory({...state,end:'window',frames:[[],[]]}),0,{...opt,method:'classic'}).valid,false);
});
test('all anchors/methods/discounts/caps conserve valid target mass',()=>{
 for(const end of ['death','success','timeout','window'])for(const gamma of [.5,.9,.995])for(const term of [0,.8,.995])for(const cap of [1,2,8])for(const method of ['classic','abs','corrected']){
  const t=trajectory({...state,end,reset:true});
  for(let anchor=0;anchor<t.length;anchor++){
   const m=targets(t,anchor,{gamma,term,cap,method});
   assert.ok(m.y.every(x=>x>=0));assert.ok(m.absorbing>=0);
   if(m.valid)close(sum(m.y)+m.absorbing,1,1e-9);
  }
 }
});
test('sampled offsets converge to RB mean; RB removes only conditional offset variance',()=>{
 const m=targets(trajectory(state),0,opt), random=rng(2026);let total=0;
 for(let i=0;i<30000;i++)total+=m.futures[drawOffset(m.futures,random)].labels[2];
 close(total/30000,sum(m.futures.map(f=>f.weight*f.labels[2])),.008);
 assert.ok(conditionalVariance(m.futures,2)>.2);
 vector(Array.from({length:5},rng(4)),Array.from({length:5},rng(4)));
});
test('CE gradient matches finite differences, fixed ∅ logit, fatal lowers every score',()=>{
 const z=[.8,.2,-.3],y=[0,0,0,1],grad=criticGradient(z,y),h=1e-5;
 grad.forEach((g,i)=>{const plus=[...z],minus=[...z];plus[i]+=h;minus[i]-=h;close(g,(crossEntropy(plus,y)-crossEntropy(minus,y))/(2*h),1e-7);assert.ok(g>0);});
 close(sum(classProbabilities(z)),1);assert.ok(crossEntropy(z.map((x,i)=>x-.5*grad[i]),y)<crossEntropy(z,y));
});
test('ranking by logits and probability can genuinely disagree',()=>{
 const z=[[2,0,0],[3,5,5],[.8,.2,-.3]];
 assert.equal(argmax(z.map(row=>row[0])),1);assert.equal(argmax(z.map(row=>classProbabilities(row)[0])),0);
});
test('PPO analytic gradient matches finite differences with both advantage signs and β',()=>{
 for(const theta of [-3,-.3,1.2,3])for(const advantage of [-2,1])for(const teacher of [0,1,2]){
  const a={theta,old:.35,advantage,clip:.2,beta:.4,progress:25,teacher},h=1e-5;
  const m=ppo(a), numerical=(ppo({...a,theta:theta+h}).loss-ppo({...a,theta:theta-h}).loss)/(2*h);
  close(m.gradient,numerical,1e-6);
 }
});
test('clipping is asymmetric; β anneals exactly to zero',()=>{
 const a={theta:3,old:.2,advantage:1,clip:.2,beta:.4,progress:0,teacher:0};
 assert.equal(ppo(a).clippedActive,true);close(ppo(a).policyGradient,0);
 assert.equal(ppo({...a,advantage:-1}).clippedActive,false);
 const noTeacher=ppo({...a,progress:100});close(noTeacher.loss,noTeacher.policyLoss);close(noTeacher.gradient,noTeacher.policyGradient);
});
test('invalid discounts/offsets are rejected',()=>{
 assert.throws(()=>targets(trajectory(state),0,{...opt,gamma:1}),RangeError);
 assert.throws(()=>targets(trajectory(state),9,opt),RangeError);
});
