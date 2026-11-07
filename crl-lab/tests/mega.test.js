import {test} from 'node:test';
import assert from 'node:assert/strict';
import {rng,sum} from '../src/math.js';
import {normalize,entropy,kl,omega,inspectPool,minimumDensityLaw,selectionLab,outcomeAnalysis,
  conditionalPreset,normalizeLab,METHODS,createModel,trainEpisode,achievedGoal,transition} from '../src/mega/engine.js';

const close=(a,b,e=1e-9)=>assert.ok(Math.abs(a-b)<e,`${a} != ${b}`);
const vector=(a,b,e)=>{assert.equal(a.length,b.length);a.forEach((v,i)=>close(v,b[i],e));};

test('selection is over achieved candidates, not zero-density unseen goals; filtering happens after generation',()=>{
  const lab=selectionLab();assert.equal(lab.winner,2);assert.ok(lab.candidateGoals.every(g=>g<3));
  assert.equal(lab.mix.alpha,0);
  const counts=[80,18,2,0,1],pool=[0,0,1,2,4];
  const result=inspectPool(counts,pool,[-1,-3,-5,-12,-25],-10);
  assert.equal(result.winner,2);assert.equal(result.occurrences[4],0);
  assert.deepEqual(result.eligible,[0,0,1,2]);
  const missing=selectionLab({seed:17});assert.equal(missing.winner,1);assert.equal(missing.occurrences[2],0);
  assert.equal(selectionLab({counts:[0,0,0,0,0]}).winner,null);
  assert.equal(selectionLab({cutoff:0}).winner,null);
});

test('finite-pool Diverse corrects repeats rather than inventing exactly uniform probabilities',()=>{
  const counts=[80,18,2,0,0],pool=[...Array(80).fill(0),...Array(18).fill(1),2,2];
  const info=inspectPool(counts,pool);
  vector(info.achieved,[.8,.18,.02,0,0]);vector(info.diverse,[1/3,1/3,1/3,0,0]);
  const finite=inspectPool(counts,[0,1,2]);
  assert.ok(finite.diverse[2]>.8);assert.notEqual(finite.diverse[0],finite.diverse[2]);
});

test('minimum-density sampling law matches exact N=1 and Monte Carlo with ties and rejections',()=>{
  vector(minimumDensityLaw([80,18,2,0,0],1).probabilities,[.8,.18,.02,0,0]);
  const counts=[6,2,2,0,0],values=[-1,-1,-9,-1,-1],cutoff=-5,N=4;
  const expected=minimumDensityLaw(counts,N,values,cutoff);
  const random=rng(713);const observed=[0,0,0,0,0];let failure=0;
  const p=normalize(counts),repetitions=30000;
  for(let i=0;i<repetitions;i++){
    const sample=Array.from({length:N},()=>{const u=random();return u<p[0]?0:u<p[0]+p[1]?1:2;});
    const g=inspectPool(counts,sample,values,cutoff).winner;if(g===null)failure++;else observed[g]++;
  }
  vector(observed.map(x=>x/repetitions),expected.probabilities,.008);
  close(failure/repetitions,expected.failure,.003);
  close(expected.failure,.2**N);
  vector(minimumDensityLaw([1,1,1,0,0],5).probabilities,[1/3,1/3,1/3,0,0]);
});

test('OMEGA handles disjoint support, finite first contact and bias saturation; objective mixture differs from sampler',()=>{
  const desired=[0,0,0,0,1];
  assert.equal(kl(desired,[.8,.18,.02,0,0]),Infinity);
  const disconnected=omega([80,18,2,0,0],desired,-3);
  close(disconnected.alpha,0);vector(disconnected.mixture,[1/3,1/3,1/3,0,0]);
  const contact=omega([800,180,20,0,1],desired,-3);
  close(contact.divergence,Math.log(1001));close(contact.alpha,1/(Math.log(1001)-3));
  const covered=omega([15,10,10,15,50],desired,-3);close(covered.alpha,1);
  const conservative=omega([15,10,10,15,50],desired,1);close(conservative.alpha,1/(1+Math.log(2)));
  const lab=selectionLab();assert.notDeepEqual(lab.probabilities.OMEGA,lab.mix.mixture);
  assert.equal(omega([0,0,0,0,0],desired).valid,false);
  assert.equal(omega([1,2,3,4,5],[0,0,0,0,0]).valid,false);
});

test('Eq.5 scores recover actual normalized finite-buffer entropy and oracle exposes bad achievability',()=>{
  const counts=[80,18,2,0,0],q=conditionalPreset('frontier'),a=outcomeAnalysis(counts,q);
  close(entropy([1,0,0,0,0]),0);close(entropy([.2,.2,.2,.2,.2]),Math.log(5));
  for(let g=0;g<5;g++){
    const candidate=a.candidates[g];
    const expected=(a.current+candidate.score)/(1+a.eta)+Math.log(1+a.eta)-a.current;
    close(candidate.gain,expected);
  }
  const perfect=outcomeAnalysis(counts,conditionalPreset());assert.ok(perfect.candidates[2].gain>perfect.candidates[1].gain);
  const broken=outcomeAnalysis(counts,conditionalPreset('broken'));assert.ok(broken.candidates[1].gain>broken.candidates[2].gain);
  assert.ok(a.outcomes[3].delta>a.outcomes[2].delta); // unseen outcome, not an available behavioural candidate
  const large=outcomeAnalysis([8e7,1.8e7,2e6,1e6,1e6],q);
  for(const candidate of large.candidates){
    const derivative=-sum(candidate.q.map((v,g)=>v*Math.log(large.p[g])))-large.current;
    close(candidate.gain*large.size,derivative,1e-6);
  }
  assert.equal(outcomeAnalysis([0,0,0,0,0],q),null);
});

test('actual agents share warmup and budgets; Q/HER logs obey real transitions and TD updates',()=>{
  const models=METHODS.map(m=>createModel(m));
  for(const m of models){assert.deepEqual(m.counts,models[0].counts);assert.deepEqual(m.q,models[0].q);}
  const mega=models[3],initial=mega.q.slice();
  for(let i=0;i<100;i++)for(const m of models)trainEpisode(m);
  assert.notDeepEqual(mega.q,initial);assert.ok(mega.firstDesired>0&&mega.firstDesired<=100);
  for(const m of models){
    const config=m.config;
    assert.equal(m.totalSteps,6+100*config.horizon);assert.equal(m.replay.length,m.totalSteps);
    assert.equal(sum(m.counts),1+m.totalSteps+100);
    assert.ok(m.q.every(x=>Number.isFinite(x)&&x<=1e-10&&x>=-1/(1-config.gamma)-1e-8));
    for(const record of m.history){
      vector(record.pBefore,normalize(record.before));
      assert.equal(sum(record.after)-sum(record.before),config.horizon+1);
      for(const s of record.steps){assert.equal(s.next,transition(s.state,s.action,config));assert.equal(s.reward,achievedGoal(s.next,config)===record.goal?0:-1);}
      for(const u of record.updates){
        close(u.target,u.reward+config.gamma*u.bootstrap);
        close(u.after,u.before+config.learningRate*(u.target-u.before));
        if(u.relabel)assert.equal(u.goal,achievedGoal(u.futureNext,config));
        if(u.reward===0)close(u.bootstrap,0);
      }
    }
  }
  const again=createModel('MEGA');for(let i=0;i<100;i++)trainEpisode(again);
  assert.deepEqual(again.history,mega.history);assert.deepEqual(again.q,mega.q);
});

test('blocked E remains unvisited; zero noise bonus is respected; malformed shared parameters are bounded',()=>{
  const model=createModel('OMEGA',{blocked:true,edge:1});
  for(let i=0;i<30;i++)trainEpisode(model);
  assert.equal(model.counts[4],0);assert.equal(model.firstDesired,null);
  const noBonus=createModel('MEGA',{bonus:0,epsilon:.1});
  trainEpisode(noBonus).steps.forEach(s=>close(s.epsilon,.1));
  const lab=normalizeLab({counts:[-1,Infinity,2,0,0],seed:1e99,candidates:-5,conditional:null});
  assert.deepEqual(lab.counts,[80,18,2,0,0]);assert.equal(lab.seed,999999);assert.equal(lab.candidates,1);
  assert.equal(normalizeLab(null).seed,2020);
});
