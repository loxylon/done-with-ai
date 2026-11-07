import {GOALS,COLORS,METHODS,normalize,entropy,omega,selectionLab,outcomeAnalysis,achievedGoal} from './engine.js';
import {sum} from '../math.js';

export const PAGES={
  problem:{eyebrow:'01 / ICML 2020 · КАКУЮ ПРОБЛЕМУ РЕШАЕТ СТАТЬЯ',title:'До далёкой цели<br>нужно <span>дойти.</span>',description:'HER умеет учиться на том, что агент уже увидел. Но где взять опыт, если нужная цель далеко? MEGA меняет поведенческие цели, чтобы постепенно расширять область достижимого.'},
  selector:{eyebrow:'02 / ALGORITHM 2 · ВЫБОР ПОВЕДЕНЧЕСКОЙ ЦЕЛИ',title:'Выбрать край.<br><span>Исследовать дальше.</span>',description:'Сэмплируем уже встречавшиеся цели, отбрасываем недостижимые и выбираем самую редкую среди оставшихся кандидатов. Измените буфер — и проследите каждый этап.'},
  environment:{eyebrow:'03 / ПЯТЬ ЦЕЛЕЙ · НАСТОЯЩЕЕ МАЛЕНЬКОЕ ОБУЧЕНИЕ',title:'Сначала вернуться.<br><span>Затем пойти дальше.</span>',description:'Агент возвращается к выбранной цели и усиливает случайное исследование, когда достигает её. Посмотрите, как это меняет траектории и буфер по сравнению с другими SELECT-правилами.'},
  entropy:{eyebrow:'04 / PROPOSITIONS 1–3 · ОТКУДА ВЗЯЛСЯ МИНИМУМ ПЛОТНОСТИ',title:'Редкость — подсказка.<br><span>Не гарантия.</span>',description:'Точная цель — увеличить энтропию исторических достигнутых целей. Минимальная плотность — удобная эвристика, когда агент действительно умеет добраться до выбранной цели.'},
  omega:{eyebrow:'05 / ФОРМУЛА 4 · ВОЗВРАЩАЕМСЯ К ВНЕШНЕЙ ЗАДАЧЕ',title:'Исследование — сначала.<br><span>Нужная цель — потом.</span>',description:'MEGA может продолжать исследовать и после открытия нужной области. OMEGA постепенно возвращает агенту внешнюю задачу, когда исторический буфер начинает её покрывать.'},
};
export const esc=x=>String(x).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
export const fmt=(x,d=4)=>x===null?'—':Number.isFinite(x)?x.toLocaleString('ru-RU',{maximumFractionDigits:d}):'∞';
const numeric=x=>`<span title="${x}">${fmt(x)}</span>`;
const pct=x=>`${fmt(100*x,1)}%`;
const texnum=x=>Number.isFinite(x)?Number(x.toFixed(5)).toString():'\\infty';
const goal=g=>`<span class="mgoal" style="background:${COLORS[g]}">${GOALS[g]}</span>`;
const text=x=>`<p class="mtext">${x}</p>`;
const callout=(x,color='')=>`<div class="mcallout ${color}">${x}</div>`;
const formula=(label,tex)=>`<div class="mformula"><div class="mkicker">${label}</div><div data-tex="${esc(tex)}"></div></div>`;
const card=(title,body,kicker='')=>`<section class="mcard">${kicker?`<div class="mkicker">${kicker}</div>`:''}<h2>${title}</h2>${body}</section>`;
const table=(headers,rows,classes=[])=>`<div class="mtable-wrap"><table class="mtable"><thead><tr>${headers.map(h=>`<th>${h}</th>`).join('')}</tr></thead><tbody>${rows.map((row,i)=>`<tr class="${classes[i]||''}">${row.map(cell=>`<td>${cell}</td>`).join('')}</tr>`).join('')}</tbody></table></div>`;
const stats=rows=>`<div class="mstats">${rows.map(([label,value,color=''])=>`<div class="mstat ${color}"><small>${label}</small><strong>${value}</strong></div>`).join('')}</div>`;
const detail=(label,body,open=false)=>`<details class="mdetail"${open?' open':''}><summary>${label}</summary>${body}</details>`;
const button=(label,action,extra='')=>`<button class="mbtn" data-action="${action}" ${extra}>${label}</button>`;
const input=(name,value,min,max,step=1,label=name)=>`<input class="mnumber" aria-label="${label}" data-lab="${name}" type="number" min="${min}" max="${max}" step="${step}" value="${value}">`;
const bars=(p,label='')=>`${label?`<div class="mkicker">${label}</div>`:''}<div class="mbars">${GOALS.map((_,g)=>`<div class="mbar-row">${goal(g)}<div class="mbar-track"><i style="width:${100*p[g]}%;background:${COLORS[g]}"></i></div><span class="mvalue">${pct(p[g])}</span></div>`).join('')}</div>`;
const presetButtons=()=>`<div class="mactions">${[['frontier','Граница'],['rare','Редкая, но недостижимая'],['empty','Пустой буфер'],['balanced','Все цели известны']].map(([id,label])=>button(label,'preset',`data-preset="${id}"`)).join('')}</div>`;

export function map({counts,edge=3,state=0,next=state,target=null,blocked=false}) {
  const last=edge*4,cell=440/(last+1),x=s=>30+(s+.5)*cell;
  return `<svg class="mmap" viewBox="0 0 500 155" role="img" aria-label="Коридор с пятью целями A B C D E. Агент в s=${state}, следующий s=${next}${target===null?'':', поведенческая цель '+GOALS[target]}">
    <defs><marker id="mega-arrow" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="4" markerHeight="4" orient="auto"><path d="M0 0L10 5L0 10z" fill="#d47551"/></marker></defs>
    ${GOALS.map((_,g)=>`<text x="${x(g*edge)}" y="23" text-anchor="middle" fill="${COLORS[g]}" font-size="12" font-weight="700">${GOALS[g]}</text><text x="${x(g*edge)}" y="129" text-anchor="middle" fill="#8e9b80" font-size="9">${counts[g]} виз.</text>`).join('')}
    ${Array.from({length:last+1},(_,s)=>{const g=Math.min(4,Math.floor(s/edge));return `<rect x="${30+s*cell+1}" y="42" width="${cell-2}" height="50" rx="5" fill="${counts[g]>0?COLORS[g]:'#eceee5'}" fill-opacity="${counts[g]>0?.18:1}" stroke="${g===target?COLORS[g]:'#d7dfcc'}" stroke-width="${g===target?2:1}"/><text x="${x(s)}" y="104" text-anchor="middle" fill="#9ca98d" font-size="7">${s}</text>`;}).join('')}
    ${blocked?`<path d="M${30+last*cell} 37v61" stroke="#b28163" stroke-width="4"/><text x="${x(last)}" y="72" text-anchor="middle" fill="#b28163" font-size="17">×</text>`:''}
    ${state!==next?`<path d="M${x(state)} 67H${x(next)}" stroke="#d47551" stroke-width="3" fill="none" marker-end="url(#mega-arrow)"/>`:''}
    <circle cx="${x(state)}" cy="67" r="11" fill="#45694e" stroke="#fff" stroke-width="2"/>
    <path d="M-3-4L4 0L-3 4z" fill="white" transform="translate(${x(state)} 67) ${next<state?'rotate(180)':''}"/>
    ${next!==state?`<circle cx="${x(next)}" cy="67" r="13" fill="none" stroke="#d47551" stroke-width="2" stroke-dasharray="3 2"/>`:''}
    <text x="30" y="150" fill="#859577" font-size="8">Зелёный — агент до шага · оранжевый — после · рамка — поведенческая цель</text>
  </svg>`;
}

function labEditor(lab) {
  const p=normalize(lab.counts);
  return table(['Цель','Наблюдений n','pAG = n / |B|','Оценка V(s₀,g)','Фильтр'],GOALS.map((_,g)=>[
    goal(g),input('count-'+g,lab.counts[g],0,50000,1,'Наблюдений '+GOALS[g]),numeric(p[g]),
    input('value-'+g,lab.values[g],-30,0,.5,'Оценка достижимости '+GOALS[g]),lab.counts[g]===0?'Не была в буфере':lab.values[g]>=lab.cutoff?'Проходит':'Ниже cutoff',
  ]));
}

export function problem({lab}) {
  const p=normalize(lab.counts), desired=normalize(lab.desired), mix=omega(lab.counts,desired,lab.bias);
  return card('Что агент уже видел — и что от него требуется?',map({counts:lab.counts})
    +`<div class="mgrid equal"><div>${bars(p,'pAG · исторически достигнутые цели')}</div><div>${bars(desired,'pDG · что потребуется на тесте')}</div></div>`
    +callout('Если E ещё не встречалась, HER не может придумать переход в E. Он может переименовать цель у существующего перехода, но не создать новое движение среды. Нужен способ собрать другие траектории.')
    +formula('ФОРМУЛА 1 · ИСХОДНАЯ ЗАДАЧА СОПОСТАВЛЕНИЯ РАСПРЕДЕЛЕНИЙ',String.raw`J_{\rm original}=D_{\rm KL}(p_{\rm dg}\Vert p_{\rm ag})=${texnum(mix.divergence)}`)
    +text('Когда pDG(E) &gt; 0, но pAG(E) = 0, forward KL бесконечна. MEGA сначала расширяет поддержку достигнутых целей; OMEGA затем возвращается к желаемому распределению.')
    +`<a class="mbtn primary" href="#selector">Разобрать выбор следующей цели</a>`,'ПОСТАНОВКА ЗАДАЧИ')
    +card('Что именно меняет статья',`<div class="mpipeline"><div><b>1 · SELECT</b>Куда хотим вернуться? MEGA выбирает редкое достигнутое.</div><div><b>2 · ROLLOUT</b>Политика идёт к цели; после достижения усиливаем исследование.</div><div><b>3 · REPLAY</b>Все посещения меняют историческое pAG.</div><div><b>4 · RELABEL</b>HER переименовывает цели в сохранённом опыте.</div><div><b>5 · OPTIMIZE</b>Обычное off-policy обучение goal-conditioned Q.</div></div>`
      +callout('Это правило выбора поведенческих целей. MEGA не заменяет HER и не добавляет просто «награду за редкость». SELECT и RELABEL решают разные задачи.'))
    +card('Чем это полезнее предыдущих правил',table(['Подход','Что выбирает','Проблема при длинном горизонте'],[
      ['HER / Desired','Только внешнюю E','Может долго не собирать нужный опыт'],
      ['Achieved · RIG-like','Часто достигавшиеся цели из буфера','Перевес уже освоенной области'],
      ['Diverse · Skew-Fit / DISCERN-like','Примерно равномерно по известным целям','Много времени во внутренней области поддержки'],
      ['MEGA','Минимум density среди N достигнутых кандидатов','Больше усилий на редко посещённой границе'],
      ['OMEGA','MEGA, затем всё чаще внешнюю цель','Позволяет прекратить ненужное исследование'],
    ])+text('В статье implemented baselines используют общий DDPG+HER и одну буферную генерацию кандидатов. Сравнивается SELECT, а не полные оригинальные архитектуры RIG / Skew-Fit.'))
    +card('Где преимущество подтверждено, а где оно не обещано',text('В экспериментах статьи O/MEGA существенно ускоряют решение PointMaze, AntMaze, FetchPickAndPlace и FetchStack2. На AntMaze авторы сообщают примерно 10× относительно hierarchical PPO+SR, на PointMaze — почти 100× относительно PPO+SR (§4, Fig. 3–6). Эти методы имеют разные базовые алгоритмы; весь разрыв нельзя приписать только SELECT.')
      +callout('MEGA не всегда лучше: ошибочная достижимость, плохая модель плотности, неудачный goal space или слишком короткий rollout могут мешать. При малом числе целей выигрыш может быть небольшим. На странице среды сравнение честно показывает результат выбранного seed, включая неудачи.','orange'));
}

export function selector({lab,method}) {
  const info=selectionLab(lab), counts=lab.counts, p=info.p;
  const occurrences=GOALS.map((_,g)=>info.candidateGoals.filter(i=>i===g).length);
  return presetButtons()+`<div class="mgrid"><div>`
    +card('1. Откуда берутся кандидаты',text('Каждый элемент буфера — одно прошлое наблюдение AG(s). Часто встречавшиеся цели чаще попадают в первоначальную выборку. Неизвестные D и E не могут стать поведенческой целью MEGA только потому, что их density равна 0.')
      +labEditor(lab)+map({counts,target:info.winner})
      +formula('СНАЧАЛА ОЦЕНИВАЕМ ИСТОРИЧЕСКУЮ ПЛОТНОСТЬ',String.raw`p_{\rm ag}(g)=\frac{n_g}{|B|},\qquad |B|=${sum(counts)}`)
      +detail('Гистограмма здесь и KDE в статье',text('Пять дискретных целей позволяют точно посчитать частоты. В статье цели непрерывные; KDE с Gaussian kernel и bandwidth 0.1 обучается на нормализованных samples. Нулевые частоты этой страницы — свойство дискретного примера, а не утверждение, что Gaussian KDE имеет нулевую плотность вне буфера.')))
    +card('2. Сэмплируем → фильтруем → выбираем',`<div class="mfields"><label class="mlabel">Кандидатов N${input('candidates',lab.candidates,1,100,1,'Кандидатов N')}</label><label class="mlabel">Seed выборки${input('seed',lab.seed,0,999999,1,'Seed выборки')}</label><label class="mlabel">Cutoff${input('cutoff',lab.cutoff,-30,0,.5,'Cutoff')}</label></div>`
      +button('Другая выборка с тем же буфером','resample')
      +table(['Цель','Выпало / N','После фильтра','Density','Минимум?'],GOALS.map((_,g)=>[goal(g),occurrences[g],info.occurrences[g],numeric(p[g]),info.winner===g?'Выбрана':occurrences[g]>0&&info.occurrences[g]===0?'Отвергнута':'']),GOALS.map((_,g)=>g===info.winner?'selected':lab.values[g]<lab.cutoff?'rejected':''))
      +formula('ALGORITHM 2 · MEGA SELECT',String.raw`\{g_i\}_{i=1}^{${lab.candidates}}\sim B,\qquad\hat g=\arg\min_{g_i\ \rm eligible}\hat p_{\rm ag}(g_i)`)
      +(sum(counts)===0?callout('Буфер пуст: Algorithm 2 пока не из чего сэмплировать. Нужен случайный разогрев. В среде он выполняется перед обучением.','orange'):info.winner===null?callout('Все выпавшие кандидаты отвергнуты. Выбор отсутствует: ослабьте cutoff или исправьте оценку достижимости.','orange'):callout(`В этой выборке MEGA выбрал <b>${GOALS[info.winner]}</b> с density ${fmt(p[info.winner])}. ${occurrences[2]===0&&counts[2]>0?'Цель C существует, но на этот раз не попала в pool. Минимум берётся по кандидатам, а не по всему пространству.':''}`))
      +detail('Как работает динамический cutoff в статье',text('В supplement C cutoff начинается с −3. Если intrinsic success за последние 10 эпизодов &gt;70%, cutoff уменьшают на 1 (с ограничением по минимальному Q кандидатов); при &lt;30% повышают на 1. Более отрицательные значения допускают трудные цели. В таблице V(s₀,g) — редактируемая учебная оценка goal-conditioned Q, не дополнительный reward.'))) +`</div><div>`
    +card('Один pool. Разные SELECT-правила.',`<div class="mactions">${METHODS.map(m=>button(m,'method',`data-method="${m}" aria-pressed="${m===method}"`)).join('')}</div>`+bars(info.probabilities[method],`${method} · вероятности при данном pool`)
      +text(method==='Diverse'?'Каждый оставшийся кандидат имеет вес 1/pAG. Повторы сохраняются: на конечном pool распределение не обязано быть точно равномерным.':method==='Achieved'?'Равномерно выбираем элемент pool. Повторения цели означают больший шанс; при большом pool это повторяет pAG.':method==='HER'?'Поведенческая цель берётся из pDG, даже если агент ещё не видел её. HER для обучения при этом остаётся доступен.':method==='OMEGA'?`Внешняя цель выбирается с вероятностью α = ${fmt(info.mix.alpha)}. Пока desired goal не покрыта, α = 0.`:'При фиксированном pool и фиксированном tie-break выбор MEGA детерминирован. Случайность находится в генерации кандидатов.')
      +text(`Сумма вероятностей = ${fmt(sum(info.probabilities[method]))}. Недостающая масса, если есть, означает отсутствие допустимого кандидата; это не дополнительная цель.`))
    +card('Почему N имеет значение',bars(info.law.probabilities,'MEGA · по многим новым выборкам')
      +formula('ШАНС ХОТЯ БЫ РАЗ УВИДЕТЬ C СРЕДИ N',String.raw`P(C\in\{g_i\})=1-(1-p_C)^N\approx${texnum(1-(1-p[2])**lab.candidates)}`)
      +text(`Если C — единственная наименее плотная допустимая цель, это и есть шанс выбрать её. При N=1 правило MEGA повторяет выборку буфера. В статье N=100. Вероятность не найти допустимый кандидат = ${pct(info.law.failure)}.`))
    +card('Две разные редкости',callout('Редкая, но уже достигавшаяся цель: кандидат для MEGA. Никогда не достигавшаяся цель: пока не кандидат. Редкая, но недостижимая теперь цель: её должен отсеять фильтр.','purple'))+`</div></div>`;
}

function trainingPlot(models) {
  const n=models.MEGA.history.length;
  if(!n)return text('Кривые появятся после первых эпизодов.');
  const width=620,height=185, x=i=>42+(i/(Math.max(1,n-1)))*550,y=h=>155-h/Math.log(5)*125;
  return `<svg class="mplot" viewBox="0 0 ${width} ${height}" role="img" aria-label="Энтропия исторического буфера по эпизодам у пяти методов"><path d="M42 25V155H600" fill="none" stroke="#d9e2cc"/>
    <path d="M42 ${y(Math.log(5))}H600" stroke="#dce5ce" stroke-dasharray="4 3"/><text x="0" y="${y(Math.log(5))+3}">ln 5</text><text x="15" y="158">0</text>
    ${METHODS.map((method,j)=>`<polyline points="${models[method].history.map((r,i)=>`${x(i)},${y(r.entropy)}`).join(' ')}" fill="none" stroke="${COLORS[j]}" stroke-width="${method==='MEGA'?2.8:1.8}"/><text x="${45+j*112}" y="180" style="fill:${COLORS[j]}">${method}</text>`).join('')}
    <text x="42" y="169">1</text><text x="560" y="169">эп. ${n}</text></svg>`;
}

export function environment(context) {
  const {models,train,trainMethod,episodeIndex,stepIndex,busy}=context;
  const model=models[trainMethod], history=model.history;
  const record=history[Math.min(episodeIndex,history.length-1)], step=record?.steps[Math.min(stepIndex,record.steps.length-1)];
  const count=models.MEGA.history.length;
  const controls=`<div class="mactions">${[1,10,100].map(n=>button(`+${n} эпизод${n===1?'':'ов'}`,'train',`data-episodes="${n}" ${busy||count>=300?'disabled':''}`)).join('')}${button('Начать заново','reset-training',busy?'disabled':'')}</div>`;
  const fields=[['seed','Seed',0,999999,1],['edge','Шагов между целями L',1,4,1],['horizon','Шагов в эпизоде',8,40,1],['updates','Q-updates / эпизод',20,300,10],['epsilon','Начальное ε',0,1,.05],['bonus','Прирост ε при успехе',0,.5,.05],['learningRate','Шаг Q-learning',.01,1,.01],['gamma','Дисконт γ',.5,.99,.01],['candidates','Кандидатов N',1,100,1],['bias','OMEGA bias b',-3,1,.1]];
  return card('Одна среда. Одна база обучения. Меняется только SELECT.',text('Пять целей — пять областей коридора. Агент начинает слева; внешняя задача — E. Каждый метод получает одинаковые шесть случайных шагов разогрева, отдельную Q-таблицу и одинаковый бюджет. Ни один метод не использует готовый маршрут.')
    +controls+`<details class="mdetail"><summary>Параметры среды и обучения</summary><form id="mega-train-form"><div class="mfields">${fields.map(([key,label,min,max,step])=>`<label class="mlabel">${label}<input name="${key}" type="number" min="${min}" max="${max}" step="${step}" value="${train[key]}" required></label>`).join('')}<label class="mlabel">Доступ к E<select name="blocked"><option value="false" ${!train.blocked?'selected':''}>Открыт</option><option value="true" ${train.blocked?'selected':''}>Перекрыт стеной</option></select></label></div><button class="mbtn primary" type="submit" ${busy?'disabled':''}>Применить и сбросить опыт</button></form></details>`
    +`<div class="mcompare">${METHODS.map((method,i)=>{const m=models[method],r=m.history.at(-1);return `<button class="mbtn" data-action="train-method" data-method="${method}" aria-pressed="${method===trainMethod}"><b style="color:${COLORS[i]}">${method}</b><strong>${m.counts.filter(n=>n>0).length}/5</strong><span>целей обнаружено</span><span>Первая E: ${m.firstDesired===null?'ещё нет':m.firstDesired===0?'разогрев':'эп. '+m.firstDesired}</span><span>Greedy тест: ${r?r.evaluation.success?'успех':'нет успеха':'ещё не оценён'}</span></button>`;}).join('')}</div>`
    +trainingPlot(models)+text(`Сейчас ${count} эпизодов на метод, по ${model.totalSteps} шагов среды, включая разогрев. На графике H[pAG], а не success rate и не результаты статьи. В конце OMEGA энтропия может снижаться, когда агент больше тренируется на E.`)
    +callout('Проверьте разные seeds и длину коридора. При маленьком горизонте, малом N или плохой политике MEGA может не помочь. Кривые здесь показывают реальные посещения выбранного учебного опыта.','purple'))
    +`<div class="mgrid"><div>`+card(`Разобрать траекторию · ${trainMethod}`,record?
      `<div class="mfields"><label class="mlabel">Эпизод<select data-episode-index>${history.map((r,i)=>`<option value="${i}" ${i===record.index?'selected':''}>${i+1} · цель ${GOALS[r.goal]}</option>`).join('')}</select></label><label class="mlabel">Шаг<select data-step-index>${record.steps.map((_,i)=>`<option value="${i}" ${i===step.t?'selected':''}>t = ${i}</option>`).join('')}</select></label></div>`
      +map({counts:record.before,edge:train.edge,state:step.state,next:step.next,target:record.goal,blocked:train.blocked})
      +stats([['Поведенческая цель',GOALS[record.goal]],['Состояние → следующее',`${step.state} → ${step.next}`],['Сейчас ε',fmt(step.epsilon)]])
      +`<div class="mactions">${button('Предыдущий шаг','previous-step',step.t===0?'disabled':'')}${button('Следующий шаг','next-step',step.t===record.steps.length-1?'disabled':'')}${button('Следующий эпизод','next-record',record.index===history.length-1?'disabled':'')}</div>`
      +table(['Действие','Q(s,a,g)','ε-greedy вероятность','Выбрано'],['Влево','Вправо','Стоять'].map((name,a)=>{const best=Math.max(...step.values),ties=step.values.filter(v=>v===best).length;return[name,numeric(step.values[a]),numeric(step.epsilon/3+(step.values[a]===best?(1-step.epsilon)/ties:0)),a===step.action?'Да':''];}))
      +callout(`u = ${fmt(step.u)} ${step.explore?'&lt;':'≥'} ε: ${step.explore?'случайное действие':'greedy действие (равные максимумы делят выбор)'}. AG(s′) = ${GOALS[step.achieved]}, reward = ${step.reward}. ${step.reward===0?`Цель достигнута: на следующем шаге ε увеличивается до ${fmt(Math.min(1,step.epsilon+train.bonus))}.`:'Пока выбранная цель не достигнута, ε не увеличивается.'}`)
      +formula('КАК УСТРОЕНА МАЛЕНЬКАЯ СРЕДА',String.raw`AG(s)=\min(4,\lfloor s/${train.edge}\rfloor),\quad R_g(s')=\mathbf1[AG(s')=g]-1`)
      +text('Параметры Q не меняются во время этого rollout. После него идут updates по replay. Goal region, а не геометрическая близость к E, определяет награду.')
      +detail('Все шаги этого эпизода',table(['t','s','a','s′','AG(s′)','r','ε','Исследование'],record.steps.map(s=>[s.t,s.state,['←','→','·'][s.action],s.next,GOALS[s.achieved],s.reward,numeric(s.epsilon),s.explore?'Да':'Нет'])))
      :text('Нажмите «+1 эпизод», чтобы увидеть настоящее движение и первые обновления. Можно сразу запустить +100, а затем вернуться к любому эпизоду.'))+`</div><div>`
    +card('Что изменилось в replay',bars(normalize(record?.after||model.counts),'pAG после выбранного эпизода')
      +(record?table(['Цель','До','Добавлено','После'],GOALS.map((_,g)=>[goal(g),record.before[g],record.after[g]-record.before[g],record.after[g]])):text('Пока здесь только одинаковый случайный разогрев.'))
      +button('Передать этот буфер в инспектор','copy-buffer')
      +text('Это независимый опыт: изменение гистограммы на вкладке выбора не подделывает данные обучения среды.'))
    +card('HER не выбирает маршрут. Он переименовывает опыт.',record?
      text(`Ниже шесть фактических updates из ${train.updates} после выбранного rollout. Replay может дать переход из более раннего эпизода. В 80% draws берётся цель из будущего этого же сохранённого эпизода; в 20% сохраняется исходная команда.`)
      +table(['s,a,s′','Исходная → новая g','Источник future','r','Q до','Bootstrap','TD target','Q после'],record.updates.map(u=>[`${u.state},${u.action},${u.next}`,`${GOALS[u.originalGoal]} → ${GOALS[u.goal]}`,u.relabel?`${u.episodeIndex<0?'разогрев':'эп.'+(u.episodeIndex+1)}, t=${u.futureT}, s′=${u.futureNext}`:'Real',u.reward,numeric(u.before),numeric(u.bootstrap),numeric(u.target),numeric(u.after)]))
      +formula('ОБЫЧНЫЙ TABULAR Q UPDATE · ОДИНАКОВ У ВСЕХ МЕТОДОВ',String.raw`y=r+\gamma\mathbf1[AG(s')\ne g]\max_{a'}Q(s',a',g),\quad Q\leftarrow Q+\eta(y-Q)`)
      +text('Успех имеет reward 0, остальные переходы −1, как в статье. Для простоты goal-terminal target обнуляет bootstrap при достижении цели. Новый маршрут рождается из новых наблюдений; relabeling не создаёт отсутствующих переходов.')
      :text('Первый batch обновлений появится после запуска эпизода.'))+`</div></div>`;
}

export function entropyPage({lab}) {
  const analysis=outcomeAnalysis(lab.counts,lab.conditional), info=selectionLab(lab),g=lab.focus;
  if(!analysis)return card('Сначала нужны наблюдения',callout('Буфер пуст: нет нормированного pAG и η = 1/|B| не определено. Добавьте данные в инспекторе или используйте исходный пример.','orange')+`<a class="mbtn" href="#selector">Открыть буфер</a>`);
  const eligible=GOALS.map((_,i)=>i).filter(i=>lab.counts[i]>0&&lab.values[i]>=lab.cutoff&&analysis.candidates[i].valid);
  const oracle=eligible.length?eligible.reduce((a,b)=>analysis.candidates[b].gain>analysis.candidates[a].gain?b:a):null;
  const a=analysis.candidates[g];
  return `<div class="mactions">${[['perfect','Агент достигает выбранного'],['broken','Редкая C ведёт обратно в A'],['frontier','Исследование открывает соседа']].map(([id,label])=>button(label,'conditional-preset',`data-preset="${id}"`)).join('')}</div>`
    +card('Цель MEGA — энтропия исторического буфера',formula('ФОРМУЛА 2',String.raw`J_{\rm MEGA}=-H[p_{\rm ag}],\qquad H[p]=-\sum_g p(g)\log p(g)`)
      +stats([['Буфер |B|',analysis.size],['Текущая H, nats',fmt(analysis.current)],['Максимум для пяти целей',fmt(Math.log(5))]])
      +table(['Цель','pAG','−p log p','H после +1 этой цели','Реальная ΔH','Score Eq.5'],GOALS.map((_,i)=>[goal(i),numeric(analysis.p[i]),numeric(analysis.p[i]===0?0:-analysis.p[i]*Math.log(analysis.p[i])),numeric(analysis.outcomes[i].after),numeric(analysis.outcomes[i].delta),numeric(analysis.outcomes[i].score)]))
      +callout('Прибавить наблюдение неизвестной D может увеличить H сильнее, чем повтор C. Но D ещё не умеем задавать как достижимую поведенческую цель. MEGA возвращается к C в надежде исследовать вокруг неё и открыть D.','purple'))
    +`<div class="mgrid"><div>`+card('Чего достигнет агент, если мы попросим '+GOALS[g]+'?',`<label class="mlabel">Поведенческая цель ĝ<select data-focus>${GOALS.map((name,i)=>`<option value="${i}" ${i===g?'selected':''}>${name}</option>`).join('')}</select></label>`
      +text('q(g′ | ĝ) — модель исходов. Здесь она задана вами для объяснения теории. Практический MEGA не знает эту таблицу и не использует её при выборе минимума плотности.')
      +`<div class="mfields five">${GOALS.map((name,i)=>`<label class="mlabel">Вес исхода ${name}<input class="mnumber" data-outcome="${i}" aria-label="Вес исхода ${name}" type="number" min="0" max="100" step=".1" value="${lab.conditional[g][i]}"></label>`).join('')}</div>`
      +bars(a.q,`Нормированные исходы q(g′ | ${GOALS[g]})`)
      +(a.valid?formula('ТОЧНАЯ ОЖИДАЕМАЯ ПРИБАВКА ПОСЛЕ ОДНОГО НАБЛЮДЕНИЯ',String.raw`\mathbb E[\Delta H\mid\hat g]=\sum_{g'}q(g'\mid\hat g)\left[H\left(\frac{n+e_{g'}}{|B|+1}\right)-H\left(\frac n{|B|}\right)\right]\approx ${texnum(a.gain)}`):callout('Все веса q равны 0: это не распределение. Задайте хотя бы один положительный исход.','orange'))
      +button('Добавить один исход по этой q','add-outcome',a.valid?'':'disabled'))+`</div><div>`
    +card('Oracle против эвристики',table(['Команда ĝ','E[ΔH]','Допустима?'],GOALS.map((name,i)=>[name,analysis.candidates[i].valid?numeric(analysis.candidates[i].gain):'q не задана',eligible.includes(i)?'Да':'Нет']),GOALS.map((_,i)=>i===oracle?'selected':'')))
      +stats([['Oracle · знает q',oracle===null?'—':GOALS[oracle]],['MEGA · текущий pool',info.winner===null?'—':GOALS[info.winner]],['|B| → |B| + 1',`${analysis.size} → ${analysis.size+1}`]])
      +callout(oracle!==info.winner?'Ответы различаются. Минимум density не гарантирует максимум реального прироста, когда поведение по редкой цели даёт другие исходы или когда oracle-цель не попала в pool.':'В этом примере ответы совпали. При надёжном достижении выбранной цели редкий исход даёт наибольшую прибавку энтропии.')+`</div></div>`
    +card('Почему minimum density — разумное приближение',formula('PROPOSITION 3 · ПРИ q(g′|ĝ) = 1[g′ = ĝ]',String.raw`\arg\max_{\hat g\in B}\{-\log p_{\rm ag}(\hat g)\}=\arg\min_{\hat g\in B}p_{\rm ag}(\hat g)`)
      +text('В пределе большого буфера один дополнительный исход даёт направленную производную энтропии. Для известной цели с p&gt;0 её приоритет равен −log p. Если выбранная редкая цель фактически не достигается, эта подстановка неверна — поэтому нужны достижимость и исследование после возврата.')
      +detail('Eq.6: что бы делал oracle, знающий все исходы?',formula('ОБЩИЙ ПРИОРИТЕТ ПРИ МАЛОМ η',String.raw`\hat g^*=\arg\max_{\hat g\in B}\left[D_{\rm KL}(q(g'\mid\hat g)\Vert p_{\rm ag})+H[q(g'\mid\hat g)]\right]`)
        +text('Oracle предпочитает разнообразные исходы, отличающиеся от истории. Если q — точечная масса на выбранной цели, H[q]=0, а KL превращается в −log pAG(ĝ). Для исхода с pAG=0 derivative бесконечна; точная finite-buffer таблица выше остаётся корректной. Энтропия здесь относится к историческим achieved goals, а не к распределению действий policy.'))
      +detail('Eq.5: score и реальная ΔH — не одно и то же',formula('SCORE С ОТБРОШЕННЫМИ ОБЩИМИ КОНСТАНТАМИ',String.raw`\eta=1/|B|,\quad S(g')=p(g')\log p(g')-(p(g')+\eta)\log(p(g')+\eta)`)
        +text('В Eq.5 score сохраняет ranking ожидаемой новой энтропии, но сам по себе не равен её приросту: в точном обновлении есть нормировка 1+η и общие константы. Таблица выше показывает оба числа. При p=0 считаем 0 log 0 = 0; новый finite-buffer H определён. Производная −log p в нуле бесконечна, поэтому теория предела требует аккуратности.')));
}

function alphaPlot(mix,bias) {
  const points=Array.from({length:101},(_,i)=>{const d=i*.12;return `${35+d/12*480},${140-(1/Math.max(bias+d,1))*110}`;}).join(' ');
  const d=mix.divergence;
  return `<svg class="mplot" viewBox="0 0 550 175" role="img" aria-label="OMEGA alpha в зависимости от KL divergence"><path d="M35 20V140H525" stroke="#d8e2cc" fill="none"/><polyline points="${points}" fill="none" stroke="#658d75" stroke-width="2.5"/>
    <text x="10" y="33">1</text><text x="10" y="143">0</text><text x="35" y="160">KL = 0</text><text x="470" y="160">KL = 12</text>
    ${Number.isFinite(d)?`<circle cx="${35+Math.min(d,12)/12*480}" cy="${140-mix.alpha*110}" r="5" fill="#cf805b"/><text x="210" y="20">α = ${fmt(mix.alpha)}</text>`:'<text x="140" y="20">Сейчас KL = ∞, α = 0</text>'}</svg>`;
}

export function omegaPage({lab}) {
  const desired=normalize(lab.desired),info=selectionLab(lab),mix=info.mix;
  return `<div class="mactions">${[['frontier','Цель ещё не встречалась'],['contact','Первый контакт с E'],['covered','Много данных об E']].map(([id,label])=>button(label,'preset',`data-preset="${id}"`)).join('')}</div>`
    +`<div class="mgrid"><div>`+card('Покрыта ли желаемая цель?',`<div class="mfields five">${GOALS.map((name,g)=>`<label class="mlabel">Вес pDG(${name})${input('desired-'+g,lab.desired[g],0,100,.1,'Вес desired '+name)}</label>`).join('')}</div>`
      +labEditor(lab)
      +table(['g','pDG','pAG','pDG log(pDG / pAG)'],GOALS.map((name,g)=>[name,numeric(desired[g]),numeric(mix.p[g]),desired[g]===0?'0':mix.p[g]===0?'∞':numeric(desired[g]*Math.log(desired[g]/mix.p[g]))]))
      +formula('ФОРМУЛА 1 · FORWARD KL',String.raw`D_{\rm KL}(p_{\rm dg}\Vert p_{\rm ag})=\sum_g p_{\rm dg}(g)\log\frac{p_{\rm dg}(g)}{p_{\rm ag}(g)}=${texnum(mix.divergence)}`)
      +(!mix.valid?callout('Нужны непустой буфер и хотя бы один положительный вес pDG. Без них objective не определён.','orange'):callout(Number.isFinite(mix.divergence)?'Вся поддержка pDG встречалась в буфере. Теперь KL конечна, и OMEGA может увеличивать долю внешних целей.':'Есть желаемая цель с pAG=0. KL бесконечна; α=0. OMEGA пока полностью использует MEGA.')))+`</div><div>`
    +card('В какой момент переключаться?',`<label class="mlabel">Bias b = ${fmt(lab.bias)}<input type="range" data-lab="bias" min="-3" max="1" step=".1" value="${lab.bias}"></label>`
      +formula('ALGORITHM 2 · ВЕРОЯТНОСТЬ ВНЕШНЕЙ ЦЕЛИ',String.raw`\alpha=\frac1{\max(b+D_{\rm KL},1)}=\frac1{\max(${texnum(lab.bias)}+${texnum(mix.divergence)},1)}=${texnum(mix.alpha)}`)
      +stats([['MEGA ветка',pct(1-mix.alpha)],['Внешняя ветка',pct(mix.alpha)],['Bias статьи', '−3']])
      +alphaPlot(mix,lab.bias)
      +text('b=−3 позволяет получить α=1 уже при KL≤4, а не только при равенстве pAG и pDG. Это осознанный bias: исторический буфер переоценивает стартовые области. α — не доля успехов и не оценка вероятности достижимости.')
      +callout(`Текущее случайное x = ${fmt(info.u)}. Проверка x &lt; α ${info.external?'истинна: выбираем внешнюю цель':'ложна: выбираем MEGA'}.`, 'purple'))+`</div></div>`
    +card('Objective и поведенческий sampler — разные распределения',`<div class="mgrid equal"><div>${bars(mix.mixture,'pα · целевое распределение в objective')}${formula('ФОРМУЛА 4',String.raw`p_\alpha=\alpha p_{\rm dg}+(1-\alpha)\mathcal U(\operatorname{supp}p_{\rm ag})`)}</div><div>${bars(info.probabilities.OMEGA,'Реальный выбор OMEGA при текущем pool')}${text('В практическом Algorithm 2 с вероятностью 1−α выбирается минимум density, а не равномерная цель. Поэтому эти два столбца могут различаться даже при α=0.')}</div></div>`
      +formula('OMEGA OBJECTIVE',String.raw`J_{\rm OMEGA}=D_{\rm KL}(p_\alpha\Vert p_{\rm ag})=${mix.objective===null?String.raw`\text{не определён}`:texnum(mix.objective)}`)
      +detail('Как Eq.2 переходит в Eq.3 и Eq.4?',formula('ФОРМУЛА 3 · ПРИБЛИЖЕНИЕ РАВНОМЕРНЫМ TARGET',String.raw`\widetilde J_{\rm MEGA}=D_{\rm KL}(\mathcal U(\operatorname{supp}p_{\rm ag})\Vert p_{\rm ag})`)
        +text('При фиксированной поддержке и −H[pAG], и это KL имеют оптимум в равномерном распределении. Но это разные функции, не равные на каждом шаге. Eq.4 смешивает равномерный target с pDG. Практический SELECT приближённо улучшает objective через MEGA / внешние цели; он не сэмплирует непосредственно из pα.'))
      +text('После открытия E MEGA может продолжать заполнять другие области. OMEGA усиливает обучение на нужных тестовых целях. В статье это важно для долгосрочной стабильности FetchStack2.'));
}

export function page(context) {
  return ({problem,selector,environment,entropy:entropyPage,omega:omegaPage})[context.route](context);
}
