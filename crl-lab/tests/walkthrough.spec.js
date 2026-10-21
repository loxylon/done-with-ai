import { test, expect } from '@playwright/test';
import { readFile } from 'node:fs/promises';

const checkFormula = async page => {
  await expect(page.locator('#formula-panel')).toBeVisible();
  await expect(page.locator('.katex-error')).toHaveCount(0);
  expect(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth+1)).toBe(true);
};

test('walk through actions, all formula stages and actual updates across iterations',async({page})=>{
  const errors=[],external=[];
  page.on('pageerror',e=>errors.push(e.message));
  page.on('request',r=>{if(!r.url().startsWith('http://127.0.0.1:4173')&&!r.url().startsWith('data:'))external.push(r.url());});
  await page.goto('/');
  await page.getByRole('link',{name:'06 Пошаговая среда Клетка → формула → обновление'}).click();
  await expect(page).toHaveURL(/walkthrough\.html/);
  await expect(page.locator('h1')).toContainText('Каждый шаг');
  await expect(page.locator('[data-goal-toggle]')).toHaveCount(5);
  await expect(page.locator('.trace-map')).toHaveAttribute('aria-label',/Агент \(0, 1\).*следующая клетка \(0, 0\)/);
  await page.locator('[data-step="2"]').click();
  await page.locator('[data-stage="1"]').click();
  await expect(page.locator('#formula-panel')).toContainText('A, D, E');
  for(let step=0;step<5;step++){
    await page.locator(`[data-step="${step}"]`).click();
    for(let stage=0;stage<6;stage++){
      await page.locator(`[data-stage="${stage}"]`).click();await checkFormula(page);
      if(stage===4){for(let epoch=0;epoch<3;epoch++){await page.locator(`[data-epoch="${epoch}"]`).click();await checkFormula(page);}}
    }
  }
  await page.locator('[data-step="0"]').click();await page.locator('[data-stage="2"]').click();
  await expect(page.locator('.math-box').filter({hasText:'РАО–БЛЭКВЕЛЛ:'}).locator('[data-tex]')).toHaveAttribute('data-tex',/\\cdot/);
  await page.locator('[data-source-step="2"]').click();
  await expect(page.locator('.source-halo')).toHaveCount(1);
  await expect(page.locator('.trace-step.source')).toHaveText('t2');
  await page.locator('[data-step="5"]').click();
  await expect(page.locator('#formula-panel')).toContainText('φold');
  await page.locator('[data-command="next-step"]').click();
  await expect(page.locator('[data-testid="iteration-label"]')).toHaveText('Итерация 2 / 5');
  await expect(page.locator('#experiment-status')).toContainText('точно совпадают с итогом итерации 1');
  await expect(page.locator('#iteration-history tbody tr')).toHaveCount(2);
  await expect(page.locator('#formula-panel')).not.toContainText('0,25 ≤ u < 0,5');
  await page.locator('[data-stage="4"]').click();await checkFormula(page);
  for(let i=2;i<5;i++){
    await page.locator('[data-step]').last().click();await page.locator('[data-command="next-step"]').click();
    await expect(page.locator('[data-testid="iteration-label"]')).toHaveText(`Итерация ${i+1} / 5`);
  }
  await page.locator('[data-step]').last().click();
  await expect(page.locator('[data-command="next-step"]')).toBeDisabled();
  await expect(page.locator('[data-command="next-formula"]')).toBeDisabled();
  await page.locator('[data-stage="4"]').click();await checkFormula(page);
  await page.locator('[data-iteration="0"]').click();
  await expect(page.locator('[data-testid="iteration-label"]')).toHaveText('Итерация 1 / 5');
  expect(errors).toEqual([]);expect(external).toEqual([]);
});

test('goal switches, settings, sharing and exported numbers reproduce the same experiment',async({page})=>{
  const errors=[];page.on('pageerror',e=>errors.push(e.message));
  await page.goto('/walkthrough.html');
  for(let i=0;i<5;i++)await page.locator(`[data-goal-toggle="${i}"]`).uncheck();
  await page.locator('[data-stage="4"]').click();
  await expect(page.locator('#formula-panel')).toContainText('loss и все critic gradients равны 0');
  await checkFormula(page);
  await page.locator('[data-stage="3"]').click();await checkFormula(page);
  await page.locator('[data-goal-toggle="0"]').check();
  await page.locator('.trace-config summary').click();
  await page.locator('[name="danger"]').selectOption('false');
  await page.locator('[name="horizon"]').fill('4');
  await page.locator('[name="clip"]').fill('0.1');
  await page.getByRole('button',{name:'Применить и начать заново'}).click();
  await expect(page.locator('.trace-small-stats')).toContainText('Лимит');
  await expect(page.locator('[data-step]')).toHaveCount(5);
  await page.locator('[data-step="4"]').click();await page.locator('[data-command="next-step"]').click();
  await page.locator('[data-step="1"]').click();await page.locator('[data-stage="4"]').click();await page.locator('[data-epoch="2"]').click();
  await page.locator('#share-experiment').click();
  const url=page.url();expect(url).toContain('experiment=');
  const before=await page.locator('#formula-panel').innerText();
  await page.reload();await expect(page.locator('[data-testid="iteration-label"]')).toHaveText('Итерация 2 / 5');
  expect(await page.locator('#formula-panel').innerText()).toBe(before);
  const downloadPromise=page.waitForEvent('download');await page.locator('#export-experiment').click();
  const download=await downloadPromise;const data=JSON.parse(await readFile(await download.path(),'utf8'));
  expect(data.iterations).toHaveLength(2);
  expect(data.iterations[1].before).toEqual(data.iterations[0].after);
  expect(data.iterations[1].student.epochs).toHaveLength(3);
  expect(data.selected).toEqual({iteration:1,step:1,stage:4,epoch:2});
  expect(data.config.enabled).toEqual([true,false,false,false,false]);
  for(const suffix of ['?experiment=%7Bbad','?experiment='+encodeURIComponent(JSON.stringify({config:null,iteration:999,stage:999,step:999}))]){
    await page.goto('/walkthrough.html'+suffix);await checkFormula(page);
  }
  expect(errors).toEqual([]);
});

test('responsive walkthrough keeps the map, formulas, settings and keyboard navigation usable',async({page})=>{
  const errors=[];page.on('pageerror',e=>errors.push(e.message));
  for(const width of [320,390,768,1440]){
    await page.setViewportSize({width,height:900});await page.goto('/walkthrough.html');
    await expect(page.locator('.trace-map')).toBeVisible();
    for(let stage=0;stage<6;stage++){
      await page.locator(`[data-stage="${stage}"]`).click();await checkFormula(page);
      for(const details of await page.locator('#formula-panel details').all()){
        await details.locator('summary').click();await checkFormula(page);
      }
    }
    await page.locator('.trace-config summary').click();await checkFormula(page);
  }
  await page.locator('[data-stage="0"]').click();await page.keyboard.press('ArrowRight');
  await expect(page.locator('[data-stage="1"]')).toBeFocused();
  await page.keyboard.press('End');await expect(page.locator('[data-stage="5"]')).toBeFocused();
  await page.locator('[data-step="0"]').click();await page.keyboard.press('ArrowRight');
  await expect(page.locator('[data-step="1"]')).toHaveAttribute('aria-pressed','true');
  await page.setViewportSize({width:1440,height:1050});await page.goto('/walkthrough.html');
  await page.locator('[data-stage="2"]').click();await page.locator('[data-source-step="2"]').click();
  await page.evaluate(()=>window.scrollTo({top:0,behavior:'instant'}));
  await page.screenshot({path:'/tmp/crl-walkthrough-desktop.png',fullPage:true});
  await page.setViewportSize({width:390,height:844});await page.goto('/walkthrough.html');
  await page.locator('[data-step="2"]').click();await page.locator('[data-stage="1"]').click();
  await page.evaluate(()=>window.scrollTo({top:0,behavior:'instant'}));
  await page.screenshot({path:'/tmp/crl-walkthrough-mobile.png',fullPage:true});
  expect(errors).toEqual([]);
});
