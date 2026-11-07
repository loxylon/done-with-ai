import {test,expect} from '@playwright/test';

const check=async page=>{
  await expect(page.locator('#mega-content')).toBeVisible();
  await expect(page.locator('.katex-error')).toHaveCount(0);
  expect(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth+1)).toBe(true);
};

test('MEGA pool, rejection, empty buffer and OMEGA support edge cases update all displayed computations',async({page})=>{
  const errors=[],external=[];page.on('pageerror',e=>errors.push(e.message));
  page.on('request',r=>{if(!r.url().startsWith('http://127.0.0.1:4173')&&!r.url().startsWith('data:'))external.push(r.url());});
  await page.goto('/mega.html#selector');
  await expect(page.locator('#mega-content')).toContainText('MEGA выбрал C');await check(page);
  await page.locator('[data-lab="seed"]').fill('17');await page.locator('[data-lab="seed"]').dispatchEvent('change');
  await expect(page.locator('#mega-content')).toContainText('не попала в pool');
  await page.locator('[data-preset="rare"]').click();
  await expect(page.locator('.mtable .rejected').filter({hasText:'E'})).toContainText('E');
  await page.locator('[data-lab="cutoff"]').fill('0');await page.locator('[data-lab="cutoff"]').dispatchEvent('change');
  await expect(page.locator('#mega-content')).toContainText('Все выпавшие кандидаты отвергнуты');await check(page);
  await page.locator('[data-preset="empty"]').click();
  await expect(page.locator('#mega-content')).toContainText('Буфер пуст');await check(page);
  await page.locator('[data-pane="entropy"]').click();await expect(page.locator('#mega-content')).toContainText('Сначала нужны наблюдения');await check(page);
  await page.locator('[data-pane="omega"]').click();
  await expect(page.locator('#mega-content')).toContainText('objective не определён');
  await expect(page.locator('.mformula').filter({hasText:'OMEGA OBJECTIVE'}).locator('[data-tex]')).toHaveAttribute('data-tex',/\\text\{не определён\}/);
  await check(page);
  await page.locator('[data-preset="contact"]').click();await check(page);
  await expect(page.locator('#mega-content')).toContainText('Вся поддержка pDG встречалась');
  await page.locator('[data-preset="covered"]').click();await check(page);
  await expect(page.locator('.mstat').filter({hasText:'Внешняя ветка'})).toContainText('100%');
  for(let g=0;g<5;g++){
    await page.locator(`[data-lab="desired-${g}"]`).fill('0');await page.locator(`[data-lab="desired-${g}"]`).dispatchEvent('change');
  }
  await expect(page.locator('#mega-content')).toContainText('Нужны непустой буфер');await check(page);
  expect(errors).toEqual([]);expect(external).toEqual([]);
});

test('entropy oracle changes when rare C stops being achievable; q weights and observed counts are actual inputs',async({page})=>{
  const errors=[];page.on('pageerror',e=>errors.push(e.message));
  await page.goto('/mega.html#entropy');await check(page);
  await expect(page.locator('.mstat').filter({hasText:'Oracle · знает q'})).toContainText('C');
  await page.locator('[data-preset="broken"]').click();
  await expect(page.locator('.mstat').filter({hasText:'Oracle · знает q'})).toContainText('B');
  await expect(page.locator('.mstat').filter({hasText:'MEGA · текущий pool'})).toContainText('C');
  await page.locator('[data-preset="frontier"]').click();await check(page);
  for(let g=0;g<5;g++){
    await page.locator(`[data-outcome="${g}"]`).fill('0');await page.locator(`[data-outcome="${g}"]`).dispatchEvent('change');
  }
  await expect(page.locator('#mega-content')).toContainText('Все веса q равны 0');
  await expect(page.locator('[data-action="add-outcome"]')).toBeDisabled();
  await page.locator('[data-outcome="3"]').fill('1');await page.locator('[data-outcome="3"]').dispatchEvent('change');
  await page.locator('[data-action="add-outcome"]').click();
  await expect(page.locator('#mega-status')).toContainText('Добавлено одно наблюдение D');
  await page.locator('[data-pane="selector"]').click();await expect(page.locator('[data-lab="count-3"]')).toHaveValue('1');
  expect(errors).toEqual([]);
});

test('five real Q/HER agents train, replay steps, transfer counts and restore the whole experience from a link',async({page})=>{
  const errors=[];page.on('pageerror',e=>errors.push(e.message));
  await page.goto('/mega.html#environment');
  await page.locator('[data-episodes="100"]').click();
  await expect(page.locator('#mega-status')).toContainText('Готово: 100 эпизодов');
  await expect(page.locator('[data-episode-index] option')).toHaveCount(100);await check(page);
  await expect(page.locator('.mcompare [data-method="MEGA"]')).toContainText('5/5');
  await page.locator('[data-episode-index]').selectOption('27');
  await page.locator('[data-step-index]').selectOption('2');
  const mapBefore=await page.locator('.mmap').getAttribute('aria-label');
  await page.locator('[data-action="next-step"]').click();
  expect(await page.locator('.mmap').getAttribute('aria-label')).not.toBe(mapBefore);
  await expect(page.locator('#mega-content')).toContainText('TD target');
  await page.locator('#mega-share').click();expect(page.url()).toContain('scenario=');
  const before=await page.locator('#mega-content').innerText();await page.reload();
  expect(await page.locator('#mega-content').innerText()).toBe(before);await check(page);
  await page.locator('[data-action="copy-buffer"]').click();await expect(page).toHaveURL(/#selector/);
  expect(Number(await page.locator('[data-lab="count-4"]').inputValue())).toBeGreaterThan(0);
  await page.locator('[data-pane="environment"]').click();
  await page.locator('.mdetail').filter({has:page.locator('#mega-train-form')}).locator('summary').click();
  await page.locator('[name="blocked"]').selectOption('true');
  await page.locator('[name="edge"]').fill('1');
  await page.getByRole('button',{name:'Применить и сбросить опыт'}).click();
  await expect(page.locator('#mega-content')).toContainText('только одинаковый случайный разогрев');
  await page.locator('[data-episodes="10"]').click();await expect(page.locator('#mega-status')).toContainText('Готово: 10 эпизодов');
  await expect(page.locator('.mcompare [data-method="MEGA"]')).toContainText('4/5');await check(page);
  expect(errors).toEqual([]);
});

test('all sections remain usable on mobile and malformed shared inputs recover',async({page})=>{
  const errors=[];page.on('pageerror',e=>errors.push(e.message));
  for(const width of [320,390,768,1440]){
    await page.setViewportSize({width,height:900});
    for(const pane of ['problem','selector','environment','entropy','omega']){
      await page.goto('/mega.html?viewport='+width+'#'+pane);await check(page);
      if(pane==='environment'){
        await page.locator('[data-episodes="1"]').click();await expect(page.locator('#mega-status')).toContainText('Готово: 1 эпизод');await check(page);
      }
      for(const details of await page.locator('#mega-content details').all()){
        await details.locator('summary').click();await check(page);
      }
    }
  }
  for(const query of ['%7Bbad',encodeURIComponent(JSON.stringify({lab:null,train:null,count:0})),encodeURIComponent(JSON.stringify({lab:{counts:[1,2],cutoff:100},count:-4}))]){
    await page.goto('/mega.html?scenario='+query+'#omega');await check(page);
  }
  await page.setViewportSize({width:1440,height:1000});await page.goto('/mega.html#selector');
  await page.evaluate(()=>window.scrollTo({top:0,behavior:'instant'}));await page.screenshot({path:'/tmp/mega-desktop.png',fullPage:true});
  await page.setViewportSize({width:390,height:844});await page.goto('/mega.html#problem');
  await page.screenshot({path:'/tmp/mega-mobile.png',fullPage:true});
  expect(errors).toEqual([]);
});
