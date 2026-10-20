import { test, expect } from '@playwright/test';

test('all lessons, live formulas, and no external runtime dependencies',async({page})=>{
 const errors=[],external=[];page.on('pageerror',e=>errors.push(e.message));
 page.on('request',r=>{if(!r.url().startsWith('http://127.0.0.1:4173')&&!r.url().startsWith('data:'))external.push(r.url());});
 await page.goto('/');await expect(page.locator('h1')).toContainText('Будущее обрывается');
 await expect(page.locator('.method.featured')).toContainText('81,0%');
 await expect(page.locator('.katex-error')).toHaveCount(0);
 await page.locator('[data-preset="timeout"]').click();await expect(page.locator('.callout').first()).toContainText('Terminal penalty не добавляется');
 await page.locator('[data-preset="death"]').click();await page.locator('[data-anchor="2"]').click();
 await expect(page.locator('.method').first()).toContainText('Нет положительной метки');
 await expect(page.locator('.method.featured')).toContainText('100,0%');
 await page.locator('[data-preset="reset"]').click();await expect(page.locator('.state-column')).toHaveCount(5);
 await page.locator('a[data-route="semantics"]').click();await expect(page.locator('h1')).toContainText('Одно состояние');
 await page.locator('#duplicate').check();await expect(page.locator('.semantic-goal')).toHaveCount(3);
 await page.locator('[data-sem="0"]').click();await page.keyboard.press('Space');await expect(page.locator('[data-sem="0"]')).toBeFocused();
 await page.locator('a[data-route="future"]').click();await page.locator('[data-sample="1000"]').click();expect(Number((await page.locator('.sample-counter strong').innerText()).replace(/\s/g,''))).toBe(1000);
 await page.locator('a[data-route="critic"]').click();
 await expect(page.locator('#workspace')).toContainText('Решения различаются');
 const before=await page.locator('.numeric-card strong').first().innerText();
 await page.locator('[data-action="critic-step"]').click();
 const after=await page.locator('.numeric-card strong').first().innerText();expect(after).not.toBe(before);
 await page.locator('a[data-route="teacher"]').click();
 await page.locator('#old').fill('0.05');await page.locator('#old').dispatchEvent('input');
 await page.locator('#progress').fill('100');await page.locator('#progress').dispatchEvent('input');
 await expect(page.locator('#workspace')).toContainText('Учитель отключён');
 await page.locator('[data-action="actor-step"]').click();
 expect(errors).toEqual([]);expect(external).toEqual([]);await expect(page.locator('.katex-error')).toHaveCount(0);
});

test('shared scenario restores values; invalid links recover; responsive layouts',async({page})=>{
 await page.goto('/?state='+encodeURIComponent(JSON.stringify({gamma:.8,term:.5,live:4,anchor:0,end:'timeout',cap:2}))+'#termination');
 await expect(page.locator('#gamma')).toHaveValue('0.8');await expect(page.locator('#end')).toHaveValue('timeout');
 await page.setViewportSize({width:1440,height:1000});await page.screenshot({path:'/tmp/crl-lab-desktop.png',fullPage:true});
 for(const width of [390,320,768,1440]){
  await page.setViewportSize({width,height:900});
  for(const route of ['termination','semantics','future','critic','teacher']){
   await page.goto('/#'+route);await expect(page.locator('h1')).toBeVisible();
   expect(await page.evaluate(()=>document.documentElement.scrollWidth<=window.innerWidth+1), `${width}px / ${route}`).toBeTruthy();
  }
 }
 await page.setViewportSize({width:390,height:844});await page.goto('/#termination');await page.locator('#mobile-controls').click();await expect(page.locator('#controls')).toBeVisible();await page.locator('#gamma').fill('0.7');await page.locator('#gamma').dispatchEvent('input');await expect(page.locator('[data-out=gamma]')).toHaveText('0,700');await page.keyboard.press('Escape');await expect(page.locator('#controls')).not.toBeVisible();await page.screenshot({path:'/tmp/crl-lab-mobile.png',fullPage:true});
 await page.goto('/?state=%7Bbad#critic');await expect(page.locator('h1')).toContainText('Скор');
 await page.locator('[data-open-about]').click();await expect(page.locator('#about')).toBeVisible();
 await page.keyboard.press('Escape');await expect(page.locator('#about')).not.toBeVisible();
});
