import { cp, mkdir, rm, readFile, writeFile } from 'node:fs/promises';
import { resolve } from 'node:path';
const root = resolve(import.meta.dirname, '..');
const out = resolve(root, 'dist');
await rm(out, { recursive: true, force: true });
await mkdir(`${out}/vendor/fonts`, { recursive: true });
for (const file of ['index.html', 'styles.css', 'src']) {
  await cp(`${root}/${file}`, `${out}/${file}`, { recursive: true });
}
for (const file of ['katex.mjs', 'katex.min.css', 'fonts']) {
  await cp(`${root}/node_modules/katex/dist/${file}`, `${out}/vendor/${file}`, { recursive: true });
}
const fonts = `${root}/node_modules/@fontsource-variable/manrope/files`;
for (const subset of ['latin', 'cyrillic']) {
  await cp(`${fonts}/manrope-${subset}-wght-normal.woff2`, `${out}/vendor/fonts/manrope-${subset}.woff2`);
}
for (const [source, target] of [
  ['katex/LICENSE', 'KATEX-LICENSE.txt'],
  ['@fontsource-variable/manrope/LICENSE', 'MANROPE-LICENSE.txt'],
]) await cp(`${root}/node_modules/${source}`, `${out}/vendor/${target}`);
await writeFile(`${out}/.nojekyll`, '');
console.log('Static site built in crl-lab/dist. All fonts, equations and calculations are local.');
