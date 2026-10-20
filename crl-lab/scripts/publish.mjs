/** Publish the tested static build, preserving main's index and gh-pages history.
 * No force-push, tokens in files, repository visibility changes, or API workarounds.
 * Pages must be configured once to serve gh-pages at / via repository settings.
 */
import { execFileSync } from 'node:child_process';
import { mkdtemp, rm, access } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { resolve, join } from 'node:path';
const root = resolve(import.meta.dirname, '..');
const dist = resolve(root, 'dist');
await access(`${dist}/index.html`);
const git = (args, options={}) => execFileSync('git', args, { cwd:root, encoding:'utf8', ...options }).trim();
const source = git(['rev-parse','HEAD']);
const gitDir = git(['rev-parse','--absolute-git-dir']);
let parent;
try {
  git(['ls-remote','--exit-code','origin','refs/heads/gh-pages']);
  git(['fetch','origin','refs/heads/gh-pages']);
  parent = git(['rev-parse','FETCH_HEAD']);
} catch(error) { if(error.status!==2) throw error; }
const scratch = await mkdtemp(join(tmpdir(),'crl-publish-'));
try {
  const env = { ...process.env, GIT_DIR:gitDir, GIT_WORK_TREE:dist, GIT_INDEX_FILE:join(scratch,'index') };
  git(['add','--all'],{env,cwd:dist});
  const tree = git(['write-tree'],{env,cwd:dist});
  if(parent && tree===git(['rev-parse',`${parent}^{tree}`])) {
    console.log('gh-pages already contains this exact build.');
  } else {
    const args=['-c','user.name=CRL Lab Publisher','-c','user.email=crl-lab@users.noreply.github.com','commit-tree',tree];
    if(parent)args.push('-p',parent);
    args.push('-F','-');
    const commit=git(args,{input:`Publish CRL Lab from ${source}\n\nStatic assets only; sources and tests remain on main.\n`});
    git(['push','origin',`${commit}:refs/heads/gh-pages`],{stdio:['ignore','pipe','inherit']});
    console.log(`Published tested assets to gh-pages: ${commit}`);
  }
} finally { await rm(scratch,{recursive:true,force:true}); }
