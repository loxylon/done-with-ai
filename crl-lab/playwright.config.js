import { defineConfig } from '@playwright/test';
export default defineConfig({
  testDir: './tests', testMatch: '**/*.spec.js', workers: 1,
  use: { baseURL:'http://127.0.0.1:4173', headless:true,
    launchOptions:{ executablePath:process.env.CHROMIUM_PATH||(process.env.CI?undefined:'/usr/bin/chromium'),args:['--no-sandbox'] } },
  webServer:{ command:'npm run preview',url:'http://127.0.0.1:4173',reuseExistingServer:!process.env.CI },
});
