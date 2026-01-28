const path = require('path');
require('dotenv').config({ path: path.join(__dirname, '.env') });
const { spawnSync } = require('child_process');

const user = process.env.EXPO_USER;
const pass = process.env.EXPO_PASS;

if (!user || !pass) {
  console.error('Falta EXPO_USER o EXPO_PASS en las variables de entorno.');
  process.exit(1);
}

let r = spawnSync('npx', ['expo', 'login', '-u', user, '-p', pass], { stdio: 'inherit', shell: true });
if (r.status !== 0) process.exit(r.status);

r = spawnSync('npx', ['expo', 'start', '--tunnel', ...process.argv.slice(2)], { stdio: 'inherit', shell: true });
process.exit(r.status || 0);