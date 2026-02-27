const path = require('path');
// Buscamos el .env en la raíz del proyecto (un nivel arriba de /scripts)
require('dotenv').config({ path: path.join(__dirname, '..', '.env') });
const { spawnSync } = require('child_process');

const user = process.env.EXPO_USER;
const pass = process.env.EXPO_PASS;

if (!user || !pass) {
  console.error('Error: Falta EXPO_USER o EXPO_PASS. Asegúrate de que estén en el archivo .env o en las variables de entorno de la configuración.');
  process.exit(1);
}

console.log('Iniciando sesión en Expo...');
let r = spawnSync('npx', ['expo', 'login', '-u', user, '-p', pass], { stdio: 'inherit', shell: true });
if (r.status !== 0) process.exit(r.status);

console.log('Iniciando servidor Expo con tunnel...');
r = spawnSync('npx', ['expo', 'start', '--tunnel', ...process.argv.slice(2)], { stdio: 'inherit', shell: true });
process.exit(r.status || 0);
