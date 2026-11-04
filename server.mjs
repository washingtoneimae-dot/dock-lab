#!/usr/bin/env node
/**
 * Tiny zero-dependency static server for docklab.
 *   node server.mjs [--port 5173] [--open]
 */
import { createServer } from 'node:http';
import { createReadStream } from 'node:fs';
import { stat } from 'node:fs/promises';
import { extname, resolve, sep } from 'node:path';
import { fileURLToPath } from 'node:url';
import { spawn } from 'node:child_process';

const ROOT = resolve(fileURLToPath(new URL('.', import.meta.url)));
const args = process.argv.slice(2);
const portArg = args.indexOf('--port');
const PORT = Number(portArg > -1 ? args[portArg + 1] : process.env.PORT || 5173);
const HOST = process.env.HOST || '127.0.0.1';
const OPEN = args.includes('--open') || process.env.OPEN === '1';

const MIME = {
  '.html': 'text/html; charset=utf-8',
  '.js': 'text/javascript; charset=utf-8',
  '.mjs': 'text/javascript; charset=utf-8',
  '.css': 'text/css; charset=utf-8',
  '.json': 'application/json; charset=utf-8',
  '.svg': 'image/svg+xml',
  '.png': 'image/png',
  '.jpg': 'image/jpeg',
  '.jpeg': 'image/jpeg',
  '.webp': 'image/webp',
  '.avif': 'image/avif',
  '.gif': 'image/gif',
  '.ico': 'image/x-icon',
  '.woff2': 'font/woff2',
  '.txt': 'text/plain; charset=utf-8',
  '.pdb': 'chemical/x-pdb',
  '.ent': 'chemical/x-pdb',
  '.md': 'text/markdown; charset=utf-8'
};

const server = createServer(async (req, res) => {
  const started = Date.now();
  let urlPath;
  try {
    urlPath = decodeURIComponent(new URL(req.url, 'http://localhost').pathname);
  } catch {
    res.writeHead(400).end('Bad request');
    return;
  }
  if (urlPath.endsWith('/')) urlPath += 'index.html';

  const file = resolve(ROOT, '.' + urlPath);
  if (file !== ROOT && !file.startsWith(ROOT + sep)) {
    res.writeHead(403).end('Forbidden');
    return;
  }

  try {
    const info = await stat(file);
    if (!info.isFile()) throw new Error('not a file');

    res.writeHead(200, {
      'Content-Type': MIME[extname(file).toLowerCase()] || 'application/octet-stream',
      'Content-Length': info.size,
      'Cache-Control': 'no-store',
      'X-Content-Type-Options': 'nosniff'
    });
    if (req.method === 'HEAD') {
      res.end();
    } else {
      createReadStream(file).pipe(res);
    }
    const ms = Date.now() - started;
    console.log(`  ${String(res.statusCode).padEnd(3)} ${req.method.padEnd(4)} ${urlPath}  ${ms}ms`);
  } catch {
    res.writeHead(404, { 'Content-Type': 'text/html; charset=utf-8' });
    res.end(`<body style="font:14px/1.5 system-ui;background:#08090c;color:#eef1f6;padding:40px">
      <h1 style="font-size:18px">404</h1><p>${urlPath} was not found.</p>
      <p><a style="color:#c6ff5e" href="/">← back to the lab</a></p></body>`);
    console.log(`  404 ${req.method.padEnd(4)} ${urlPath}`);
  }
});

server.on('error', (err) => {
  if (err.code === 'EADDRINUSE') {
    console.error(`\n  Port ${PORT} is already in use. Try:  node server.mjs --port 5174\n`);
    process.exit(1);
  }
  throw err;
});

/**
 * Open `url` in the default browser, trying platform-appropriate helpers.
 * Never throws and never keeps the server alive if no opener is available.
 */
function openBrowser(url) {
  const attempts = process.platform === 'darwin'
    ? [['open', [url]]]
    : process.platform === 'win32'
      // Windows: `start` is a cmd builtin, not an executable; the empty string is its title argument.
      ? [['cmd', ['/c', 'start', '', url]]]
      : [['xdg-open', [url]], ['gio', ['open', url]], ['sensible-browser', [url]]];

  const tryNext = (index) => {
    if (index >= attempts.length) {
      console.log(`  Could not open a browser automatically. Open ${url} manually.`);
      return;
    }
    const [command, args] = attempts[index];
    let child;
    try {
      child = spawn(command, args, {
        stdio: 'ignore',
        detached: true,
        windowsHide: true
      });
    } catch {
      tryNext(index + 1);
      return;
    }

    let settled = false;
    const next = () => {
      if (settled) return;
      settled = true;
      tryNext(index + 1);
    };
    child.once('error', next);
    child.once('exit', (code) => {
      if (!settled && code !== 0) next();
      else settled = true;
    });
    child.unref();
  };

  tryNext(0);
}

server.listen(PORT, HOST, () => {
  const url = `http://${HOST === '0.0.0.0' ? 'localhost' : HOST}:${PORT}/`;
  console.log(`\n  docklab  →  ${url}\n  serving ${ROOT}\n  ctrl+c to stop\n`);
  if (OPEN) openBrowser(url);
});
