import * as child_process from 'child_process';
import * as fs from 'fs';
import * as path from 'path';

const log = require('npmlog')

export function exec(cmd: string, options?: child_process.ExecOptions): Promise<string> {
  log.silly('install', 'executing:', cmd)
  return new Promise(function(resolve, reject) {
    child_process.exec(cmd, options, function(err, stdout, stderr) {
      const _err = err || stderr
      if (_err) return reject(_err)
      return resolve(stdout.toString())
    })
  })
}

export function execFile(cmd: string, args: string[], options?: child_process.ExecOptions): Promise<string> {
  log.silly('install', 'executing:', cmd, args)
  return new Promise(function(resolve, reject) {
    const child = child_process.execFile(cmd, args, options, function(err, stdout, stderr) {
      const _err = err || stderr
      if (_err) return reject(_err)
      return resolve(stdout.toString())
    })
    child.stdin && child.stdin.end()
  })
}

export function spawn(cmd: string, args: string[], options?: child_process.ExecOptions): Promise<string> {
  log.silly('install', 'spawning:', cmd, args)
  return new Promise(function(resolve, reject) {
    try {
      const child = child_process.spawn(cmd, args, Object.assign({}, { stdio: 'inherit' }, options))

      child.on('exit', function (code) {
        if (typeof code !== 'number') {
          code = null
        }
        const msg = 'child process exited with code ' + code + ' (for more info, set \'--loglevel silly\')'
        if (code !== 0) {
          return reject(msg)
        }
        return resolve(msg)
      })
    } catch(err) {
      return reject(err)
    }
  })
}

async function requireCmd(cmd: string, hint: string) {
  log.info('install', `executing: ${cmd}`)
  try {
    const stdout = await exec(cmd)
    log.info('install', `${cmd}: ${stdout}`)
  } catch (err) {
    const errMessage = `failed to execute ${cmd}, ${hint}, error is: ${err.toString()}`
    throw new Error(errMessage)
  }
}

export async function requireGit() {
  await requireCmd('git --version', 'git is required')
}

export async function requireCmake() {
  await requireCmd('cmake --version', 'cmake is required to build opencv')
}

export function isWin () {
  return process.platform == 'win32'
}

export function isOSX() {
  return process.platform == 'darwin'
}

export function isUnix() {
  return !isWin() && !isOSX()
}

export async function isCudaAvailable() {
  log.info('install', 'Check if CUDA is available & what version...');

  if (isWin()) {
    try {
      await requireCmd('nvcc --version', 'CUDA availability check');
      return true;
    } catch (err) {
      log.info('install', 'Seems like CUDA is not installed.');
      return false;
    }
  }

  // Because NVCC is not installed by default & requires an extra install step,
  // this is work around that always works
  const cudaVersionFilePath = path.resolve('/usr/local/cuda/version.txt');

  if (fs.existsSync(cudaVersionFilePath)) {
    const content = fs.readFileSync(cudaVersionFilePath, 'utf8');
    log.info('install', content);
    return true;
  } else {
    log.info('install', 'CUDA version file could not be found.');
    return false;
  }
}
 