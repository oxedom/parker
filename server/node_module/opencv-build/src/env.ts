import * as fs from 'fs';
import * as os from 'os';
import * as path from 'path';

import { dirs } from './dirs';
import { AutoBuildFile } from './types';

const log = require('npmlog')

export function isAutoBuildDisabled() {
  return !!process.env.OPENCV4NODEJS_DISABLE_AUTOBUILD
}

export function buildWithCuda() : boolean {
  return !!process.env.OPENCV4NODEJS_BUILD_CUDA || false;
}

export function isWithoutContrib() {
  return !!process.env.OPENCV4NODEJS_AUTOBUILD_WITHOUT_CONTRIB
}

export function autoBuildFlags(): string {
  return process.env.OPENCV4NODEJS_AUTOBUILD_FLAGS || ''
}

export function opencvVersion() {
  return process.env.OPENCV4NODEJS_AUTOBUILD_OPENCV_VERSION || '3.4.6'
}

export function numberOfCoresAvailable() {
  return os.cpus().length
}

export function parseAutoBuildFlags(): string[] {
  const flagStr = autoBuildFlags()
  if (typeof(flagStr) === 'string' && flagStr.length) {
    log.silly('install', 'using flags from OPENCV4NODEJS_AUTOBUILD_FLAGS:', flagStr)
    return flagStr.split(' ')
  }
  return []
}

export function readAutoBuildFile(): AutoBuildFile | undefined {
  try {
    const fileExists = fs.existsSync(dirs.autoBuildFile)
    if (fileExists) {
      const autoBuildFile = JSON.parse(fs.readFileSync(dirs.autoBuildFile).toString()) as AutoBuildFile
      if (!autoBuildFile.opencvVersion || !('autoBuildFlags' in autoBuildFile) || !Array.isArray(autoBuildFile.modules)) {
        throw new Error('auto-build.json has invalid contents')
      }
      return autoBuildFile
    }
    log.info('readAutoBuildFile', 'file does not exists: %s', dirs.autoBuildFile, dirs.autoBuildFile)
  } catch (err) {
    log.error('readAutoBuildFile', 'failed to read auto-build.json from: %s, with error: %s', dirs.autoBuildFile, err.toString())
  }
  return undefined
}

export function getCwd() {
  const cwd = process.env.INIT_CWD || process.cwd()
  if (!cwd) {
    throw new Error('process.env.INIT_CWD || process.cwd() is undefined or empty')
  }
  return cwd
}

function parsePackageJson() {
  const absPath = path.resolve(getCwd(), 'package.json')
  if (!fs.existsSync(absPath)) {
    return null
  }
  return JSON.parse(fs.readFileSync(absPath).toString())
}

export function readEnvsFromPackageJson() {
  const rootPackageJSON = parsePackageJson()
  return rootPackageJSON
    ? (rootPackageJSON.opencv4nodejs || {})
    : {}
}

export function applyEnvsFromPackageJson() {
  let envs: any = {}
  try {
    envs = readEnvsFromPackageJson()
  } catch (err) {
    log.error('failed to parse package.json:')
    log.error(err)
  }

  const envKeys = Object.keys(envs)
  if (envKeys.length) {
    log.info('the following opencv4nodejs environment variables are set in the package.json:')
    envKeys.forEach(key => log.info(`${key}: ${envs[key]}`))
  }

  const {
    autoBuildBuildCuda,
    autoBuildFlags,
    autoBuildOpencvVersion,
    autoBuildWithoutContrib,
    disableAutoBuild,
    opencvIncludeDir,
    opencvLibDir,
    opencvBinDir
  } = envs

  if (autoBuildFlags) {
    process.env.OPENCV4NODEJS_AUTOBUILD_FLAGS = autoBuildFlags
  }

  if (autoBuildBuildCuda) {
    process.env.OPENCV4NODEJS_BUILD_CUDA = autoBuildBuildCuda
  }

  if (autoBuildOpencvVersion) {
    process.env.OPENCV4NODEJS_AUTOBUILD_OPENCV_VERSION = autoBuildOpencvVersion
  }

  if (autoBuildWithoutContrib) {
    process.env.OPENCV4NODEJS_AUTOBUILD_WITHOUT_CONTRIB = autoBuildWithoutContrib
  }

  if (disableAutoBuild) {
    process.env.OPENCV4NODEJS_DISABLE_AUTOBUILD = disableAutoBuild
  }

  if (opencvIncludeDir) {
    process.env.OPENCV_INCLUDE_DIR = opencvIncludeDir
  }

  if (opencvLibDir) {
    process.env.OPENCV_LIB_DIR = opencvLibDir
  }

  if (opencvBinDir) {
    process.env.OPENCV_BIN_DIR = opencvBinDir
  }
}