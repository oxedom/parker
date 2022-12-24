import * as path from 'path';

import { isWin } from './utils';

const rootDir = path.resolve(__dirname, '../')
const opencvRoot = path.join(rootDir, 'opencv')
const opencvSrc = path.join(opencvRoot, 'opencv')
const opencvContribSrc = path.join(opencvRoot, 'opencv_contrib')
const opencvContribModules = path.join(opencvContribSrc, 'modules')
const opencvBuild = path.join(opencvRoot, 'build')
const opencvInclude = path.join(opencvBuild, 'include')
const opencv4Include = path.join(opencvInclude, 'opencv4')
const opencvLibDir = isWin() ? path.join(opencvBuild, 'lib/Release') : path.join(opencvBuild, 'lib')
const opencvBinDir = isWin() ? path.join(opencvBuild, 'bin/Release') : path.join(opencvBuild, 'bin')
const autoBuildFile = path.join(opencvRoot, 'auto-build.json')

export const dirs = {
  rootDir,
  opencvRoot,
  opencvSrc,
  opencvContribSrc,
  opencvContribModules,
  opencvBuild,
  opencvInclude,
  opencv4Include,
  opencvLibDir,
  opencvBinDir,
  autoBuildFile
}