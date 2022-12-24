"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
var path = require("path");
var utils_1 = require("./utils");
var rootDir = path.resolve(__dirname, '../');
var opencvRoot = path.join(rootDir, 'opencv');
var opencvSrc = path.join(opencvRoot, 'opencv');
var opencvContribSrc = path.join(opencvRoot, 'opencv_contrib');
var opencvContribModules = path.join(opencvContribSrc, 'modules');
var opencvBuild = path.join(opencvRoot, 'build');
var opencvInclude = path.join(opencvBuild, 'include');
var opencv4Include = path.join(opencvInclude, 'opencv4');
var opencvLibDir = utils_1.isWin() ? path.join(opencvBuild, 'lib/Release') : path.join(opencvBuild, 'lib');
var opencvBinDir = utils_1.isWin() ? path.join(opencvBuild, 'bin/Release') : path.join(opencvBuild, 'bin');
var autoBuildFile = path.join(opencvRoot, 'auto-build.json');
exports.dirs = {
    rootDir: rootDir,
    opencvRoot: opencvRoot,
    opencvSrc: opencvSrc,
    opencvContribSrc: opencvContribSrc,
    opencvContribModules: opencvContribModules,
    opencvBuild: opencvBuild,
    opencvInclude: opencvInclude,
    opencv4Include: opencv4Include,
    opencvLibDir: opencvLibDir,
    opencvBinDir: opencvBinDir,
    autoBuildFile: autoBuildFile
};
