"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
var fs = require("fs");
var os = require("os");
var path = require("path");
var dirs_1 = require("./dirs");
var log = require('npmlog');
function isAutoBuildDisabled() {
    return !!process.env.OPENCV4NODEJS_DISABLE_AUTOBUILD;
}
exports.isAutoBuildDisabled = isAutoBuildDisabled;
function buildWithCuda() {
    return !!process.env.OPENCV4NODEJS_BUILD_CUDA || false;
}
exports.buildWithCuda = buildWithCuda;
function isWithoutContrib() {
    return !!process.env.OPENCV4NODEJS_AUTOBUILD_WITHOUT_CONTRIB;
}
exports.isWithoutContrib = isWithoutContrib;
function autoBuildFlags() {
    return process.env.OPENCV4NODEJS_AUTOBUILD_FLAGS || '';
}
exports.autoBuildFlags = autoBuildFlags;
function opencvVersion() {
    return process.env.OPENCV4NODEJS_AUTOBUILD_OPENCV_VERSION || '3.4.6';
}
exports.opencvVersion = opencvVersion;
function numberOfCoresAvailable() {
    return os.cpus().length;
}
exports.numberOfCoresAvailable = numberOfCoresAvailable;
function parseAutoBuildFlags() {
    var flagStr = autoBuildFlags();
    if (typeof (flagStr) === 'string' && flagStr.length) {
        log.silly('install', 'using flags from OPENCV4NODEJS_AUTOBUILD_FLAGS:', flagStr);
        return flagStr.split(' ');
    }
    return [];
}
exports.parseAutoBuildFlags = parseAutoBuildFlags;
function readAutoBuildFile() {
    try {
        var fileExists = fs.existsSync(dirs_1.dirs.autoBuildFile);
        if (fileExists) {
            var autoBuildFile = JSON.parse(fs.readFileSync(dirs_1.dirs.autoBuildFile).toString());
            if (!autoBuildFile.opencvVersion || !('autoBuildFlags' in autoBuildFile) || !Array.isArray(autoBuildFile.modules)) {
                throw new Error('auto-build.json has invalid contents');
            }
            return autoBuildFile;
        }
        log.info('readAutoBuildFile', 'file does not exists: %s', dirs_1.dirs.autoBuildFile, dirs_1.dirs.autoBuildFile);
    }
    catch (err) {
        log.error('readAutoBuildFile', 'failed to read auto-build.json from: %s, with error: %s', dirs_1.dirs.autoBuildFile, err.toString());
    }
    return undefined;
}
exports.readAutoBuildFile = readAutoBuildFile;
function getCwd() {
    var cwd = process.env.INIT_CWD || process.cwd();
    if (!cwd) {
        throw new Error('process.env.INIT_CWD || process.cwd() is undefined or empty');
    }
    return cwd;
}
exports.getCwd = getCwd;
function parsePackageJson() {
    var absPath = path.resolve(getCwd(), 'package.json');
    if (!fs.existsSync(absPath)) {
        return null;
    }
    return JSON.parse(fs.readFileSync(absPath).toString());
}
function readEnvsFromPackageJson() {
    var rootPackageJSON = parsePackageJson();
    return rootPackageJSON
        ? (rootPackageJSON.opencv4nodejs || {})
        : {};
}
exports.readEnvsFromPackageJson = readEnvsFromPackageJson;
function applyEnvsFromPackageJson() {
    var envs = {};
    try {
        envs = readEnvsFromPackageJson();
    }
    catch (err) {
        log.error('failed to parse package.json:');
        log.error(err);
    }
    var envKeys = Object.keys(envs);
    if (envKeys.length) {
        log.info('the following opencv4nodejs environment variables are set in the package.json:');
        envKeys.forEach(function (key) { return log.info(key + ": " + envs[key]); });
    }
    var autoBuildBuildCuda = envs.autoBuildBuildCuda, autoBuildFlags = envs.autoBuildFlags, autoBuildOpencvVersion = envs.autoBuildOpencvVersion, autoBuildWithoutContrib = envs.autoBuildWithoutContrib, disableAutoBuild = envs.disableAutoBuild, opencvIncludeDir = envs.opencvIncludeDir, opencvLibDir = envs.opencvLibDir, opencvBinDir = envs.opencvBinDir;
    if (autoBuildFlags) {
        process.env.OPENCV4NODEJS_AUTOBUILD_FLAGS = autoBuildFlags;
    }
    if (autoBuildBuildCuda) {
        process.env.OPENCV4NODEJS_BUILD_CUDA = autoBuildBuildCuda;
    }
    if (autoBuildOpencvVersion) {
        process.env.OPENCV4NODEJS_AUTOBUILD_OPENCV_VERSION = autoBuildOpencvVersion;
    }
    if (autoBuildWithoutContrib) {
        process.env.OPENCV4NODEJS_AUTOBUILD_WITHOUT_CONTRIB = autoBuildWithoutContrib;
    }
    if (disableAutoBuild) {
        process.env.OPENCV4NODEJS_DISABLE_AUTOBUILD = disableAutoBuild;
    }
    if (opencvIncludeDir) {
        process.env.OPENCV_INCLUDE_DIR = opencvIncludeDir;
    }
    if (opencvLibDir) {
        process.env.OPENCV_LIB_DIR = opencvLibDir;
    }
    if (opencvBinDir) {
        process.env.OPENCV_BIN_DIR = opencvBinDir;
    }
}
exports.applyEnvsFromPackageJson = applyEnvsFromPackageJson;
