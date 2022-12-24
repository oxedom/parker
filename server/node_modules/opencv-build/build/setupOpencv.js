"use strict";
var __awaiter = (this && this.__awaiter) || function (thisArg, _arguments, P, generator) {
    function adopt(value) { return value instanceof P ? value : new P(function (resolve) { resolve(value); }); }
    return new (P || (P = Promise))(function (resolve, reject) {
        function fulfilled(value) { try { step(generator.next(value)); } catch (e) { reject(e); } }
        function rejected(value) { try { step(generator["throw"](value)); } catch (e) { reject(e); } }
        function step(result) { result.done ? resolve(result.value) : adopt(result.value).then(fulfilled, rejected); }
        step((generator = generator.apply(thisArg, _arguments || [])).next());
    });
};
var __generator = (this && this.__generator) || function (thisArg, body) {
    var _ = { label: 0, sent: function() { if (t[0] & 1) throw t[1]; return t[1]; }, trys: [], ops: [] }, f, y, t, g;
    return g = { next: verb(0), "throw": verb(1), "return": verb(2) }, typeof Symbol === "function" && (g[Symbol.iterator] = function() { return this; }), g;
    function verb(n) { return function (v) { return step([n, v]); }; }
    function step(op) {
        if (f) throw new TypeError("Generator is already executing.");
        while (_) try {
            if (f = 1, y && (t = op[0] & 2 ? y["return"] : op[0] ? y["throw"] || ((t = y["return"]) && t.call(y), 0) : y.next) && !(t = t.call(y, op[1])).done) return t;
            if (y = 0, t) op = [op[0] & 2, t.value];
            switch (op[0]) {
                case 0: case 1: t = op; break;
                case 4: _.label++; return { value: op[1], done: false };
                case 5: _.label++; y = op[1]; op = [0]; continue;
                case 7: op = _.ops.pop(); _.trys.pop(); continue;
                default:
                    if (!(t = _.trys, t = t.length > 0 && t[t.length - 1]) && (op[0] === 6 || op[0] === 2)) { _ = 0; continue; }
                    if (op[0] === 3 && (!t || (op[1] > t[0] && op[1] < t[3]))) { _.label = op[1]; break; }
                    if (op[0] === 6 && _.label < t[1]) { _.label = t[1]; t = op; break; }
                    if (t && _.label < t[2]) { _.label = t[2]; _.ops.push(op); break; }
                    if (t[2]) _.ops.pop();
                    _.trys.pop(); continue;
            }
            op = body.call(thisArg, _);
        } catch (e) { op = [6, e]; y = 0; } finally { f = t = 0; }
        if (op[0] & 5) throw op[1]; return { value: op[0] ? op[1] : void 0, done: true };
    }
};
Object.defineProperty(exports, "__esModule", { value: true });
var fs = require("fs");
var path = require("path");
var _1 = require(".");
var constants_1 = require("./constants");
var dirs_1 = require("./dirs");
var env_1 = require("./env");
var findMsBuild_1 = require("./findMsBuild");
var utils_1 = require("./utils");
var log = require('npmlog');
function getIfExistsDirCmd(dirname, exists) {
    if (exists === void 0) { exists = true; }
    return utils_1.isWin() ? "if " + (!exists ? 'not ' : '') + "exist " + dirname : '';
}
function getMkDirCmd(dirname) {
    return utils_1.isWin() ? getIfExistsDirCmd(dirname, false) + " mkdir " + dirname : "mkdir -p " + dirname;
}
function getRmDirCmd(dirname) {
    return utils_1.isWin() ? getIfExistsDirCmd(dirname) + " rd /s /q " + dirname : "rm -rf " + dirname;
}
function getMsbuildCmd(sln) {
    return [
        sln,
        '/p:Configuration=Release',
        "/p:Platform=" + (process.arch === 'x64' ? 'x64' : 'x86')
    ];
}
function getRunBuildCmd(msbuildExe) {
    var _this = this;
    if (msbuildExe) {
        return function () { return __awaiter(_this, void 0, void 0, function () {
            return __generator(this, function (_a) {
                switch (_a.label) {
                    case 0: return [4 /*yield*/, utils_1.spawn("" + msbuildExe, getMsbuildCmd('./OpenCV.sln'), { cwd: dirs_1.dirs.opencvBuild })];
                    case 1:
                        _a.sent();
                        return [4 /*yield*/, utils_1.spawn("" + msbuildExe, getMsbuildCmd('./INSTALL.vcxproj'), { cwd: dirs_1.dirs.opencvBuild })];
                    case 2:
                        _a.sent();
                        return [2 /*return*/];
                }
            });
        }); };
    }
    return function () { return __awaiter(_this, void 0, void 0, function () {
        return __generator(this, function (_a) {
            switch (_a.label) {
                case 0: return [4 /*yield*/, utils_1.spawn('make', ['install', "-j" + env_1.numberOfCoresAvailable()], { cwd: dirs_1.dirs.opencvBuild })
                    // revert the strange archiving of libopencv.so going on with make install
                ];
                case 1:
                    _a.sent();
                    // revert the strange archiving of libopencv.so going on with make install
                    return [4 /*yield*/, utils_1.spawn('make', ['all', "-j" + env_1.numberOfCoresAvailable()], { cwd: dirs_1.dirs.opencvBuild })];
                case 2:
                    // revert the strange archiving of libopencv.so going on with make install
                    _a.sent();
                    return [2 /*return*/];
            }
        });
    }); };
}
function getCudaCmakeFlags() {
    return [
        '-DWITH_CUDA=ON',
        '-DBUILD_opencv_cudacodec=OFF',
        '-DCUDA_FAST_MATH=ON',
        '-DWITH_CUBLAS=ON',
    ];
}
function getSharedCmakeFlags() {
    var conditionalFlags = env_1.isWithoutContrib()
        ? []
        : [
            '-DOPENCV_ENABLE_NONFREE=ON',
            "-DOPENCV_EXTRA_MODULES_PATH=" + dirs_1.dirs.opencvContribModules
        ];
    if (env_1.buildWithCuda() && utils_1.isCudaAvailable()) {
        log.info('install', 'Adding CUDA flags...');
        conditionalFlags = conditionalFlags.concat(getCudaCmakeFlags());
    }
    return constants_1.defaultCmakeFlags
        .concat(conditionalFlags)
        .concat(env_1.parseAutoBuildFlags());
}
function getWinCmakeFlags(msversion) {
    var cmakeVsCompiler = constants_1.cmakeVsCompilers[msversion];
    var cmakeArch = constants_1.cmakeArchs[process.arch];
    if (!cmakeVsCompiler) {
        throw new Error("no cmake vs compiler found for msversion: " + msversion);
    }
    if (!cmakeArch) {
        throw new Error("no cmake arch found for process.arch: " + process.arch);
    }
    return [
        '-G',
        "" + cmakeVsCompiler + cmakeArch
    ].concat(getSharedCmakeFlags());
}
function getCmakeArgs(cmakeFlags) {
    return [dirs_1.dirs.opencvSrc].concat(cmakeFlags);
}
function getMsbuildIfWin() {
    return __awaiter(this, void 0, void 0, function () {
        var msbuild;
        return __generator(this, function (_a) {
            switch (_a.label) {
                case 0:
                    if (!utils_1.isWin()) return [3 /*break*/, 2];
                    return [4 /*yield*/, findMsBuild_1.findMsBuild()];
                case 1:
                    msbuild = _a.sent();
                    log.info('install', 'using msbuild:', msbuild);
                    return [2 /*return*/, msbuild];
                case 2: return [2 /*return*/];
            }
        });
    });
}
function writeAutoBuildFile() {
    var autoBuildFile = {
        opencvVersion: env_1.opencvVersion(),
        autoBuildFlags: env_1.autoBuildFlags(),
        modules: _1.getLibs(dirs_1.dirs.opencvLibDir)
    };
    log.info('install', 'writing auto-build file into directory: %s', dirs_1.dirs.autoBuildFile);
    log.info('install', autoBuildFile);
    fs.writeFileSync(dirs_1.dirs.autoBuildFile, JSON.stringify(autoBuildFile));
}
function setupOpencv() {
    return __awaiter(this, void 0, void 0, function () {
        var msbuild, cMakeFlags, tag, cmakeArgs, rmOpenCV, err_1, rmOpenCVContrib, err_2;
        return __generator(this, function (_a) {
            switch (_a.label) {
                case 0: return [4 /*yield*/, getMsbuildIfWin()
                    // Get cmake flags here to check for CUDA early on instead of the start of the building process
                ];
                case 1:
                    msbuild = _a.sent();
                    cMakeFlags = utils_1.isWin() ? getWinCmakeFlags(msbuild.version) : getSharedCmakeFlags();
                    tag = env_1.opencvVersion();
                    log.info('install', 'installing opencv version %s into directory: %s', tag, dirs_1.dirs.opencvRoot);
                    return [4 /*yield*/, utils_1.exec(getMkDirCmd('opencv'), { cwd: dirs_1.dirs.rootDir })];
                case 2:
                    _a.sent();
                    return [4 /*yield*/, utils_1.exec(getRmDirCmd('build'), { cwd: dirs_1.dirs.opencvRoot })];
                case 3:
                    _a.sent();
                    return [4 /*yield*/, utils_1.exec(getMkDirCmd('build'), { cwd: dirs_1.dirs.opencvRoot })];
                case 4:
                    _a.sent();
                    return [4 /*yield*/, utils_1.exec(getRmDirCmd('opencv'), { cwd: dirs_1.dirs.opencvRoot })];
                case 5:
                    _a.sent();
                    return [4 /*yield*/, utils_1.exec(getRmDirCmd('opencv_contrib'), { cwd: dirs_1.dirs.opencvRoot })];
                case 6:
                    _a.sent();
                    if (!env_1.isWithoutContrib()) return [3 /*break*/, 7];
                    log.info('install', 'skipping download of opencv_contrib since OPENCV4NODEJS_AUTOBUILD_WITHOUT_CONTRIB is set');
                    return [3 /*break*/, 9];
                case 7: return [4 /*yield*/, utils_1.spawn('git', ['clone', '-b', "" + tag, '--single-branch', '--depth', '1', '--progress', constants_1.opencvContribRepoUrl], { cwd: dirs_1.dirs.opencvRoot })];
                case 8:
                    _a.sent();
                    _a.label = 9;
                case 9: return [4 /*yield*/, utils_1.spawn('git', ['clone', '-b', "" + tag, '--single-branch', '--depth', '1', '--progress', constants_1.opencvRepoUrl], { cwd: dirs_1.dirs.opencvRoot })];
                case 10:
                    _a.sent();
                    cmakeArgs = getCmakeArgs(cMakeFlags);
                    log.info('install', 'running cmake %s', cmakeArgs);
                    return [4 /*yield*/, utils_1.spawn('cmake', cmakeArgs, { cwd: dirs_1.dirs.opencvBuild })];
                case 11:
                    _a.sent();
                    log.info('install', 'starting build...');
                    return [4 /*yield*/, getRunBuildCmd(utils_1.isWin() ? msbuild.path : undefined)()];
                case 12:
                    _a.sent();
                    writeAutoBuildFile();
                    rmOpenCV = getRmDirCmd('opencv');
                    _a.label = 13;
                case 13:
                    _a.trys.push([13, 15, , 16]);
                    return [4 /*yield*/, utils_1.exec(rmOpenCV, { cwd: dirs_1.dirs.opencvRoot })];
                case 14:
                    _a.sent();
                    return [3 /*break*/, 16];
                case 15:
                    err_1 = _a.sent();
                    log.error('install', 'failed to clean opencv source folder:', err_1);
                    log.error('install', 'command was: %s', rmOpenCV);
                    log.error('install', 'consider removing the folder yourself: %s', path.join(dirs_1.dirs.opencvRoot, 'opencv'));
                    return [3 /*break*/, 16];
                case 16:
                    rmOpenCVContrib = getRmDirCmd('opencv_contrib');
                    _a.label = 17;
                case 17:
                    _a.trys.push([17, 19, , 20]);
                    return [4 /*yield*/, utils_1.exec(rmOpenCVContrib, { cwd: dirs_1.dirs.opencvRoot })];
                case 18:
                    _a.sent();
                    return [3 /*break*/, 20];
                case 19:
                    err_2 = _a.sent();
                    log.error('install', 'failed to clean opencv_contrib source folder:', err_2);
                    log.error('install', 'command was: %s', rmOpenCV);
                    log.error('install', 'consider removing the folder yourself: %s', path.join(dirs_1.dirs.opencvRoot, 'opencv_contrib'));
                    return [3 /*break*/, 20];
                case 20: return [2 /*return*/];
            }
        });
    });
}
exports.setupOpencv = setupOpencv;
