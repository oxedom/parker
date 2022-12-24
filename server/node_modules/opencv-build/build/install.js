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
var constants_1 = require("./constants");
var dirs_1 = require("./dirs");
var env_1 = require("./env");
var getLibsFactory_1 = require("./getLibsFactory");
var setupOpencv_1 = require("./setupOpencv");
var utils_1 = require("./utils");
var log = require('npmlog');
var getLibs = getLibsFactory_1.getLibsFactory({ isWin: utils_1.isWin, isOSX: utils_1.isOSX, opencvModules: constants_1.opencvModules, path: path, fs: fs });
function checkInstalledLibs(autoBuildFile) {
    var hasLibs = true;
    log.info('install', 'checking for opencv libraries');
    if (!fs.existsSync(dirs_1.dirs.opencvLibDir)) {
        log.info('install', 'library dir does not exist:', dirs_1.dirs.opencvLibDir);
        return;
    }
    var installedLibs = getLibs(dirs_1.dirs.opencvLibDir);
    autoBuildFile.modules.forEach(function (_a) {
        var opencvModule = _a.opencvModule, libPath = _a.libPath;
        if (!libPath) {
            log.info('install', '%s: %s', opencvModule, 'ignored');
            return;
        }
        var foundLib = installedLibs.find(function (lib) { return lib.opencvModule === opencvModule; });
        hasLibs = hasLibs && !!foundLib;
        log.info('install', '%s: %s', opencvModule, foundLib ? foundLib.libPath : 'not found');
    });
    return hasLibs;
}
function install() {
    return __awaiter(this, void 0, void 0, function () {
        var autoBuildFile, hasLibs, err_1;
        return __generator(this, function (_a) {
            switch (_a.label) {
                case 0:
                    // if project directory has a package.json containing opencv4nodejs variables
                    // apply these variables to the process environment
                    env_1.applyEnvsFromPackageJson();
                    if (env_1.isAutoBuildDisabled()) {
                        log.info('install', 'OPENCV4NODEJS_DISABLE_AUTOBUILD is set');
                        log.info('install', 'skipping auto build...');
                        return [2 /*return*/];
                    }
                    log.info('install', 'if you want to use an own OpenCV installation set OPENCV4NODEJS_DISABLE_AUTOBUILD');
                    autoBuildFile = env_1.readAutoBuildFile();
                    if (autoBuildFile) {
                        log.info('install', "found auto-build.json: " + dirs_1.dirs.autoBuildFile);
                        if (autoBuildFile.opencvVersion !== env_1.opencvVersion()) {
                            log.info('install', "auto build opencv version is " + autoBuildFile.opencvVersion + ", but OPENCV4NODEJS_AUTOBUILD_OPENCV_VERSION=" + env_1.opencvVersion());
                        }
                        else if (autoBuildFile.autoBuildFlags !== env_1.autoBuildFlags()) {
                            log.info('install', "auto build flags are " + autoBuildFile.autoBuildFlags + ", but OPENCV4NODEJS_AUTOBUILD_FLAGS=" + env_1.autoBuildFlags());
                        }
                        else {
                            hasLibs = checkInstalledLibs(autoBuildFile);
                            if (hasLibs) {
                                log.info('install', 'found all libraries');
                                return [2 /*return*/];
                            }
                            else {
                                log.info('install', 'missing some libraries');
                            }
                        }
                    }
                    else {
                        log.info('install', "failed to find auto-build.json: " + dirs_1.dirs.autoBuildFile);
                    }
                    log.info('install', '');
                    log.info('install', 'running install script...');
                    log.info('install', '');
                    log.info('install', 'opencv version: %s', env_1.opencvVersion());
                    log.info('install', 'with opencv contrib: %s', env_1.isWithoutContrib() ? 'no' : 'yes');
                    log.info('install', 'custom build flags: %s', env_1.autoBuildFlags());
                    log.info('install', '');
                    _a.label = 1;
                case 1:
                    _a.trys.push([1, 5, , 6]);
                    return [4 /*yield*/, utils_1.requireGit()];
                case 2:
                    _a.sent();
                    return [4 /*yield*/, utils_1.requireCmake()];
                case 3:
                    _a.sent();
                    return [4 /*yield*/, setupOpencv_1.setupOpencv()];
                case 4:
                    _a.sent();
                    return [3 /*break*/, 6];
                case 5:
                    err_1 = _a.sent();
                    log.error(err_1);
                    process.exit(1);
                    return [3 /*break*/, 6];
                case 6: return [2 /*return*/];
            }
        });
    });
}
exports.install = install;
