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
var child_process = require("child_process");
var fs = require("fs");
var path = require("path");
var log = require('npmlog');
function exec(cmd, options) {
    log.silly('install', 'executing:', cmd);
    return new Promise(function (resolve, reject) {
        child_process.exec(cmd, options, function (err, stdout, stderr) {
            var _err = err || stderr;
            if (_err)
                return reject(_err);
            return resolve(stdout.toString());
        });
    });
}
exports.exec = exec;
function execFile(cmd, args, options) {
    log.silly('install', 'executing:', cmd, args);
    return new Promise(function (resolve, reject) {
        var child = child_process.execFile(cmd, args, options, function (err, stdout, stderr) {
            var _err = err || stderr;
            if (_err)
                return reject(_err);
            return resolve(stdout.toString());
        });
        child.stdin && child.stdin.end();
    });
}
exports.execFile = execFile;
function spawn(cmd, args, options) {
    log.silly('install', 'spawning:', cmd, args);
    return new Promise(function (resolve, reject) {
        try {
            var child = child_process.spawn(cmd, args, Object.assign({}, { stdio: 'inherit' }, options));
            child.on('exit', function (code) {
                if (typeof code !== 'number') {
                    code = null;
                }
                var msg = 'child process exited with code ' + code + ' (for more info, set \'--loglevel silly\')';
                if (code !== 0) {
                    return reject(msg);
                }
                return resolve(msg);
            });
        }
        catch (err) {
            return reject(err);
        }
    });
}
exports.spawn = spawn;
function requireCmd(cmd, hint) {
    return __awaiter(this, void 0, void 0, function () {
        var stdout, err_1, errMessage;
        return __generator(this, function (_a) {
            switch (_a.label) {
                case 0:
                    log.info('install', "executing: " + cmd);
                    _a.label = 1;
                case 1:
                    _a.trys.push([1, 3, , 4]);
                    return [4 /*yield*/, exec(cmd)];
                case 2:
                    stdout = _a.sent();
                    log.info('install', cmd + ": " + stdout);
                    return [3 /*break*/, 4];
                case 3:
                    err_1 = _a.sent();
                    errMessage = "failed to execute " + cmd + ", " + hint + ", error is: " + err_1.toString();
                    throw new Error(errMessage);
                case 4: return [2 /*return*/];
            }
        });
    });
}
function requireGit() {
    return __awaiter(this, void 0, void 0, function () {
        return __generator(this, function (_a) {
            switch (_a.label) {
                case 0: return [4 /*yield*/, requireCmd('git --version', 'git is required')];
                case 1:
                    _a.sent();
                    return [2 /*return*/];
            }
        });
    });
}
exports.requireGit = requireGit;
function requireCmake() {
    return __awaiter(this, void 0, void 0, function () {
        return __generator(this, function (_a) {
            switch (_a.label) {
                case 0: return [4 /*yield*/, requireCmd('cmake --version', 'cmake is required to build opencv')];
                case 1:
                    _a.sent();
                    return [2 /*return*/];
            }
        });
    });
}
exports.requireCmake = requireCmake;
function isWin() {
    return process.platform == 'win32';
}
exports.isWin = isWin;
function isOSX() {
    return process.platform == 'darwin';
}
exports.isOSX = isOSX;
function isUnix() {
    return !isWin() && !isOSX();
}
exports.isUnix = isUnix;
function isCudaAvailable() {
    return __awaiter(this, void 0, void 0, function () {
        var err_2, cudaVersionFilePath, content;
        return __generator(this, function (_a) {
            switch (_a.label) {
                case 0:
                    log.info('install', 'Check if CUDA is available & what version...');
                    if (!isWin()) return [3 /*break*/, 4];
                    _a.label = 1;
                case 1:
                    _a.trys.push([1, 3, , 4]);
                    return [4 /*yield*/, requireCmd('nvcc --version', 'CUDA availability check')];
                case 2:
                    _a.sent();
                    return [2 /*return*/, true];
                case 3:
                    err_2 = _a.sent();
                    log.info('install', 'Seems like CUDA is not installed.');
                    return [2 /*return*/, false];
                case 4:
                    cudaVersionFilePath = path.resolve('/usr/local/cuda/version.txt');
                    if (fs.existsSync(cudaVersionFilePath)) {
                        content = fs.readFileSync(cudaVersionFilePath, 'utf8');
                        log.info('install', content);
                        return [2 /*return*/, true];
                    }
                    else {
                        log.info('install', 'CUDA version file could not be found.');
                        return [2 /*return*/, false];
                    }
                    return [2 /*return*/];
            }
        });
    });
}
exports.isCudaAvailable = isCudaAvailable;
