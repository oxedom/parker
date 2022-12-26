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
var path = require('path');
var fs = require('fs');
var log = require('npmlog');
var _a = require('./utils'), exec = _a.exec, execFile = _a.execFile;
/* this codesnippet is partly taken from the node-gyp source: https://github.com/nodejs/node-gyp */
function findVs2017() {
    var ps = path.join(process.env.SystemRoot, 'System32', 'WindowsPowerShell', 'v1.0', 'powershell.exe');
    var args = ['-ExecutionPolicy', 'Unrestricted', '-Command',
        '&{Add-Type -Path \'' + path.join(__dirname, '../Find-VS2017.cs') +
            '\'; [VisualStudioConfiguration.Main]::Query()}'];
    log.silly('find-msbuild', 'find vs2017 via powershell:', ps, args);
    return execFile(ps, args, { encoding: 'utf8' })
        .then(function (stdout) {
        log.silly('find-msbuild', 'find vs2017: ', stdout);
        var vsSetup = JSON.parse(stdout);
        if (!vsSetup || !vsSetup.path || !vsSetup.sdk) {
            return Promise.reject('unexpected powershell output');
        }
        log.silly('find-msbuild', 'found vs2017');
        log.silly('find-msbuild', 'path', vsSetup.path);
        log.silly('find-msbuild', 'sdk', vsSetup.sdk);
        var build = {
            path: path.join(vsSetup.path, 'MSBuild', '15.0', 'Bin', 'MSBuild.exe'),
            version: 15
        };
        log.silly('find-msbuild', 'using following msbuild:');
        log.silly('find-msbuild', 'version:', build.version);
        log.silly('find-msbuild', 'path:', build.path);
        return build;
    });
}
function parseMsBuilds(stdout) {
    var reVers = /ToolsVersions\\([^\\]+)$/i, rePath = /\r\n[ \t]+MSBuildToolsPath[ \t]+REG_SZ[ \t]+([^\r]+)/i, r;
    var msbuilds = [];
    stdout.split('\r\n\r\n').forEach(function (l) {
        if (!l)
            return;
        l = l.trim();
        if (r = reVers.exec(l.substring(0, l.indexOf('\r\n')))) {
            var ver = parseFloat(r[1]);
            if (ver >= 3.5) {
                if (r = rePath.exec(l)) {
                    msbuilds.push({
                        version: ver,
                        path: r[1]
                    });
                }
            }
        }
    });
    return msbuilds;
}
function findMsbuildInRegistry() {
    var cmd = "reg query \"HKLM\\Software\\Microsoft\\MSBuild\\ToolsVersions\" /s" + (process.arch === 'ia32' ? '' : ' /reg:32');
    log.silly('find-msbuild', 'find msbuild in registry:', cmd);
    return exec(cmd)
        .then(function (stdout) {
        log.silly('find-msbuild', 'find vs: ', stdout);
        // use most recent version
        var msbuilds = parseMsBuilds(stdout)
            .sort(function (m1, m2) { return m2.version - m1.version; })
            .map(function (msbuild) { return Object.assign({}, msbuild, { path: path.resolve(msbuild.path, 'msbuild.exe') }); });
        if (!msbuilds.length) {
            return Promise.reject('failed to find msbuild in registry');
        }
        log.info('find-msbuild', 'trying the following msbuild paths:');
        msbuilds.forEach(function (msbuild) {
            log.info('find-msbuild', 'version: %s, path: %s', msbuild.version, msbuild.path);
        });
        var build = msbuilds.find(function (msbuild) {
            try {
                return fs.statSync(msbuild.path);
            }
            catch (err) {
                if (err.code == 'ENOENT') {
                    return false;
                }
                throw err;
            }
        });
        if (!build) {
            return Promise.reject('could not find msbuild.exe from path in registry');
        }
        log.silly('find-msbuild', 'using following msbuild:');
        log.silly('find-msbuild', 'version:', build.version);
        log.silly('find-msbuild', 'path:', build.path);
        return Promise.resolve(build);
    });
}
function findMsBuild() {
    return __awaiter(this, void 0, void 0, function () {
        var err_1;
        return __generator(this, function (_a) {
            switch (_a.label) {
                case 0:
                    _a.trys.push([0, 2, , 4]);
                    return [4 /*yield*/, findVs2017()];
                case 1: return [2 /*return*/, _a.sent()];
                case 2:
                    err_1 = _a.sent();
                    log.info('find-msbuild', 'failed to find vs2017 via powershell:', err_1);
                    log.info('find-msbuild', 'attempting to find msbuild via registry query...');
                    return [4 /*yield*/, findMsbuildInRegistry()];
                case 3: return [2 /*return*/, _a.sent()];
                case 4: return [2 /*return*/];
            }
        });
    });
}
exports.findMsBuild = findMsBuild;
