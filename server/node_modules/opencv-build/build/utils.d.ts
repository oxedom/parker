/// <reference types="node" />
import * as child_process from 'child_process';
export declare function exec(cmd: string, options?: child_process.ExecOptions): Promise<string>;
export declare function execFile(cmd: string, args: string[], options?: child_process.ExecOptions): Promise<string>;
export declare function spawn(cmd: string, args: string[], options?: child_process.ExecOptions): Promise<string>;
export declare function requireGit(): Promise<void>;
export declare function requireCmake(): Promise<void>;
export declare function isWin(): boolean;
export declare function isOSX(): boolean;
export declare function isUnix(): boolean;
export declare function isCudaAvailable(): Promise<boolean>;
