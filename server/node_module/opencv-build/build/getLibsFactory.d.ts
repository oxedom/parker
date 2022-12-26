import { OpencvModule } from './types';
export declare function getLibsFactory(args: {
    opencvModules: string[];
    isWin: () => boolean;
    isOSX: () => boolean;
    fs: any;
    path: any;
}): (libDir: string) => OpencvModule[];
