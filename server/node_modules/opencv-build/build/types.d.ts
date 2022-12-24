export declare type OpencvModule = {
    opencvModule: string;
    libPath: string | undefined;
};
export declare type AutoBuildFile = {
    opencvVersion: string;
    autoBuildFlags: string;
    modules: OpencvModule[];
};
