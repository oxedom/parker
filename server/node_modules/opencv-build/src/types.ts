export type OpencvModule = {
  opencvModule: string
  libPath: string | undefined
}

export type AutoBuildFile = {
  opencvVersion: string
  autoBuildFlags: string
  modules: OpencvModule[]
}