[![Build Status](https://travis-ci.org/justadudewhohacks/npm-opencv-build.svg?branch=master)](http://travis-ci.org/justadudewhohacks/npm-opencv-build)
[![Build status](https://ci.appveyor.com/api/projects/status/uv8n2sruno95rxtq/branch/master?svg=true)](https://ci.appveyor.com/project/justadudewhohacks/npm-opencv-build/branch/master)

A simple script to auto build recent OpenCV + contrib version via npm. This script is used to auto build <a href="https://github.com/justadudewhohacks/opencv4nodejs"><b>opencv4nodejs</b></a>.

# Install

``` bash
npm install opencv-build
```

## Requirements

- cmake

### Windows

- windows build tools or Visual Studio

``` bash
npm install --global windows-build-tools
```

## Environment Variables

It's possible to specify build environment variables by inserting them into the `package.json` where the dependency is declared an object like:

```json
{
  "opencv4nodejs": {
    "autoBuildFlags": "-DOPENCV_GENERATE_PKGCONFIG=ON -DOPENCV_PC_FILE_NAME=opencv.pc",
    "autoBuildOpencvVersion": "4.1.0"
  }
}
```

The following environment variables can be passed:

- autoBuildBuildCuda
- autoBuildFlags
- autoBuildOpencvVersion
- autoBuildWithoutContrib
- disableAutoBuild
- opencvIncludeDir
- opencvLibDir
- opencvBinDir
