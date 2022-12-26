"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
var dirs_1 = require("./dirs");
exports.OPENCV_CONTRIB_ARCHIVE = 'https://github.com/opencv/opencv_contrib/archive/3.4.6.tar.gz';
exports.opencvRepoUrl = 'https://github.com/opencv/opencv.git';
exports.opencvContribRepoUrl = 'https://github.com/opencv/opencv_contrib.git';
exports.opencvModules = [
    'core',
    'highgui',
    'imgcodecs',
    'imgproc',
    'features2d',
    'calib3d',
    'photo',
    'objdetect',
    'ml',
    'video',
    'videoio',
    'videostab',
    'dnn',
    'face',
    'text',
    'tracking',
    'xfeatures2d',
    'ximgproc'
];
exports.cmakeVsCompilers = {
    '10': 'Visual Studio 10 2010',
    '11': 'Visual Studio 11 2012',
    '12': 'Visual Studio 12 2013',
    '14': 'Visual Studio 14 2015',
    '15': 'Visual Studio 15 2017'
};
exports.cmakeArchs = {
    'x64': ' Win64',
    'ia32': '',
    'arm': ' ARM'
};
exports.defaultCmakeFlags = [
    "-DCMAKE_INSTALL_PREFIX=" + dirs_1.dirs.opencvBuild,
    '-DCMAKE_BUILD_TYPE=Release',
    '-DBUILD_EXAMPLES=OFF',
    '-DBUILD_DOCS=OFF',
    '-DBUILD_TESTS=OFF',
    '-DBUILD_PERF_TESTS=OFF',
    '-DBUILD_JAVA=OFF',
    '-DCUDA_NVCC_FLAGS=--expt-relaxed-constexpr',
    '-DBUILD_opencv_apps=OFF',
    '-DBUILD_opencv_aruco=OFF',
    '-DBUILD_opencv_bgsegm=OFF',
    '-DBUILD_opencv_bioinspired=OFF',
    '-DBUILD_opencv_ccalib=OFF',
    '-DBUILD_opencv_datasets=OFF',
    '-DBUILD_opencv_dnn_objdetect=OFF',
    '-DBUILD_opencv_dpm=OFF',
    '-DBUILD_opencv_fuzzy=OFF',
    '-DBUILD_opencv_hfs=OFF',
    '-DBUILD_opencv_java_bindings_generator=OFF',
    '-DBUILD_opencv_js=OFF',
    '-DBUILD_opencv_img_hash=OFF',
    '-DBUILD_opencv_line_descriptor=OFF',
    '-DBUILD_opencv_optflow=OFF',
    '-DBUILD_opencv_phase_unwrapping=OFF',
    '-DBUILD_opencv_python3=OFF',
    '-DBUILD_opencv_python_bindings_generator=OFF',
    '-DBUILD_opencv_reg=OFF',
    '-DBUILD_opencv_rgbd=OFF',
    '-DBUILD_opencv_saliency=OFF',
    '-DBUILD_opencv_shape=OFF',
    '-DBUILD_opencv_stereo=OFF',
    '-DBUILD_opencv_stitching=OFF',
    '-DBUILD_opencv_structured_light=OFF',
    '-DBUILD_opencv_superres=OFF',
    '-DBUILD_opencv_surface_matching=OFF',
    '-DBUILD_opencv_ts=OFF',
    '-DBUILD_opencv_xobjdetect=OFF',
    '-DBUILD_opencv_xphoto=OFF',
    '-DWITH_VTK=OFF'
];
