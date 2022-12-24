cmd_Release/obj.target/opencv4nodejs/cc/opencv4nodejs.o := g++ -o Release/obj.target/opencv4nodejs/cc/opencv4nodejs.o ../cc/opencv4nodejs.cc '-DNODE_GYP_MODULE_NAME=opencv4nodejs' '-DUSING_UV_SHARED=1' '-DUSING_V8_SHARED=1' '-DV8_DEPRECATION_WARNINGS=1' '-DV8_DEPRECATION_WARNINGS' '-DV8_IMMINENT_DEPRECATION_WARNINGS' '-D_GLIBCXX_USE_CXX11_ABI=1' '-D_LARGEFILE_SOURCE' '-D_FILE_OFFSET_BITS=64' '-D__STDC_FORMAT_MACROS' '-DOPENSSL_NO_PINSHARED' '-DOPENSSL_THREADS' '-DOPENCV4NODEJS_FOUND_LIBRARY_CORE' '-DOPENCV4NODEJS_FOUND_LIBRARY_HIGHGUI' '-DOPENCV4NODEJS_FOUND_LIBRARY_IMGCODECS' '-DOPENCV4NODEJS_FOUND_LIBRARY_IMGPROC' '-DOPENCV4NODEJS_FOUND_LIBRARY_FEATURES2D' '-DOPENCV4NODEJS_FOUND_LIBRARY_CALIB3D' '-DOPENCV4NODEJS_FOUND_LIBRARY_PHOTO' '-DOPENCV4NODEJS_FOUND_LIBRARY_OBJDETECT' '-DOPENCV4NODEJS_FOUND_LIBRARY_ML' '-DOPENCV4NODEJS_FOUND_LIBRARY_VIDEO' '-DOPENCV4NODEJS_FOUND_LIBRARY_VIDEOIO' '-DOPENCV4NODEJS_FOUND_LIBRARY_VIDEOSTAB' '-DOPENCV4NODEJS_FOUND_LIBRARY_DNN' '-DOPENCV4NODEJS_FOUND_LIBRARY_FACE' '-DOPENCV4NODEJS_FOUND_LIBRARY_TEXT' '-DOPENCV4NODEJS_FOUND_LIBRARY_TRACKING' '-DOPENCV4NODEJS_FOUND_LIBRARY_XFEATURES2D' '-DOPENCV4NODEJS_FOUND_LIBRARY_XIMGPROC' '-DBUILDING_NODE_EXTENSION' -I/home/sam/.cache/node-gyp/18.12.1/include/node -I/home/sam/.cache/node-gyp/18.12.1/src -I/home/sam/.cache/node-gyp/18.12.1/deps/openssl/config -I/home/sam/.cache/node-gyp/18.12.1/deps/openssl/openssl/include -I/home/sam/.cache/node-gyp/18.12.1/deps/uv/include -I/home/sam/.cache/node-gyp/18.12.1/deps/zlib -I/home/sam/.cache/node-gyp/18.12.1/deps/v8/include -I/home/sam/projects/parker/server/node_modules/opencv-build/opencv/build/include -I/home/sam/projects/parker/server/node_modules/opencv-build/opencv/build/include/opencv4 -I../cc -I../cc/core -I../../nan -I/home/sam/projects/parker/server/node_modules/native-node-utils/src  -fPIC -pthread -Wall -Wextra -Wno-unused-parameter -m64 -std=c++11 -O3 -fno-omit-frame-pointer -std=gnu++17 -MMD -MF ./Release/.deps/Release/obj.target/opencv4nodejs/cc/opencv4nodejs.o.d.raw   -c
Release/obj.target/opencv4nodejs/cc/opencv4nodejs.o: \
 ../cc/opencv4nodejs.cc \
 /home/sam/.cache/node-gyp/18.12.1/include/node/node.h \
 /home/sam/.cache/node-gyp/18.12.1/include/node/v8.h \
 /home/sam/.cache/node-gyp/18.12.1/include/node/cppgc/common.h \
 /home/sam/.cache/node-gyp/18.12.1/include/node/v8config.h \
 /home/sam/.cache/node-gyp/18.12.1/include/node/v8-array-buffer.h \
 /home/sam/.cache/node-gyp/18.12.1/include/node/v8-local-handle.h \
 /home/sam/.cache/node-gyp/18.12.1/include/node/v8-internal.h \
 /home/sam/.cache/node-gyp/18.12.1/include/node/v8-version.h \
 /home/sam/.cache/node-gyp/18.12.1/include/node/v8config.h \
 /home/sam/.cache/node-gyp/18.12.1/include/node/v8-object.h \
 /home/sam/.cache/node-gyp/18.12.1/include/node/v8-maybe.h \
 /home/sam/.cache/node-gyp/18.12.1/include/node/v8-persistent-handle.h \
 /home/sam/.cache/node-gyp/18.12.1/include/node/v8-weak-callback-info.h \
 /home/sam/.cache/node-gyp/18.12.1/include/node/v8-primitive.h \
 /home/sam/.cache/node-gyp/18.12.1/include/node/v8-data.h \
 /home/sam/.cache/node-gyp/18.12.1/include/node/v8-value.h \
 /home/sam/.cache/node-gyp/18.12.1/include/node/v8-traced-handle.h \
 /home/sam/.cache/node-gyp/18.12.1/include/node/v8-container.h \
 /home/sam/.cache/node-gyp/18.12.1/include/node/v8-context.h \
 /home/sam/.cache/node-gyp/18.12.1/include/node/v8-snapshot.h \
 /home/sam/.cache/node-gyp/18.12.1/include/node/v8-date.h \
 /home/sam/.cache/node-gyp/18.12.1/include/node/v8-debug.h \
 /home/sam/.cache/node-gyp/18.12.1/include/node/v8-script.h \
 /home/sam/.cache/node-gyp/18.12.1/include/node/v8-message.h \
 /home/sam/.cache/node-gyp/18.12.1/include/node/v8-exception.h \
 /home/sam/.cache/node-gyp/18.12.1/include/node/v8-extension.h \
 /home/sam/.cache/node-gyp/18.12.1/include/node/v8-external.h \
 /home/sam/.cache/node-gyp/18.12.1/include/node/v8-function.h \
 /home/sam/.cache/node-gyp/18.12.1/include/node/v8-function-callback.h \
 /home/sam/.cache/node-gyp/18.12.1/include/node/v8-template.h \
 /home/sam/.cache/node-gyp/18.12.1/include/node/v8-memory-span.h \
 /home/sam/.cache/node-gyp/18.12.1/include/node/v8-initialization.h \
 /home/sam/.cache/node-gyp/18.12.1/include/node/v8-callbacks.h \
 /home/sam/.cache/node-gyp/18.12.1/include/node/v8-isolate.h \
 /home/sam/.cache/node-gyp/18.12.1/include/node/v8-embedder-heap.h \
 /home/sam/.cache/node-gyp/18.12.1/include/node/v8-microtask.h \
 /home/sam/.cache/node-gyp/18.12.1/include/node/v8-statistics.h \
 /home/sam/.cache/node-gyp/18.12.1/include/node/v8-promise.h \
 /home/sam/.cache/node-gyp/18.12.1/include/node/v8-unwinder.h \
 /home/sam/.cache/node-gyp/18.12.1/include/node/v8-embedder-state-scope.h \
 /home/sam/.cache/node-gyp/18.12.1/include/node/v8-platform.h \
 /home/sam/.cache/node-gyp/18.12.1/include/node/v8-json.h \
 /home/sam/.cache/node-gyp/18.12.1/include/node/v8-locker.h \
 /home/sam/.cache/node-gyp/18.12.1/include/node/v8-microtask-queue.h \
 /home/sam/.cache/node-gyp/18.12.1/include/node/v8-primitive-object.h \
 /home/sam/.cache/node-gyp/18.12.1/include/node/v8-proxy.h \
 /home/sam/.cache/node-gyp/18.12.1/include/node/v8-regexp.h \
 /home/sam/.cache/node-gyp/18.12.1/include/node/v8-typed-array.h \
 /home/sam/.cache/node-gyp/18.12.1/include/node/v8-value-serializer.h \
 /home/sam/.cache/node-gyp/18.12.1/include/node/v8-wasm.h \
 /home/sam/.cache/node-gyp/18.12.1/include/node/node_version.h \
 ../cc/ExternalMemTracking.h ../cc/macros.h \
 /home/sam/projects/parker/server/node_modules/native-node-utils/src/NativeNodeUtils.h \
 /home/sam/projects/parker/server/node_modules/native-node-utils/src/AbstractConverter.h \
 /home/sam/projects/parker/server/node_modules/native-node-utils/src/utils.h \
 ../../nan/nan.h \
 /home/sam/.cache/node-gyp/18.12.1/include/node/node_version.h \
 /home/sam/.cache/node-gyp/18.12.1/include/node/uv.h \
 /home/sam/.cache/node-gyp/18.12.1/include/node/uv/errno.h \
 /home/sam/.cache/node-gyp/18.12.1/include/node/uv/version.h \
 /home/sam/.cache/node-gyp/18.12.1/include/node/uv/unix.h \
 /home/sam/.cache/node-gyp/18.12.1/include/node/uv/threadpool.h \
 /home/sam/.cache/node-gyp/18.12.1/include/node/uv/linux.h \
 /home/sam/.cache/node-gyp/18.12.1/include/node/node_buffer.h \
 /home/sam/.cache/node-gyp/18.12.1/include/node/node.h \
 /home/sam/.cache/node-gyp/18.12.1/include/node/node_object_wrap.h \
 ../../nan/nan_callbacks.h ../../nan/nan_callbacks_12_inl.h \
 ../../nan/nan_maybe_43_inl.h ../../nan/nan_converters.h \
 ../../nan/nan_converters_43_inl.h ../../nan/nan_new.h \
 ../../nan/nan_implementation_12_inl.h ../../nan/nan_persistent_12_inl.h \
 ../../nan/nan_weak.h ../../nan/nan_object_wrap.h ../../nan/nan_private.h \
 ../../nan/nan_typedarray_contents.h ../../nan/nan_json.h \
 ../../nan/nan_scriptorigin.h \
 /home/sam/projects/parker/server/node_modules/native-node-utils/src/ArrayConverter.h \
 /home/sam/projects/parker/server/node_modules/native-node-utils/src/ArrayOfArraysConverter.h \
 /home/sam/projects/parker/server/node_modules/native-node-utils/src/AsyncWorker.h \
 /home/sam/projects/parker/server/node_modules/native-node-utils/src/IWorker.h \
 /home/sam/projects/parker/server/node_modules/native-node-utils/src/Binding.h \
 /home/sam/projects/parker/server/node_modules/native-node-utils/src/IWorker.h \
 /home/sam/projects/parker/server/node_modules/native-node-utils/src/TryCatch.h \
 /home/sam/projects/parker/server/node_modules/native-node-utils/src/Converters.h \
 /home/sam/projects/parker/server/node_modules/native-node-utils/src/PrimitiveTypeConverters.h \
 /home/sam/projects/parker/server/node_modules/native-node-utils/src/UnwrapperBase.h \
 /home/sam/projects/parker/server/node_modules/native-node-utils/src/BindingBase.h \
 /home/sam/projects/parker/server/node_modules/native-node-utils/src/InstanceConverter.h \
 /home/sam/projects/parker/server/node_modules/native-node-utils/src/ObjectWrap.h \
 /home/sam/projects/parker/server/node_modules/opencv-build/opencv/build/include/opencv2/core.hpp \
 /home/sam/projects/parker/server/node_modules/opencv-build/opencv/build/include/opencv2/core/cvdef.h \
 /home/sam/projects/parker/server/node_modules/opencv-build/opencv/build/include/opencv2/core/hal/interface.h \
 /home/sam/projects/parker/server/node_modules/opencv-build/opencv/build/include/opencv2/core/cv_cpu_dispatch.h \
 /home/sam/projects/parker/server/node_modules/opencv-build/opencv/build/include/opencv2/core/version.hpp \
 /home/sam/projects/parker/server/node_modules/opencv-build/opencv/build/include/opencv2/core/base.hpp \
 /home/sam/projects/parker/server/node_modules/opencv-build/opencv/build/include/opencv2/opencv_modules.hpp \
 /home/sam/projects/parker/server/node_modules/opencv-build/opencv/build/include/opencv2/core/cvstd.hpp \
 /home/sam/projects/parker/server/node_modules/opencv-build/opencv/build/include/opencv2/core/ptr.inl.hpp \
 /home/sam/projects/parker/server/node_modules/opencv-build/opencv/build/include/opencv2/core/neon_utils.hpp \
 /home/sam/projects/parker/server/node_modules/opencv-build/opencv/build/include/opencv2/core/vsx_utils.hpp \
 /home/sam/projects/parker/server/node_modules/opencv-build/opencv/build/include/opencv2/core/check.hpp \
 /home/sam/projects/parker/server/node_modules/opencv-build/opencv/build/include/opencv2/core/traits.hpp \
 /home/sam/projects/parker/server/node_modules/opencv-build/opencv/build/include/opencv2/core/matx.hpp \
 /home/sam/projects/parker/server/node_modules/opencv-build/opencv/build/include/opencv2/core/saturate.hpp \
 /home/sam/projects/parker/server/node_modules/opencv-build/opencv/build/include/opencv2/core/fast_math.hpp \
 /home/sam/projects/parker/server/node_modules/opencv-build/opencv/build/include/opencv2/core/types.hpp \
 /home/sam/projects/parker/server/node_modules/opencv-build/opencv/build/include/opencv2/core/mat.hpp \
 /home/sam/projects/parker/server/node_modules/opencv-build/opencv/build/include/opencv2/core/bufferpool.hpp \
 /home/sam/projects/parker/server/node_modules/opencv-build/opencv/build/include/opencv2/core/mat.inl.hpp \
 /home/sam/projects/parker/server/node_modules/opencv-build/opencv/build/include/opencv2/core/persistence.hpp \
 /home/sam/projects/parker/server/node_modules/opencv-build/opencv/build/include/opencv2/core/operations.hpp \
 /home/sam/projects/parker/server/node_modules/opencv-build/opencv/build/include/opencv2/core/cvstd.inl.hpp \
 /home/sam/projects/parker/server/node_modules/opencv-build/opencv/build/include/opencv2/core/utility.hpp \
 /home/sam/projects/parker/server/node_modules/opencv-build/opencv/build/include/opencv2/core/core_c.h \
 /home/sam/projects/parker/server/node_modules/opencv-build/opencv/build/include/opencv2/core/types_c.h \
 /home/sam/projects/parker/server/node_modules/opencv-build/opencv/build/include/opencv2/core/optim.hpp \
 /home/sam/projects/parker/server/node_modules/opencv-build/opencv/build/include/opencv2/core/ovx.hpp \
 /home/sam/projects/parker/server/node_modules/opencv-build/opencv/build/include/opencv2/core/cvdef.h \
 ../cc/CustomMatAllocator.h ../cc/core/Size.h ../cc/macros.h \
 ../cc/core/coreUtils.h ../cc/core/matUtils.h ../cc/core/Vec.h \
 ../cc/core/Vec2.h ../cc/core/coreUtils.h ../cc/core/Vec3.h \
 ../cc/core/Vec4.h ../cc/core/Vec6.h ../cc/core/Point2.h \
 ../cc/core/Rect.h ../cc/CatchCvExceptionWorker.h ../cc/core/Size.h \
 ../cc/core/RotatedRect.h ../cc/core/Point.h ../cc/core/Point2.h \
 ../cc/core/Point3.h ../cc/core/Rect.h ../cc/opencv_modules.h \
 ../cc/core/core.h ../cc/core/Mat.h ../cc/core/matUtils.h \
 ../cc/core/Vec.h ../cc/core/RotatedRect.h ../cc/ExternalMemTracking.h \
 ../cc/core/TermCriteria.h ../cc/core/HistAxes.h ../cc/calib3d/calib3d.h \
 ../cc/calib3d/calib3dBindings.h \
 /home/sam/projects/parker/server/node_modules/opencv-build/opencv/build/include/opencv2/calib3d.hpp \
 /home/sam/projects/parker/server/node_modules/opencv-build/opencv/build/include/opencv2/features2d.hpp \
 /home/sam/projects/parker/server/node_modules/opencv-build/opencv/build/include/opencv2/flann/miniflann.hpp \
 /home/sam/projects/parker/server/node_modules/opencv-build/opencv/build/include/opencv2/flann/defines.h \
 /home/sam/projects/parker/server/node_modules/opencv-build/opencv/build/include/opencv2/flann/config.h \
 /home/sam/projects/parker/server/node_modules/opencv-build/opencv/build/include/opencv2/core/affine.hpp \
 /home/sam/projects/parker/server/node_modules/opencv-build/opencv/build/include/opencv2/calib3d/calib3d_c.h \
 ../cc/core/Mat.h ../cc/core/Point.h ../cc/core/TermCriteria.h \
 ../cc/CvBinding.h ../cc/dnn/dnn.h \
 /home/sam/projects/parker/server/node_modules/opencv-build/opencv/build/include/opencv2/dnn.hpp \
 /home/sam/projects/parker/server/node_modules/opencv-build/opencv/build/include/opencv2/dnn/dnn.hpp \
 /home/sam/projects/parker/server/node_modules/opencv-build/opencv/build/include/opencv2/dnn/dict.hpp \
 /home/sam/projects/parker/server/node_modules/opencv-build/opencv/build/include/opencv2/dnn/layer.hpp \
 /home/sam/projects/parker/server/node_modules/opencv-build/opencv/build/include/opencv2/dnn/dnn.inl.hpp \
 /home/sam/projects/parker/server/node_modules/opencv-build/opencv/build/include/opencv2/dnn/utils/inference_engine.hpp \
 /home/sam/projects/parker/server/node_modules/opencv-build/opencv/build/include/opencv2/dnn/utils/../dnn.hpp \
 ../cc/dnn/Net.h ../cc/face/face.h \
 /home/sam/projects/parker/server/node_modules/opencv-build/opencv/build/include/opencv2/face.hpp \
 /home/sam/projects/parker/server/node_modules/opencv-build/opencv/build/include/opencv2/face/predict_collector.hpp \
 /home/sam/projects/parker/server/node_modules/opencv-build/opencv/build/include/opencv2/face/facerec.hpp \
 /home/sam/projects/parker/server/node_modules/opencv-build/opencv/build/include/opencv2/face/facemark.hpp \
 /home/sam/projects/parker/server/node_modules/opencv-build/opencv/build/include/opencv2/face/facemark_train.hpp \
 /home/sam/projects/parker/server/node_modules/opencv-build/opencv/build/include/opencv2/objdetect.hpp \
 /home/sam/projects/parker/server/node_modules/opencv-build/opencv/build/include/opencv2/objdetect/detection_based_tracker.hpp \
 /home/sam/projects/parker/server/node_modules/opencv-build/opencv/build/include/opencv2/objdetect/objdetect_c.h \
 /home/sam/projects/parker/server/node_modules/opencv-build/opencv/build/include/opencv2/face/facemarkLBF.hpp \
 /home/sam/projects/parker/server/node_modules/opencv-build/opencv/build/include/opencv2/face/facemarkAAM.hpp \
 /home/sam/projects/parker/server/node_modules/opencv-build/opencv/build/include/opencv2/face/face_alignment.hpp \
 /home/sam/projects/parker/server/node_modules/opencv-build/opencv/build/include/opencv2/face/mace.hpp \
 ../cc/features2d/features2d.h ../cc/imgproc/imgproc.h \
 /home/sam/projects/parker/server/node_modules/opencv-build/opencv/build/include/opencv2/imgproc.hpp \
 /home/sam/projects/parker/server/node_modules/opencv-build/opencv/build/include/opencv2/imgproc/imgproc_c.h \
 /home/sam/projects/parker/server/node_modules/opencv-build/opencv/build/include/opencv2/imgproc/types_c.h \
 ../cc/imgproc/Contour.h ../cc/imgproc/Moments.h ../cc/core/HistAxes.h \
 ../cc/io/io.h \
 /home/sam/projects/parker/server/node_modules/opencv-build/opencv/build/include/opencv2/highgui.hpp \
 /home/sam/projects/parker/server/node_modules/opencv-build/opencv/build/include/opencv2/imgcodecs.hpp \
 /home/sam/projects/parker/server/node_modules/opencv-build/opencv/build/include/opencv2/videoio.hpp \
 /home/sam/projects/parker/server/node_modules/opencv-build/opencv/build/include/opencv2/highgui/highgui_c.h \
 /home/sam/projects/parker/server/node_modules/opencv-build/opencv/build/include/opencv2/imgcodecs/imgcodecs_c.h \
 /home/sam/projects/parker/server/node_modules/opencv-build/opencv/build/include/opencv2/videoio/videoio_c.h \
 ../cc/io/VideoCapture.h ../cc/io/VideoWriter.h \
 ../cc/machinelearning/machinelearning.h \
 /home/sam/projects/parker/server/node_modules/opencv-build/opencv/build/include/opencv2/ml.hpp \
 /home/sam/projects/parker/server/node_modules/opencv-build/opencv/build/include/opencv2/ml/ml.inl.hpp \
 ../cc/objdetect/objdetect.h ../cc/photo/photo.h \
 /home/sam/projects/parker/server/node_modules/opencv-build/opencv/build/include/opencv2/photo.hpp \
 /home/sam/projects/parker/server/node_modules/opencv-build/opencv/build/include/opencv2/photo/photo_c.h \
 ../cc/text/text.h ../cc/text/OCRHMMClassifier.h \
 /home/sam/projects/parker/server/node_modules/opencv-build/opencv/build/include/opencv2/text.hpp \
 /home/sam/projects/parker/server/node_modules/opencv-build/opencv/build/include/opencv2/text/erfilter.hpp \
 /home/sam/projects/parker/server/node_modules/opencv-build/opencv/build/include/opencv2/text/ocr.hpp \
 /home/sam/projects/parker/server/node_modules/opencv-build/opencv/build/include/opencv2/text/textDetector.hpp \
 /home/sam/projects/parker/server/node_modules/opencv-build/opencv/build/include/opencv2/text/ocr.hpp \
 ../cc/text/OCRHMMDecoder.h ../cc/tracking/tracking.h \
 /home/sam/projects/parker/server/node_modules/opencv-build/opencv/build/include/opencv2/tracking.hpp \
 /home/sam/projects/parker/server/node_modules/opencv-build/opencv/build/include/opencv2/tracking/tracker.hpp \
 /home/sam/projects/parker/server/node_modules/opencv-build/opencv/build/include/opencv2/tracking/feature.hpp \
 /home/sam/projects/parker/server/node_modules/opencv-build/opencv/build/include/opencv2/tracking/onlineMIL.hpp \
 /home/sam/projects/parker/server/node_modules/opencv-build/opencv/build/include/opencv2/tracking/onlineBoosting.hpp \
 /home/sam/projects/parker/server/node_modules/opencv-build/opencv/build/include/opencv2/tracking/tldDataset.hpp \
 ../cc/video/video.h \
 /home/sam/projects/parker/server/node_modules/opencv-build/opencv/build/include/opencv2/video.hpp \
 /home/sam/projects/parker/server/node_modules/opencv-build/opencv/build/include/opencv2/video/tracking.hpp \
 /home/sam/projects/parker/server/node_modules/opencv-build/opencv/build/include/opencv2/video/background_segm.hpp \
 /home/sam/projects/parker/server/node_modules/opencv-build/opencv/build/include/opencv2/video/tracking_c.h \
 ../cc/xfeatures2d/xfeatures2d.h ../cc/ximgproc/ximgproc.h \
 /home/sam/projects/parker/server/node_modules/opencv-build/opencv/build/include/opencv2/ximgproc.hpp \
 /home/sam/projects/parker/server/node_modules/opencv-build/opencv/build/include/opencv2/ximgproc/edge_filter.hpp \
 /home/sam/projects/parker/server/node_modules/opencv-build/opencv/build/include/opencv2/ximgproc/disparity_filter.hpp \
 /home/sam/projects/parker/server/node_modules/opencv-build/opencv/build/include/opencv2/ximgproc/sparse_match_interpolator.hpp \
 /home/sam/projects/parker/server/node_modules/opencv-build/opencv/build/include/opencv2/ximgproc/structured_edge_detection.hpp \
 /home/sam/projects/parker/server/node_modules/opencv-build/opencv/build/include/opencv2/ximgproc/edgeboxes.hpp \
 /home/sam/projects/parker/server/node_modules/opencv-build/opencv/build/include/opencv2/ximgproc/seeds.hpp \
 /home/sam/projects/parker/server/node_modules/opencv-build/opencv/build/include/opencv2/ximgproc/segmentation.hpp \
 /home/sam/projects/parker/server/node_modules/opencv-build/opencv/build/include/opencv2/ximgproc/fast_hough_transform.hpp \
 /home/sam/projects/parker/server/node_modules/opencv-build/opencv/build/include/opencv2/ximgproc/estimated_covariance.hpp \
 /home/sam/projects/parker/server/node_modules/opencv-build/opencv/build/include/opencv2/ximgproc/weighted_median_filter.hpp \
 /home/sam/projects/parker/server/node_modules/opencv-build/opencv/build/include/opencv2/ximgproc/slic.hpp \
 /home/sam/projects/parker/server/node_modules/opencv-build/opencv/build/include/opencv2/ximgproc/lsc.hpp \
 /home/sam/projects/parker/server/node_modules/opencv-build/opencv/build/include/opencv2/ximgproc/paillou_filter.hpp \
 /home/sam/projects/parker/server/node_modules/opencv-build/opencv/build/include/opencv2/ximgproc/fast_line_detector.hpp \
 /home/sam/projects/parker/server/node_modules/opencv-build/opencv/build/include/opencv2/ximgproc/deriche_filter.hpp \
 /home/sam/projects/parker/server/node_modules/opencv-build/opencv/build/include/opencv2/ximgproc/peilin.hpp \
 /home/sam/projects/parker/server/node_modules/opencv-build/opencv/build/include/opencv2/ximgproc/fourier_descriptors.hpp \
 /home/sam/projects/parker/server/node_modules/opencv-build/opencv/build/include/opencv2/ximgproc/ridgefilter.hpp \
 /home/sam/projects/parker/server/node_modules/opencv-build/opencv/build/include/opencv2/ximgproc/brightedges.hpp
../cc/opencv4nodejs.cc:
/home/sam/.cache/node-gyp/18.12.1/include/node/node.h:
/home/sam/.cache/node-gyp/18.12.1/include/node/v8.h:
/home/sam/.cache/node-gyp/18.12.1/include/node/cppgc/common.h:
/home/sam/.cache/node-gyp/18.12.1/include/node/v8config.h:
/home/sam/.cache/node-gyp/18.12.1/include/node/v8-array-buffer.h:
/home/sam/.cache/node-gyp/18.12.1/include/node/v8-local-handle.h:
/home/sam/.cache/node-gyp/18.12.1/include/node/v8-internal.h:
/home/sam/.cache/node-gyp/18.12.1/include/node/v8-version.h:
/home/sam/.cache/node-gyp/18.12.1/include/node/v8config.h:
/home/sam/.cache/node-gyp/18.12.1/include/node/v8-object.h:
/home/sam/.cache/node-gyp/18.12.1/include/node/v8-maybe.h:
/home/sam/.cache/node-gyp/18.12.1/include/node/v8-persistent-handle.h:
/home/sam/.cache/node-gyp/18.12.1/include/node/v8-weak-callback-info.h:
/home/sam/.cache/node-gyp/18.12.1/include/node/v8-primitive.h:
/home/sam/.cache/node-gyp/18.12.1/include/node/v8-data.h:
/home/sam/.cache/node-gyp/18.12.1/include/node/v8-value.h:
/home/sam/.cache/node-gyp/18.12.1/include/node/v8-traced-handle.h:
/home/sam/.cache/node-gyp/18.12.1/include/node/v8-container.h:
/home/sam/.cache/node-gyp/18.12.1/include/node/v8-context.h:
/home/sam/.cache/node-gyp/18.12.1/include/node/v8-snapshot.h:
/home/sam/.cache/node-gyp/18.12.1/include/node/v8-date.h:
/home/sam/.cache/node-gyp/18.12.1/include/node/v8-debug.h:
/home/sam/.cache/node-gyp/18.12.1/include/node/v8-script.h:
/home/sam/.cache/node-gyp/18.12.1/include/node/v8-message.h:
/home/sam/.cache/node-gyp/18.12.1/include/node/v8-exception.h:
/home/sam/.cache/node-gyp/18.12.1/include/node/v8-extension.h:
/home/sam/.cache/node-gyp/18.12.1/include/node/v8-external.h:
/home/sam/.cache/node-gyp/18.12.1/include/node/v8-function.h:
/home/sam/.cache/node-gyp/18.12.1/include/node/v8-function-callback.h:
/home/sam/.cache/node-gyp/18.12.1/include/node/v8-template.h:
/home/sam/.cache/node-gyp/18.12.1/include/node/v8-memory-span.h:
/home/sam/.cache/node-gyp/18.12.1/include/node/v8-initialization.h:
/home/sam/.cache/node-gyp/18.12.1/include/node/v8-callbacks.h:
/home/sam/.cache/node-gyp/18.12.1/include/node/v8-isolate.h:
/home/sam/.cache/node-gyp/18.12.1/include/node/v8-embedder-heap.h:
/home/sam/.cache/node-gyp/18.12.1/include/node/v8-microtask.h:
/home/sam/.cache/node-gyp/18.12.1/include/node/v8-statistics.h:
/home/sam/.cache/node-gyp/18.12.1/include/node/v8-promise.h:
/home/sam/.cache/node-gyp/18.12.1/include/node/v8-unwinder.h:
/home/sam/.cache/node-gyp/18.12.1/include/node/v8-embedder-state-scope.h:
/home/sam/.cache/node-gyp/18.12.1/include/node/v8-platform.h:
/home/sam/.cache/node-gyp/18.12.1/include/node/v8-json.h:
/home/sam/.cache/node-gyp/18.12.1/include/node/v8-locker.h:
/home/sam/.cache/node-gyp/18.12.1/include/node/v8-microtask-queue.h:
/home/sam/.cache/node-gyp/18.12.1/include/node/v8-primitive-object.h:
/home/sam/.cache/node-gyp/18.12.1/include/node/v8-proxy.h:
/home/sam/.cache/node-gyp/18.12.1/include/node/v8-regexp.h:
/home/sam/.cache/node-gyp/18.12.1/include/node/v8-typed-array.h:
/home/sam/.cache/node-gyp/18.12.1/include/node/v8-value-serializer.h:
/home/sam/.cache/node-gyp/18.12.1/include/node/v8-wasm.h:
/home/sam/.cache/node-gyp/18.12.1/include/node/node_version.h:
../cc/ExternalMemTracking.h:
../cc/macros.h:
/home/sam/projects/parker/server/node_modules/native-node-utils/src/NativeNodeUtils.h:
/home/sam/projects/parker/server/node_modules/native-node-utils/src/AbstractConverter.h:
/home/sam/projects/parker/server/node_modules/native-node-utils/src/utils.h:
../../nan/nan.h:
/home/sam/.cache/node-gyp/18.12.1/include/node/node_version.h:
/home/sam/.cache/node-gyp/18.12.1/include/node/uv.h:
/home/sam/.cache/node-gyp/18.12.1/include/node/uv/errno.h:
/home/sam/.cache/node-gyp/18.12.1/include/node/uv/version.h:
/home/sam/.cache/node-gyp/18.12.1/include/node/uv/unix.h:
/home/sam/.cache/node-gyp/18.12.1/include/node/uv/threadpool.h:
/home/sam/.cache/node-gyp/18.12.1/include/node/uv/linux.h:
/home/sam/.cache/node-gyp/18.12.1/include/node/node_buffer.h:
/home/sam/.cache/node-gyp/18.12.1/include/node/node.h:
/home/sam/.cache/node-gyp/18.12.1/include/node/node_object_wrap.h:
../../nan/nan_callbacks.h:
../../nan/nan_callbacks_12_inl.h:
../../nan/nan_maybe_43_inl.h:
../../nan/nan_converters.h:
../../nan/nan_converters_43_inl.h:
../../nan/nan_new.h:
../../nan/nan_implementation_12_inl.h:
../../nan/nan_persistent_12_inl.h:
../../nan/nan_weak.h:
../../nan/nan_object_wrap.h:
../../nan/nan_private.h:
../../nan/nan_typedarray_contents.h:
../../nan/nan_json.h:
../../nan/nan_scriptorigin.h:
/home/sam/projects/parker/server/node_modules/native-node-utils/src/ArrayConverter.h:
/home/sam/projects/parker/server/node_modules/native-node-utils/src/ArrayOfArraysConverter.h:
/home/sam/projects/parker/server/node_modules/native-node-utils/src/AsyncWorker.h:
/home/sam/projects/parker/server/node_modules/native-node-utils/src/IWorker.h:
/home/sam/projects/parker/server/node_modules/native-node-utils/src/Binding.h:
/home/sam/projects/parker/server/node_modules/native-node-utils/src/IWorker.h:
/home/sam/projects/parker/server/node_modules/native-node-utils/src/TryCatch.h:
/home/sam/projects/parker/server/node_modules/native-node-utils/src/Converters.h:
/home/sam/projects/parker/server/node_modules/native-node-utils/src/PrimitiveTypeConverters.h:
/home/sam/projects/parker/server/node_modules/native-node-utils/src/UnwrapperBase.h:
/home/sam/projects/parker/server/node_modules/native-node-utils/src/BindingBase.h:
/home/sam/projects/parker/server/node_modules/native-node-utils/src/InstanceConverter.h:
/home/sam/projects/parker/server/node_modules/native-node-utils/src/ObjectWrap.h:
/home/sam/projects/parker/server/node_modules/opencv-build/opencv/build/include/opencv2/core.hpp:
/home/sam/projects/parker/server/node_modules/opencv-build/opencv/build/include/opencv2/core/cvdef.h:
/home/sam/projects/parker/server/node_modules/opencv-build/opencv/build/include/opencv2/core/hal/interface.h:
/home/sam/projects/parker/server/node_modules/opencv-build/opencv/build/include/opencv2/core/cv_cpu_dispatch.h:
/home/sam/projects/parker/server/node_modules/opencv-build/opencv/build/include/opencv2/core/version.hpp:
/home/sam/projects/parker/server/node_modules/opencv-build/opencv/build/include/opencv2/core/base.hpp:
/home/sam/projects/parker/server/node_modules/opencv-build/opencv/build/include/opencv2/opencv_modules.hpp:
/home/sam/projects/parker/server/node_modules/opencv-build/opencv/build/include/opencv2/core/cvstd.hpp:
/home/sam/projects/parker/server/node_modules/opencv-build/opencv/build/include/opencv2/core/ptr.inl.hpp:
/home/sam/projects/parker/server/node_modules/opencv-build/opencv/build/include/opencv2/core/neon_utils.hpp:
/home/sam/projects/parker/server/node_modules/opencv-build/opencv/build/include/opencv2/core/vsx_utils.hpp:
/home/sam/projects/parker/server/node_modules/opencv-build/opencv/build/include/opencv2/core/check.hpp:
/home/sam/projects/parker/server/node_modules/opencv-build/opencv/build/include/opencv2/core/traits.hpp:
/home/sam/projects/parker/server/node_modules/opencv-build/opencv/build/include/opencv2/core/matx.hpp:
/home/sam/projects/parker/server/node_modules/opencv-build/opencv/build/include/opencv2/core/saturate.hpp:
/home/sam/projects/parker/server/node_modules/opencv-build/opencv/build/include/opencv2/core/fast_math.hpp:
/home/sam/projects/parker/server/node_modules/opencv-build/opencv/build/include/opencv2/core/types.hpp:
/home/sam/projects/parker/server/node_modules/opencv-build/opencv/build/include/opencv2/core/mat.hpp:
/home/sam/projects/parker/server/node_modules/opencv-build/opencv/build/include/opencv2/core/bufferpool.hpp:
/home/sam/projects/parker/server/node_modules/opencv-build/opencv/build/include/opencv2/core/mat.inl.hpp:
/home/sam/projects/parker/server/node_modules/opencv-build/opencv/build/include/opencv2/core/persistence.hpp:
/home/sam/projects/parker/server/node_modules/opencv-build/opencv/build/include/opencv2/core/operations.hpp:
/home/sam/projects/parker/server/node_modules/opencv-build/opencv/build/include/opencv2/core/cvstd.inl.hpp:
/home/sam/projects/parker/server/node_modules/opencv-build/opencv/build/include/opencv2/core/utility.hpp:
/home/sam/projects/parker/server/node_modules/opencv-build/opencv/build/include/opencv2/core/core_c.h:
/home/sam/projects/parker/server/node_modules/opencv-build/opencv/build/include/opencv2/core/types_c.h:
/home/sam/projects/parker/server/node_modules/opencv-build/opencv/build/include/opencv2/core/optim.hpp:
/home/sam/projects/parker/server/node_modules/opencv-build/opencv/build/include/opencv2/core/ovx.hpp:
/home/sam/projects/parker/server/node_modules/opencv-build/opencv/build/include/opencv2/core/cvdef.h:
../cc/CustomMatAllocator.h:
../cc/core/Size.h:
../cc/macros.h:
../cc/core/coreUtils.h:
../cc/core/matUtils.h:
../cc/core/Vec.h:
../cc/core/Vec2.h:
../cc/core/coreUtils.h:
../cc/core/Vec3.h:
../cc/core/Vec4.h:
../cc/core/Vec6.h:
../cc/core/Point2.h:
../cc/core/Rect.h:
../cc/CatchCvExceptionWorker.h:
../cc/core/Size.h:
../cc/core/RotatedRect.h:
../cc/core/Point.h:
../cc/core/Point2.h:
../cc/core/Point3.h:
../cc/core/Rect.h:
../cc/opencv_modules.h:
../cc/core/core.h:
../cc/core/Mat.h:
../cc/core/matUtils.h:
../cc/core/Vec.h:
../cc/core/RotatedRect.h:
../cc/ExternalMemTracking.h:
../cc/core/TermCriteria.h:
../cc/core/HistAxes.h:
../cc/calib3d/calib3d.h:
../cc/calib3d/calib3dBindings.h:
/home/sam/projects/parker/server/node_modules/opencv-build/opencv/build/include/opencv2/calib3d.hpp:
/home/sam/projects/parker/server/node_modules/opencv-build/opencv/build/include/opencv2/features2d.hpp:
/home/sam/projects/parker/server/node_modules/opencv-build/opencv/build/include/opencv2/flann/miniflann.hpp:
/home/sam/projects/parker/server/node_modules/opencv-build/opencv/build/include/opencv2/flann/defines.h:
/home/sam/projects/parker/server/node_modules/opencv-build/opencv/build/include/opencv2/flann/config.h:
/home/sam/projects/parker/server/node_modules/opencv-build/opencv/build/include/opencv2/core/affine.hpp:
/home/sam/projects/parker/server/node_modules/opencv-build/opencv/build/include/opencv2/calib3d/calib3d_c.h:
../cc/core/Mat.h:
../cc/core/Point.h:
../cc/core/TermCriteria.h:
../cc/CvBinding.h:
../cc/dnn/dnn.h:
/home/sam/projects/parker/server/node_modules/opencv-build/opencv/build/include/opencv2/dnn.hpp:
/home/sam/projects/parker/server/node_modules/opencv-build/opencv/build/include/opencv2/dnn/dnn.hpp:
/home/sam/projects/parker/server/node_modules/opencv-build/opencv/build/include/opencv2/dnn/dict.hpp:
/home/sam/projects/parker/server/node_modules/opencv-build/opencv/build/include/opencv2/dnn/layer.hpp:
/home/sam/projects/parker/server/node_modules/opencv-build/opencv/build/include/opencv2/dnn/dnn.inl.hpp:
/home/sam/projects/parker/server/node_modules/opencv-build/opencv/build/include/opencv2/dnn/utils/inference_engine.hpp:
/home/sam/projects/parker/server/node_modules/opencv-build/opencv/build/include/opencv2/dnn/utils/../dnn.hpp:
../cc/dnn/Net.h:
../cc/face/face.h:
/home/sam/projects/parker/server/node_modules/opencv-build/opencv/build/include/opencv2/face.hpp:
/home/sam/projects/parker/server/node_modules/opencv-build/opencv/build/include/opencv2/face/predict_collector.hpp:
/home/sam/projects/parker/server/node_modules/opencv-build/opencv/build/include/opencv2/face/facerec.hpp:
/home/sam/projects/parker/server/node_modules/opencv-build/opencv/build/include/opencv2/face/facemark.hpp:
/home/sam/projects/parker/server/node_modules/opencv-build/opencv/build/include/opencv2/face/facemark_train.hpp:
/home/sam/projects/parker/server/node_modules/opencv-build/opencv/build/include/opencv2/objdetect.hpp:
/home/sam/projects/parker/server/node_modules/opencv-build/opencv/build/include/opencv2/objdetect/detection_based_tracker.hpp:
/home/sam/projects/parker/server/node_modules/opencv-build/opencv/build/include/opencv2/objdetect/objdetect_c.h:
/home/sam/projects/parker/server/node_modules/opencv-build/opencv/build/include/opencv2/face/facemarkLBF.hpp:
/home/sam/projects/parker/server/node_modules/opencv-build/opencv/build/include/opencv2/face/facemarkAAM.hpp:
/home/sam/projects/parker/server/node_modules/opencv-build/opencv/build/include/opencv2/face/face_alignment.hpp:
/home/sam/projects/parker/server/node_modules/opencv-build/opencv/build/include/opencv2/face/mace.hpp:
../cc/features2d/features2d.h:
../cc/imgproc/imgproc.h:
/home/sam/projects/parker/server/node_modules/opencv-build/opencv/build/include/opencv2/imgproc.hpp:
/home/sam/projects/parker/server/node_modules/opencv-build/opencv/build/include/opencv2/imgproc/imgproc_c.h:
/home/sam/projects/parker/server/node_modules/opencv-build/opencv/build/include/opencv2/imgproc/types_c.h:
../cc/imgproc/Contour.h:
../cc/imgproc/Moments.h:
../cc/core/HistAxes.h:
../cc/io/io.h:
/home/sam/projects/parker/server/node_modules/opencv-build/opencv/build/include/opencv2/highgui.hpp:
/home/sam/projects/parker/server/node_modules/opencv-build/opencv/build/include/opencv2/imgcodecs.hpp:
/home/sam/projects/parker/server/node_modules/opencv-build/opencv/build/include/opencv2/videoio.hpp:
/home/sam/projects/parker/server/node_modules/opencv-build/opencv/build/include/opencv2/highgui/highgui_c.h:
/home/sam/projects/parker/server/node_modules/opencv-build/opencv/build/include/opencv2/imgcodecs/imgcodecs_c.h:
/home/sam/projects/parker/server/node_modules/opencv-build/opencv/build/include/opencv2/videoio/videoio_c.h:
../cc/io/VideoCapture.h:
../cc/io/VideoWriter.h:
../cc/machinelearning/machinelearning.h:
/home/sam/projects/parker/server/node_modules/opencv-build/opencv/build/include/opencv2/ml.hpp:
/home/sam/projects/parker/server/node_modules/opencv-build/opencv/build/include/opencv2/ml/ml.inl.hpp:
../cc/objdetect/objdetect.h:
../cc/photo/photo.h:
/home/sam/projects/parker/server/node_modules/opencv-build/opencv/build/include/opencv2/photo.hpp:
/home/sam/projects/parker/server/node_modules/opencv-build/opencv/build/include/opencv2/photo/photo_c.h:
../cc/text/text.h:
../cc/text/OCRHMMClassifier.h:
/home/sam/projects/parker/server/node_modules/opencv-build/opencv/build/include/opencv2/text.hpp:
/home/sam/projects/parker/server/node_modules/opencv-build/opencv/build/include/opencv2/text/erfilter.hpp:
/home/sam/projects/parker/server/node_modules/opencv-build/opencv/build/include/opencv2/text/ocr.hpp:
/home/sam/projects/parker/server/node_modules/opencv-build/opencv/build/include/opencv2/text/textDetector.hpp:
/home/sam/projects/parker/server/node_modules/opencv-build/opencv/build/include/opencv2/text/ocr.hpp:
../cc/text/OCRHMMDecoder.h:
../cc/tracking/tracking.h:
/home/sam/projects/parker/server/node_modules/opencv-build/opencv/build/include/opencv2/tracking.hpp:
/home/sam/projects/parker/server/node_modules/opencv-build/opencv/build/include/opencv2/tracking/tracker.hpp:
/home/sam/projects/parker/server/node_modules/opencv-build/opencv/build/include/opencv2/tracking/feature.hpp:
/home/sam/projects/parker/server/node_modules/opencv-build/opencv/build/include/opencv2/tracking/onlineMIL.hpp:
/home/sam/projects/parker/server/node_modules/opencv-build/opencv/build/include/opencv2/tracking/onlineBoosting.hpp:
/home/sam/projects/parker/server/node_modules/opencv-build/opencv/build/include/opencv2/tracking/tldDataset.hpp:
../cc/video/video.h:
/home/sam/projects/parker/server/node_modules/opencv-build/opencv/build/include/opencv2/video.hpp:
/home/sam/projects/parker/server/node_modules/opencv-build/opencv/build/include/opencv2/video/tracking.hpp:
/home/sam/projects/parker/server/node_modules/opencv-build/opencv/build/include/opencv2/video/background_segm.hpp:
/home/sam/projects/parker/server/node_modules/opencv-build/opencv/build/include/opencv2/video/tracking_c.h:
../cc/xfeatures2d/xfeatures2d.h:
../cc/ximgproc/ximgproc.h:
/home/sam/projects/parker/server/node_modules/opencv-build/opencv/build/include/opencv2/ximgproc.hpp:
/home/sam/projects/parker/server/node_modules/opencv-build/opencv/build/include/opencv2/ximgproc/edge_filter.hpp:
/home/sam/projects/parker/server/node_modules/opencv-build/opencv/build/include/opencv2/ximgproc/disparity_filter.hpp:
/home/sam/projects/parker/server/node_modules/opencv-build/opencv/build/include/opencv2/ximgproc/sparse_match_interpolator.hpp:
/home/sam/projects/parker/server/node_modules/opencv-build/opencv/build/include/opencv2/ximgproc/structured_edge_detection.hpp:
/home/sam/projects/parker/server/node_modules/opencv-build/opencv/build/include/opencv2/ximgproc/edgeboxes.hpp:
/home/sam/projects/parker/server/node_modules/opencv-build/opencv/build/include/opencv2/ximgproc/seeds.hpp:
/home/sam/projects/parker/server/node_modules/opencv-build/opencv/build/include/opencv2/ximgproc/segmentation.hpp:
/home/sam/projects/parker/server/node_modules/opencv-build/opencv/build/include/opencv2/ximgproc/fast_hough_transform.hpp:
/home/sam/projects/parker/server/node_modules/opencv-build/opencv/build/include/opencv2/ximgproc/estimated_covariance.hpp:
/home/sam/projects/parker/server/node_modules/opencv-build/opencv/build/include/opencv2/ximgproc/weighted_median_filter.hpp:
/home/sam/projects/parker/server/node_modules/opencv-build/opencv/build/include/opencv2/ximgproc/slic.hpp:
/home/sam/projects/parker/server/node_modules/opencv-build/opencv/build/include/opencv2/ximgproc/lsc.hpp:
/home/sam/projects/parker/server/node_modules/opencv-build/opencv/build/include/opencv2/ximgproc/paillou_filter.hpp:
/home/sam/projects/parker/server/node_modules/opencv-build/opencv/build/include/opencv2/ximgproc/fast_line_detector.hpp:
/home/sam/projects/parker/server/node_modules/opencv-build/opencv/build/include/opencv2/ximgproc/deriche_filter.hpp:
/home/sam/projects/parker/server/node_modules/opencv-build/opencv/build/include/opencv2/ximgproc/peilin.hpp:
/home/sam/projects/parker/server/node_modules/opencv-build/opencv/build/include/opencv2/ximgproc/fourier_descriptors.hpp:
/home/sam/projects/parker/server/node_modules/opencv-build/opencv/build/include/opencv2/ximgproc/ridgefilter.hpp:
/home/sam/projects/parker/server/node_modules/opencv-build/opencv/build/include/opencv2/ximgproc/brightedges.hpp:
