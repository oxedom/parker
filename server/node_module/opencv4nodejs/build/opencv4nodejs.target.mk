# This file is generated by gyp; do not edit.

TOOLSET := target
TARGET := opencv4nodejs
DEFS_Debug := \
	'-DNODE_GYP_MODULE_NAME=opencv4nodejs' \
	'-DUSING_UV_SHARED=1' \
	'-DUSING_V8_SHARED=1' \
	'-DV8_DEPRECATION_WARNINGS=1' \
	'-DV8_DEPRECATION_WARNINGS' \
	'-DV8_IMMINENT_DEPRECATION_WARNINGS' \
	'-D_GLIBCXX_USE_CXX11_ABI=1' \
	'-D_LARGEFILE_SOURCE' \
	'-D_FILE_OFFSET_BITS=64' \
	'-D__STDC_FORMAT_MACROS' \
	'-DOPENSSL_NO_PINSHARED' \
	'-DOPENSSL_THREADS' \
	'-DOPENCV4NODEJS_FOUND_LIBRARY_CORE' \
	'-DOPENCV4NODEJS_FOUND_LIBRARY_HIGHGUI' \
	'-DOPENCV4NODEJS_FOUND_LIBRARY_IMGCODECS' \
	'-DOPENCV4NODEJS_FOUND_LIBRARY_IMGPROC' \
	'-DOPENCV4NODEJS_FOUND_LIBRARY_FEATURES2D' \
	'-DOPENCV4NODEJS_FOUND_LIBRARY_CALIB3D' \
	'-DOPENCV4NODEJS_FOUND_LIBRARY_PHOTO' \
	'-DOPENCV4NODEJS_FOUND_LIBRARY_OBJDETECT' \
	'-DOPENCV4NODEJS_FOUND_LIBRARY_ML' \
	'-DOPENCV4NODEJS_FOUND_LIBRARY_VIDEO' \
	'-DOPENCV4NODEJS_FOUND_LIBRARY_VIDEOIO' \
	'-DOPENCV4NODEJS_FOUND_LIBRARY_VIDEOSTAB' \
	'-DOPENCV4NODEJS_FOUND_LIBRARY_DNN' \
	'-DOPENCV4NODEJS_FOUND_LIBRARY_FACE' \
	'-DOPENCV4NODEJS_FOUND_LIBRARY_TEXT' \
	'-DOPENCV4NODEJS_FOUND_LIBRARY_TRACKING' \
	'-DOPENCV4NODEJS_FOUND_LIBRARY_XFEATURES2D' \
	'-DOPENCV4NODEJS_FOUND_LIBRARY_XIMGPROC' \
	'-DBUILDING_NODE_EXTENSION' \
	'-DDEBUG' \
	'-D_DEBUG' \
	'-DV8_ENABLE_CHECKS'

# Flags passed to all source files.
CFLAGS_Debug := \
	-fPIC \
	-pthread \
	-Wall \
	-Wextra \
	-Wno-unused-parameter \
	-m64 \
	-std=c++11 \
	-g \
	-O0 \
	--coverage

# Flags passed to only C files.
CFLAGS_C_Debug :=

# Flags passed to only C++ files.
CFLAGS_CC_Debug := \
	-std=gnu++17

INCS_Debug := \
	-I/home/sam/.cache/node-gyp/18.12.1/include/node \
	-I/home/sam/.cache/node-gyp/18.12.1/src \
	-I/home/sam/.cache/node-gyp/18.12.1/deps/openssl/config \
	-I/home/sam/.cache/node-gyp/18.12.1/deps/openssl/openssl/include \
	-I/home/sam/.cache/node-gyp/18.12.1/deps/uv/include \
	-I/home/sam/.cache/node-gyp/18.12.1/deps/zlib \
	-I/home/sam/.cache/node-gyp/18.12.1/deps/v8/include \
	-I/home/sam/projects/parker/server/node_modules/opencv-build/opencv/build/include \
	-I/home/sam/projects/parker/server/node_modules/opencv-build/opencv/build/include/opencv4 \
	-I$(srcdir)/cc \
	-I$(srcdir)/cc/core \
	-I$(srcdir)/../nan \
	-I/home/sam/projects/parker/server/node_modules/native-node-utils/src

DEFS_Release := \
	'-DNODE_GYP_MODULE_NAME=opencv4nodejs' \
	'-DUSING_UV_SHARED=1' \
	'-DUSING_V8_SHARED=1' \
	'-DV8_DEPRECATION_WARNINGS=1' \
	'-DV8_DEPRECATION_WARNINGS' \
	'-DV8_IMMINENT_DEPRECATION_WARNINGS' \
	'-D_GLIBCXX_USE_CXX11_ABI=1' \
	'-D_LARGEFILE_SOURCE' \
	'-D_FILE_OFFSET_BITS=64' \
	'-D__STDC_FORMAT_MACROS' \
	'-DOPENSSL_NO_PINSHARED' \
	'-DOPENSSL_THREADS' \
	'-DOPENCV4NODEJS_FOUND_LIBRARY_CORE' \
	'-DOPENCV4NODEJS_FOUND_LIBRARY_HIGHGUI' \
	'-DOPENCV4NODEJS_FOUND_LIBRARY_IMGCODECS' \
	'-DOPENCV4NODEJS_FOUND_LIBRARY_IMGPROC' \
	'-DOPENCV4NODEJS_FOUND_LIBRARY_FEATURES2D' \
	'-DOPENCV4NODEJS_FOUND_LIBRARY_CALIB3D' \
	'-DOPENCV4NODEJS_FOUND_LIBRARY_PHOTO' \
	'-DOPENCV4NODEJS_FOUND_LIBRARY_OBJDETECT' \
	'-DOPENCV4NODEJS_FOUND_LIBRARY_ML' \
	'-DOPENCV4NODEJS_FOUND_LIBRARY_VIDEO' \
	'-DOPENCV4NODEJS_FOUND_LIBRARY_VIDEOIO' \
	'-DOPENCV4NODEJS_FOUND_LIBRARY_VIDEOSTAB' \
	'-DOPENCV4NODEJS_FOUND_LIBRARY_DNN' \
	'-DOPENCV4NODEJS_FOUND_LIBRARY_FACE' \
	'-DOPENCV4NODEJS_FOUND_LIBRARY_TEXT' \
	'-DOPENCV4NODEJS_FOUND_LIBRARY_TRACKING' \
	'-DOPENCV4NODEJS_FOUND_LIBRARY_XFEATURES2D' \
	'-DOPENCV4NODEJS_FOUND_LIBRARY_XIMGPROC' \
	'-DBUILDING_NODE_EXTENSION'

# Flags passed to all source files.
CFLAGS_Release := \
	-fPIC \
	-pthread \
	-Wall \
	-Wextra \
	-Wno-unused-parameter \
	-m64 \
	-std=c++11 \
	-O3 \
	-fno-omit-frame-pointer

# Flags passed to only C files.
CFLAGS_C_Release :=

# Flags passed to only C++ files.
CFLAGS_CC_Release := \
	-std=gnu++17

INCS_Release := \
	-I/home/sam/.cache/node-gyp/18.12.1/include/node \
	-I/home/sam/.cache/node-gyp/18.12.1/src \
	-I/home/sam/.cache/node-gyp/18.12.1/deps/openssl/config \
	-I/home/sam/.cache/node-gyp/18.12.1/deps/openssl/openssl/include \
	-I/home/sam/.cache/node-gyp/18.12.1/deps/uv/include \
	-I/home/sam/.cache/node-gyp/18.12.1/deps/zlib \
	-I/home/sam/.cache/node-gyp/18.12.1/deps/v8/include \
	-I/home/sam/projects/parker/server/node_modules/opencv-build/opencv/build/include \
	-I/home/sam/projects/parker/server/node_modules/opencv-build/opencv/build/include/opencv4 \
	-I$(srcdir)/cc \
	-I$(srcdir)/cc/core \
	-I$(srcdir)/../nan \
	-I/home/sam/projects/parker/server/node_modules/native-node-utils/src

OBJS := \
	$(obj).target/$(TARGET)/cc/opencv4nodejs.o \
	$(obj).target/$(TARGET)/cc/CustomMatAllocator.o \
	$(obj).target/$(TARGET)/cc/ExternalMemTracking.o \
	$(obj).target/$(TARGET)/cc/core/core.o \
	$(obj).target/$(TARGET)/cc/core/coreConstants.o \
	$(obj).target/$(TARGET)/cc/core/HistAxes.o \
	$(obj).target/$(TARGET)/cc/core/Mat.o \
	$(obj).target/$(TARGET)/cc/core/Point.o \
	$(obj).target/$(TARGET)/cc/core/Vec.o \
	$(obj).target/$(TARGET)/cc/core/Size.o \
	$(obj).target/$(TARGET)/cc/core/Rect.o \
	$(obj).target/$(TARGET)/cc/core/RotatedRect.o \
	$(obj).target/$(TARGET)/cc/core/TermCriteria.o \
	$(obj).target/$(TARGET)/cc/imgproc/imgproc.o \
	$(obj).target/$(TARGET)/cc/imgproc/imgprocConstants.o \
	$(obj).target/$(TARGET)/cc/imgproc/MatImgproc.o \
	$(obj).target/$(TARGET)/cc/imgproc/Contour.o \
	$(obj).target/$(TARGET)/cc/imgproc/Moments.o \
	$(obj).target/$(TARGET)/cc/calib3d/calib3d.o \
	$(obj).target/$(TARGET)/cc/calib3d/calib3dConstants.o \
	$(obj).target/$(TARGET)/cc/calib3d/MatCalib3d.o \
	$(obj).target/$(TARGET)/cc/io/io.o \
	$(obj).target/$(TARGET)/cc/io/ioConstants.o \
	$(obj).target/$(TARGET)/cc/io/VideoCapture.o \
	$(obj).target/$(TARGET)/cc/io/VideoWriter.o \
	$(obj).target/$(TARGET)/cc/photo/photo.o \
	$(obj).target/$(TARGET)/cc/photo/photoConstants.o \
	$(obj).target/$(TARGET)/cc/photo/MatPhoto.o \
	$(obj).target/$(TARGET)/cc/video/video.o \
	$(obj).target/$(TARGET)/cc/video/BackgroundSubtractor.o \
	$(obj).target/$(TARGET)/cc/video/BackgroundSubtractorMOG2.o \
	$(obj).target/$(TARGET)/cc/video/BackgroundSubtractorKNN.o \
	$(obj).target/$(TARGET)/cc/ximgproc/ximgproc.o \
	$(obj).target/$(TARGET)/cc/ximgproc/MatXimgproc.o \
	$(obj).target/$(TARGET)/cc/ximgproc/SuperpixelSEEDS.o \
	$(obj).target/$(TARGET)/cc/ximgproc/SuperpixelSLIC.o \
	$(obj).target/$(TARGET)/cc/ximgproc/SuperpixelLSC.o \
	$(obj).target/$(TARGET)/cc/objdetect/objdetect.o \
	$(obj).target/$(TARGET)/cc/objdetect/CascadeClassifier.o \
	$(obj).target/$(TARGET)/cc/objdetect/HOGDescriptor.o \
	$(obj).target/$(TARGET)/cc/objdetect/DetectionROI.o \
	$(obj).target/$(TARGET)/cc/machinelearning/machinelearning.o \
	$(obj).target/$(TARGET)/cc/machinelearning/machinelearningConstants.o \
	$(obj).target/$(TARGET)/cc/machinelearning/ParamGrid.o \
	$(obj).target/$(TARGET)/cc/machinelearning/StatModel.o \
	$(obj).target/$(TARGET)/cc/machinelearning/SVM.o \
	$(obj).target/$(TARGET)/cc/machinelearning/TrainData.o \
	$(obj).target/$(TARGET)/cc/dnn/dnn.o \
	$(obj).target/$(TARGET)/cc/dnn/Net.o \
	$(obj).target/$(TARGET)/cc/face/face.o \
	$(obj).target/$(TARGET)/cc/face/FaceRecognizer.o \
	$(obj).target/$(TARGET)/cc/face/EigenFaceRecognizer.o \
	$(obj).target/$(TARGET)/cc/face/FisherFaceRecognizer.o \
	$(obj).target/$(TARGET)/cc/face/LBPHFaceRecognizer.o \
	$(obj).target/$(TARGET)/cc/face/Facemark.o \
	$(obj).target/$(TARGET)/cc/face/FacemarkAAM.o \
	$(obj).target/$(TARGET)/cc/face/FacemarkAAMData.o \
	$(obj).target/$(TARGET)/cc/face/FacemarkAAMParams.o \
	$(obj).target/$(TARGET)/cc/face/FacemarkLBF.o \
	$(obj).target/$(TARGET)/cc/face/FacemarkLBFParams.o \
	$(obj).target/$(TARGET)/cc/text/text.o \
	$(obj).target/$(TARGET)/cc/text/OCRHMMClassifier.o \
	$(obj).target/$(TARGET)/cc/text/OCRHMMDecoder.o \
	$(obj).target/$(TARGET)/cc/tracking/tracking.o \
	$(obj).target/$(TARGET)/cc/tracking/Tracker.o \
	$(obj).target/$(TARGET)/cc/tracking/MultiTracker.o \
	$(obj).target/$(TARGET)/cc/tracking/Trackers/TrackerBoosting.o \
	$(obj).target/$(TARGET)/cc/tracking/Trackers/TrackerBoostingParams.o \
	$(obj).target/$(TARGET)/cc/tracking/Trackers/TrackerKCF.o \
	$(obj).target/$(TARGET)/cc/tracking/Trackers/TrackerKCFParams.o \
	$(obj).target/$(TARGET)/cc/tracking/Trackers/TrackerMIL.o \
	$(obj).target/$(TARGET)/cc/tracking/Trackers/TrackerMILParams.o \
	$(obj).target/$(TARGET)/cc/tracking/Trackers/TrackerMedianFlow.o \
	$(obj).target/$(TARGET)/cc/tracking/Trackers/TrackerTLD.o \
	$(obj).target/$(TARGET)/cc/tracking/Trackers/TrackerGOTURN.o \
	$(obj).target/$(TARGET)/cc/tracking/Trackers/TrackerCSRT.o \
	$(obj).target/$(TARGET)/cc/tracking/Trackers/TrackerCSRTParams.o \
	$(obj).target/$(TARGET)/cc/tracking/Trackers/TrackerMOSSE.o \
	$(obj).target/$(TARGET)/cc/features2d/features2d.o \
	$(obj).target/$(TARGET)/cc/features2d/KeyPoint.o \
	$(obj).target/$(TARGET)/cc/features2d/KeyPointMatch.o \
	$(obj).target/$(TARGET)/cc/features2d/DescriptorMatch.o \
	$(obj).target/$(TARGET)/cc/features2d/BFMatcher.o \
	$(obj).target/$(TARGET)/cc/features2d/FeatureDetector.o \
	$(obj).target/$(TARGET)/cc/features2d/descriptorMatching.o \
	$(obj).target/$(TARGET)/cc/features2d/descriptorMatchingKnn.o \
	$(obj).target/$(TARGET)/cc/features2d/detectors/AGASTDetector.o \
	$(obj).target/$(TARGET)/cc/features2d/detectors/AKAZEDetector.o \
	$(obj).target/$(TARGET)/cc/features2d/detectors/BRISKDetector.o \
	$(obj).target/$(TARGET)/cc/features2d/detectors/FASTDetector.o \
	$(obj).target/$(TARGET)/cc/features2d/detectors/GFTTDetector.o \
	$(obj).target/$(TARGET)/cc/features2d/detectors/KAZEDetector.o \
	$(obj).target/$(TARGET)/cc/features2d/detectors/MSERDetector.o \
	$(obj).target/$(TARGET)/cc/features2d/detectors/ORBDetector.o \
	$(obj).target/$(TARGET)/cc/features2d/detectors/SimpleBlobDetector.o \
	$(obj).target/$(TARGET)/cc/features2d/detectors/SimpleBlobDetectorParams.o \
	$(obj).target/$(TARGET)/cc/xfeatures2d/xfeatures2d.o \
	$(obj).target/$(TARGET)/cc/xfeatures2d/SIFTDetector.o \
	$(obj).target/$(TARGET)/cc/xfeatures2d/SURFDetector.o

# Add to the list of files we specially track dependencies for.
all_deps += $(OBJS)

# CFLAGS et al overrides must be target-local.
# See "Target-specific Variable Values" in the GNU Make manual.
$(OBJS): TOOLSET := $(TOOLSET)
$(OBJS): GYP_CFLAGS := $(DEFS_$(BUILDTYPE)) $(INCS_$(BUILDTYPE))  $(CFLAGS_$(BUILDTYPE)) $(CFLAGS_C_$(BUILDTYPE))
$(OBJS): GYP_CXXFLAGS := $(DEFS_$(BUILDTYPE)) $(INCS_$(BUILDTYPE))  $(CFLAGS_$(BUILDTYPE)) $(CFLAGS_CC_$(BUILDTYPE))

# Suffix rules, putting all outputs into $(obj).

$(obj).$(TOOLSET)/$(TARGET)/%.o: $(srcdir)/%.cc FORCE_DO_CMD
	@$(call do_cmd,cxx,1)

# Try building from generated source, too.

$(obj).$(TOOLSET)/$(TARGET)/%.o: $(obj).$(TOOLSET)/%.cc FORCE_DO_CMD
	@$(call do_cmd,cxx,1)

$(obj).$(TOOLSET)/$(TARGET)/%.o: $(obj)/%.cc FORCE_DO_CMD
	@$(call do_cmd,cxx,1)

# End of this set of suffix rules
### Rules for final target.
LDFLAGS_Debug := \
	-pthread \
	-rdynamic \
	-m64 \
	-Wl,-rpath,'$$ORIGIN' \
	--coverage

LDFLAGS_Release := \
	-pthread \
	-rdynamic \
	-m64 \
	-Wl,-rpath,'$$ORIGIN'

LIBS := \
	-L/home/sam/projects/parker/server/node_modules/opencv-build/opencv/build/lib \
	-lopencv_core \
	-lopencv_highgui \
	-lopencv_imgcodecs \
	-lopencv_imgproc \
	-lopencv_features2d \
	-lopencv_calib3d \
	-lopencv_photo \
	-lopencv_objdetect \
	-lopencv_ml \
	-lopencv_video \
	-lopencv_videoio \
	-lopencv_videostab \
	-lopencv_dnn \
	-lopencv_face \
	-lopencv_text \
	-lopencv_tracking \
	-lopencv_xfeatures2d \
	-lopencv_ximgproc \
	-Wl,-rpath,/home/sam/projects/parker/server/node_modules/opencv-build/opencv/build/lib

$(obj).target/opencv4nodejs.node: GYP_LDFLAGS := $(LDFLAGS_$(BUILDTYPE))
$(obj).target/opencv4nodejs.node: LIBS := $(LIBS)
$(obj).target/opencv4nodejs.node: TOOLSET := $(TOOLSET)
$(obj).target/opencv4nodejs.node: $(OBJS) FORCE_DO_CMD
	$(call do_cmd,solink_module)

all_deps += $(obj).target/opencv4nodejs.node
# Add target alias
.PHONY: opencv4nodejs
opencv4nodejs: $(builddir)/opencv4nodejs.node

# Copy this to the executable output path.
$(builddir)/opencv4nodejs.node: TOOLSET := $(TOOLSET)
$(builddir)/opencv4nodejs.node: $(obj).target/opencv4nodejs.node FORCE_DO_CMD
	$(call do_cmd,copy)

all_deps += $(builddir)/opencv4nodejs.node
# Short alias for building this executable.
.PHONY: opencv4nodejs.node
opencv4nodejs.node: $(obj).target/opencv4nodejs.node $(builddir)/opencv4nodejs.node

# Add executable to "all" target.
.PHONY: all
all: $(builddir)/opencv4nodejs.node

