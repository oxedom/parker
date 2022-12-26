#include "FaceRecognizer.h"

#ifndef __FF_EIGENFACERECOGNIZER_H__
#define __FF_EIGENFACERECOGNIZER_H__

class EigenFaceRecognizer : public FaceRecognizer {
public:
	cv::Ptr<cv::face::FaceRecognizer> faceRecognizer;
	void save(std::string path) {
		faceRecognizer->save(path);
	}

	void load(std::string path) {
#if CV_VERSION_GREATER_EQUAL(3, 3, 0)
		faceRecognizer = cv::Algorithm::load<cv::face::EigenFaceRecognizer>(path);
#else
		faceRecognizer->load(path);
#endif
	}

	static NAN_MODULE_INIT(Init);
	static NAN_METHOD(New);

	static Nan::Persistent<v8::FunctionTemplate> constructor;

	cv::Ptr<cv::face::FaceRecognizer> getFaceRecognizer() {
		return faceRecognizer;
	}
};

#endif