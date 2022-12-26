#include "FaceRecognizer.h"

#ifndef __FF_LBPHFACERECOGNIZER_H__
#define __FF_LBPHFACERECOGNIZER_H__

class LBPHFaceRecognizer : public FaceRecognizer {
public:
	cv::Ptr<cv::face::FaceRecognizer> faceRecognizer;
	void save(std::string path) {
		faceRecognizer->save(path);
	}

	void load(std::string path) {
#if CV_VERSION_GREATER_EQUAL(3, 3, 0)
		faceRecognizer = cv::Algorithm::load<cv::face::LBPHFaceRecognizer>(path);
#else
		faceRecognizer->load(path);
#endif
	}

	static NAN_MODULE_INIT(Init);
	struct NewWorker;
	static NAN_METHOD(New);

	static Nan::Persistent<v8::FunctionTemplate> constructor;

	cv::Ptr<cv::face::FaceRecognizer> getFaceRecognizer() {
		return faceRecognizer;
	}
};

#endif