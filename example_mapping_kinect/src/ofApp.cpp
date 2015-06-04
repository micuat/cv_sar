#include "ofApp.h"

using namespace ofxActiveScan;

// entry point
void ofApp::setup() {
	if( rootDir.size() > 0 ) {
		init();
		pathLoaded = true;
	} else {
		pathLoaded = false;
	}
}

void ofApp::init() {
	ofSetLogLevel(OF_LOG_VERBOSE);
	
	// enable depth->video image calibration
	kinect.setRegistration(true);
    
	kinect.init();
	
	kinect.open();		// opens first available kinect
	
	cameraMode = PRO_MODE;
	
	cv::FileStorage fs(ofToDataPath(rootDir[0] + "/config.yml"), cv::FileStorage::READ);
	fs["proWidth"] >> options.projector_width;
	fs["proHeight"] >> options.projector_height;
	fs["vertical_center"] >> options.projector_horizontal_center;
	fs["nsamples"] >> options.nsamples;
	
	cv::FileStorage cfs(ofToDataPath(rootDir[0] + "/calibration.yml"), cv::FileStorage::READ);
	cfs["proIntrinsic"] >> proIntrinsic;
	cfs["proExtrinsic"] >> proExtrinsic;
	
	proSize.width = options.projector_width;
	proSize.height = options.projector_height;
	cout << proIntrinsic << endl;
	cout << proExtrinsic << endl;
	
	// set parameters for projection
	proCalibration.setup(proIntrinsic, proSize);
	
	ofEnableDepthTest();
	
	preMeshes.resize(2);
	preMeshes.at(0).load(ofToDataPath(rootDir[0] + "/planar000.ply"));
	preMeshes.at(1).load(ofToDataPath(rootDir[0] + "/planar001.ply"));
}

void ofApp::update() {
	kinect.update();
}

void ofApp::draw() {
	if( pathLoaded ) {
		
		ofBackground(0);
		
		if(cameraMode == EASYCAM_MODE) {
			cam.begin();
			ofScale(1, -1, -1);
			//ofScale(1000, 1000, 1000);
			ofTranslate(0, 0, -2000);
		} else if(cameraMode == PRO_MODE) {
			ofSetupScreenPerspective(options.projector_width, options.projector_height);
			proCalibration.loadProjectionMatrix(0.0001, 100000000.0);
			cv::Mat m = proExtrinsic;
			cv::Mat extrinsics = (cv::Mat1d(4,4) <<
								  m.at<double>(0,0), m.at<double>(0,1), m.at<double>(0,2), m.at<double>(0,3),
								  m.at<double>(1,0), m.at<double>(1,1), m.at<double>(1,2), m.at<double>(1,3),
								  m.at<double>(2,0), m.at<double>(2,1), m.at<double>(2,2), m.at<double>(2,3),
								  0, 0, 0, 1);
			extrinsics = extrinsics.t();
			glMultMatrixd((GLdouble*) extrinsics.ptr(0, 0));
		}
		
		ofSetColor(255);
		for(int i = 0; i < preMeshes.size(); i++) {
			preMeshes.at(i).draw();
		}
		if(ofGetKeyPressed('1'))
			drawPointCloud();
		
		if(cameraMode == EASYCAM_MODE) {
			cam.end();
		}
		
	}
}


void ofApp::drawPointCloud() {
	
	int w = 640;
	int h = 480;
	mesh.clear();
	mesh.setMode(OF_PRIMITIVE_POINTS);
	glPointSize(2);
	int step = 2;
	
	for(int y = 0; y < h; y += step) {
		for(int x = 0; x < w; x += step) {
			if(kinect.getDistanceAt(x, y) > 0 ) {
				ofColor c;
				if( x % (step*2) ) {
					c.setHsb(kinect.getDepthPixelsRef().getColor(x,y).getBrightness(), 255, 255);
				} else {
					c.setHsb(255 - kinect.getDepthPixelsRef().getColor(x,y).getBrightness(), 255, 255);
				}

				mesh.addColor(c);
				mesh.addVertex(kinect.getWorldCoordinateAt(x, y));
			}
		}
	}
	mesh.draw();
}

void ofApp::keyPressed(int key) {
	if( pathLoaded ) {
		
		switch(key) {
			case '1': cameraMode = EASYCAM_MODE; break;
			case '2': cameraMode = PRO_MODE; break;
		}
		
	}
	
	if( key == 'f' ) {
		ofToggleFullscreen();
	}
	
	if( key == ' ' ) {
		mesh.save(ofToDataPath(rootDir[0] + "/snap.ply"));
	}
}

void ofApp::dragEvent(ofDragInfo dragInfo){
	if( !pathLoaded ) {
		for( int i = 0 ; i < dragInfo.files.size() ; i++ ) {
			rootDir.push_back(dragInfo.files[i]);
		}
		init();
		pathLoaded = true;
	}
}