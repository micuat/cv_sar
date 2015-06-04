#include "ofApp.h"

//--------------------------------------------------------------
void ofApp::setup(){
	ofSetLogLevel(OF_LOG_VERBOSE);
	
	glEnable(GL_DEPTH_TEST);
	
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
	vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> clouds;
	
	mesh.load(ofToDataPath("out.ply"));
	cloud = ofxPCL::toPCL<ofxPCL::PointXYZCloud>(mesh);
	
	textures.resize(2);
	textures.at(0).loadImage(ofToDataPath("tex0.jpg"));
	textures.at(1).loadImage(ofToDataPath("tex1.jpg"));
	
	pcl::SacModel model_type = pcl::SACMODEL_PLANE;
	float distance_threshold = 30;
	int min_points_limit = 10;
	int max_segment_count = 30;
	
	pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients());
	pcl::PointIndices::Ptr inliers(new pcl::PointIndices());
	
	pcl::SACSegmentation<pcl::PointXYZ> seg;
	seg.setOptimizeCoefficients(false);
	
	seg.setModelType(model_type);
	seg.setMethodType(pcl::SAC_RANSAC);
	seg.setDistanceThreshold(distance_threshold);
	seg.setMaxIterations(500);
	
	pcl::PointCloud<pcl::PointXYZ>::Ptr temp(new pcl::PointCloud<pcl::PointXYZ>(*cloud));
	const size_t original_szie = temp->points.size();
	
	pcl::ExtractIndices<pcl::PointXYZ> extract;
	
	int segment_count = 0;
	while (temp->size() > original_szie * 0.3)
	{
		if (segment_count > max_segment_count) break;
		segment_count++;
		
		seg.setInputCloud(temp);
		seg.segment(*inliers, *coefficients);
		
		if (inliers->indices.size() < min_points_limit)
			break;
		
		pcl::PointCloud<pcl::PointXYZ>::Ptr filterd_point_cloud(new pcl::PointCloud<pcl::PointXYZ>);
		
		extract.setInputCloud(temp);
		extract.setIndices(inliers);
		extract.setNegative(false);
		extract.filter(*filterd_point_cloud);
		
		if (filterd_point_cloud->points.size() > 0)
		{
			clouds.push_back(filterd_point_cloud);
		}
		
		extract.setNegative(true);
		extract.filter(*temp);
		
		ofMesh m;
		pcl::PointCloud<pcl::PointNormal>::Ptr cloud_with_normals(new pcl::PointCloud<pcl::PointNormal>);
		
		ofxPCL::normalEstimation(filterd_point_cloud, cloud_with_normals);
		
		m = ofxPCL::triangulate(cloud_with_normals, 100);
		m.clearColors();
		jointMesh.addVertices(m.getVertices());
		jointMesh.addColors(m.getColors());
		
		ofVec3f center = m.getCentroid();
		ofVec3f planeNormal(coefficients->values[0], coefficients->values[1], coefficients->values[2]);
		
		ofVec3f tangent, bitangent;
		ofVec3f arb(0, 1, 0);
		tangent = arb.cross(planeNormal).normalize();
		bitangent = planeNormal.cross(tangent).normalize();
		
		for(int j = 0; j < m.getNumVertices(); j++) {
			float x = m.getVertex(j).dot(tangent) * 0.001;
			x = x - (long)x;
			if(x < 0) x += 1;
			float y = m.getVertex(j).dot(bitangent) * 0.001;
			y = y - (long)y;
			if(y < 0) y += 1;
			m.addTexCoord(ofVec2f(x, y));
		}
		meshes.push_back(m);
	}
	
	ofxPCL::convert(temp, meshResidual);
	
	ofLogVerbose() << clouds.size() << " meshes extracted";
	
	center = jointMesh.getCentroid();
}

//--------------------------------------------------------------
void ofApp::update(){

}

//--------------------------------------------------------------
void ofApp::draw(){
	ofBackground(0);
	
	cam.begin();
	ofScale(1, -1, -1);
	ofTranslate(-center);
	
	ofEnableNormalizedTexCoords();
	for( int i = 0; i < meshes.size(); i++ ) {
		textures.at(i).bind();
		if(ofGetKeyPressed('w'))
			meshes.at(i).drawWireframe();
		else
			meshes.at(i).draw();
		textures.at(i).unbind();
	}
	ofSetColor(255);
	meshResidual.drawVertices();
	
	cam.end();
}

//--------------------------------------------------------------
void ofApp::keyPressed(int key){
	if(key == ' ') {
		for(int i = 0; i < meshes.size(); i++) {
			meshes.at(i).save(ofToDataPath("planar" + ofToString(i, 3, '0') + ".ply"));
		}
	}
}

//--------------------------------------------------------------
void ofApp::keyReleased(int key){

}

//--------------------------------------------------------------
void ofApp::mouseMoved(int x, int y ){

}

//--------------------------------------------------------------
void ofApp::mouseDragged(int x, int y, int button){

}

//--------------------------------------------------------------
void ofApp::mousePressed(int x, int y, int button){

}

//--------------------------------------------------------------
void ofApp::mouseReleased(int x, int y, int button){

}

//--------------------------------------------------------------
void ofApp::windowResized(int w, int h){

}

//--------------------------------------------------------------
void ofApp::gotMessage(ofMessage msg){

}

//--------------------------------------------------------------
void ofApp::dragEvent(ofDragInfo dragInfo){ 

}
