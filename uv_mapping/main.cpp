#include <opencv2/core/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/rgbd.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <fstream>
#include <cmath>

using namespace cv;
using namespace cv::rgbd;
using namespace std;

float limit() {return 1;}
Vec3f hueToRgb(float hue){
	hue -= (long)hue;
	float r, g, b;
	float saturation = 1, brightness = 1;
	float hueSix = hue * 6.f / limit();
	float saturationNorm = saturation / limit();
	int hueSixCategory = (int) floorf(hueSix);
	float hueSixRemainder = hueSix - hueSixCategory;
	float pv = ((1.f - saturationNorm) * brightness);
	float qv = ((1.f - saturationNorm * hueSixRemainder) * brightness);
	float tv = ((1.f - saturationNorm * (1.f - hueSixRemainder)) * brightness);
	switch(hueSixCategory) {
	case 0: case 6: // r
		r = brightness;
		g = tv;
		b = pv;
		break;
	case 1: // g
		r = qv;
		g = brightness;
		b = pv;
		break;
	case 2:
		r = pv;
		g = brightness;
		b = tv;
		break;
	case 3: // b
		r = pv;
		g = qv;
		b = brightness;
		break;
	case 4:
		r = tv;
		g = pv;
		b = brightness;
		break;
	case 5: // back to r
		r = brightness;
		g = pv;
		b = qv;
		break;
	}
	return Vec3f(b, g, r);
}

struct RgbdPoint {
public:
	Point3f world_xyz;
	Point3f bgr;
	Point2i image_xy;
	Point2f texture_uv;
};

class RgbdCluster {
public:
	Mat mask;
	Mat depth; // original depth image
	Mat points3d; // original 3d points
	Mat normals; // original normal map
	Mat pointsIndex; // image to vector point map
	vector<RgbdPoint> points;
	vector<int> faceIndices; // face indices
	bool bPlane;
	bool bPointsUpdated;
	bool bFaceIndicesUpdated;

	RgbdCluster() : bPlane(false), bPointsUpdated(false), bFaceIndicesUpdated(false) {}

	int getNumPoints() {
		if(bPointsUpdated)
			return points.size();
		else return -1;
	}

	void calculatePoints() {
		pointsIndex = Mat_<int>::eye(mask.rows, mask.cols) * -1;
		points.clear();
		for (int i = 0; i < mask.rows; i++)  {
			for (int j = 0; j < mask.cols; j++)  {
				if(mask.at<uchar>(i, j) > 0) {
					if(depth.at<float>(i, j) > 0) {
						RgbdPoint point;
						point.world_xyz = points3d.at<Point3f>(i, j);
						point.image_xy = Point2i(j, i);
						
						pointsIndex.at<int>(i, j) = points.size();
						points.push_back(point);
					} else {
						mask.at<uchar>(i, j) = 0;
					}
				}
			}
		}
		bPointsUpdated = true;
	}

	void calculateFaceIndices() {
		if(!bPointsUpdated) {
			calculatePoints();
		}
		for (int i = 0; i < mask.rows; i++)  {
			for (int j = 0; j < mask.cols; j++)  {
				if(mask.at<uchar>(i, j) == 0) {
					continue;
				}
				if(i + 1 < mask.rows &&
					j + 1 < mask.cols &&
					mask.at<uchar>(i + 1, j) > 0 &&
					mask.at<uchar>(i, j + 1) > 0 &&
					mask.at<uchar>(i + 1, j + 1) > 0)
				{
					faceIndices.push_back(pointsIndex.at<int>(i, j));
					faceIndices.push_back(pointsIndex.at<int>(i+1, j));
					faceIndices.push_back(pointsIndex.at<int>(i, j+1));
					faceIndices.push_back(pointsIndex.at<int>(i, j+1));
					faceIndices.push_back(pointsIndex.at<int>(i+1, j));
					faceIndices.push_back(pointsIndex.at<int>(i+1, j+1));
				}
			}
		}

	}

	void save(string &path) {
		if(!bFaceIndicesUpdated) {
			calculateFaceIndices();
		}
		ofstream fs;
		fs.open(path);
		for(int i = 0; i < points.size(); i++) {
			auto & v = points.at(i).world_xyz;
			fs << "v " << to_string(v.x) << " " << to_string(v.y) << " " << to_string(v.z) << endl;
		}
		for(int i = 0; i < faceIndices.size(); i+=3) {
			fs << "f " << to_string(faceIndices.at(i))
				<< "// " << to_string(faceIndices.at(i+1))
				<< "// " << to_string(faceIndices.at(i+2))
				<< "//" << endl;
		}
		fs.close();
	}

};

void eliminateSmallClusters(vector<RgbdCluster>& clusters, int minPoints) {
	for(int i = 0; i < clusters.size(); ) {
		if(clusters.at(i).getNumPoints() >= 0 && clusters.at(i).getNumPoints() <= minPoints) {
			clusters.erase(clusters.begin() + i);
		} else {
			i++;
		}
	}
}

void deleteEmptyClusters(vector<RgbdCluster>& clusters) {
	eliminateSmallClusters(clusters, 0);
}

void planarSegmentation(RgbdCluster& mainCluster, vector<RgbdCluster>& clusters, int maxPlaneNum = 3, int minArea = 400) {
	// assert frame size == points3d size

	auto plane = makePtr<RgbdPlane>();
	plane->setThreshold(0.025f);
	Mat mask;
	vector<Vec4f> coeffs;
	//(*plane)(points3d, frame->normals, mask, coeffs);
	(*plane)(mainCluster.points3d, mask, coeffs);

	int curLabel = 0;
	Mat colorLabels = Mat_<Vec3f>(mask.rows, mask.cols);
	for(int label = 0; label < maxPlaneNum + 1; label++) {
		clusters.push_back(RgbdCluster());
		RgbdCluster& cluster = clusters.back();
		mainCluster.depth.copyTo(cluster.depth);
		mainCluster.points3d.copyTo(cluster.points3d);
		if(label < maxPlaneNum) {
			compare(mask, label, cluster.mask, CMP_EQ);
			cluster.bPlane = true;
		} else {
			compare(mask, label, cluster.mask, CMP_GE); // residual
		}
		cluster.calculatePoints();
	}
}

void euclideanClustering(RgbdCluster& mainCluster, vector<RgbdCluster>& clusters, int minArea = 400) {
	Mat labels, stats, centroids;
	connectedComponentsWithStats(mainCluster.mask, labels, stats, centroids, 8);
	for(int label = 1; label < stats.rows; label++) { // 0: background label
		if(stats.at<int>(label, CC_STAT_AREA) >= minArea) {
			clusters.push_back(RgbdCluster());
			RgbdCluster& cluster = clusters.back();
			mainCluster.depth.copyTo(cluster.depth);
			mainCluster.points3d.copyTo(cluster.points3d);
			compare(labels, label, cluster.mask, CMP_EQ);
			cluster.calculatePoints();
		}
	}
}

int main( int argc, char** argv )
{
	Mat image, depth;
	float pixelSize, refDistance;
	cv::FileStorage file("rgbd.txt", cv::FileStorage::READ);
	
	file["depth"] >> depth;
	file["zeroPlanePixelSize"] >> pixelSize;
	file["zeroPlaneDistance"] >> refDistance;
	depth = depth * 0.001f; // libfreenect is in [mm]

	float fx = refDistance * 0.5f / pixelSize,
		fy = refDistance * 0.5f / pixelSize,
		cx = 319.5f,
		cy = 239.5f;

	Mat cameraMatrix = Mat::eye(3,3,CV_32FC1);
	{
		cameraMatrix.at<float>(0,0) = fx;
		cameraMatrix.at<float>(1,1) = fy;
		cameraMatrix.at<float>(0,2) = cx;
		cameraMatrix.at<float>(1,2) = cy;
	}

	auto frame = makePtr<RgbdFrame>(image, depth);
	auto cleaner = makePtr<DepthCleaner>(CV_32F, 5);
	Mat tmp;
	(*cleaner)(frame->depth, tmp);
	frame->depth = tmp;

	//auto normals = makePtr<RgbdNormals>(frame->depth.rows, frame->depth.cols, frame->depth.depth(), cameraMatrix, 5, RgbdNormals::RGBD_NORMALS_METHOD_LINEMOD);
	Mat points3d;
	depthTo3d(frame->depth, cameraMatrix, points3d);
	//(*normals)(frame->depth, frame->normals);

	RgbdCluster mainCluster;
	mainCluster.points3d = points3d;
	mainCluster.depth = frame->depth;
	vector<RgbdCluster> clusters;
	planarSegmentation(mainCluster, clusters);
	deleteEmptyClusters(clusters);

	for(int i = 0; i < clusters.size(); i++) {
		imshow(to_string(i), clusters.at(i).mask * 255);

		Mat labels;
		Mat stats;
		Mat centroids;

		if(clusters.at(i).bPlane) {
			continue;
		}
		
		vector<RgbdCluster> smallClusters;
		euclideanClustering(clusters.at(i), smallClusters);
		//deleteEmptyClusters(smallClusters);
		for(int j = 0; j < smallClusters.size(); j++) {
			imshow(to_string(i) + to_string(j), smallClusters.at(j).mask * 255);
			smallClusters.at(j).save(to_string(i) + to_string(j) + "mesh.obj");
		}
	}

	waitKey(0);
	return 0;
}
