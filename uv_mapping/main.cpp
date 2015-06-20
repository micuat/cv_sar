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

int main( int argc, char** argv )
{
	vector<float> depthLookUp(2048);
	
	Mat image, depth;
	float pixelSize, refDistance;
	cv::FileStorage file("rgbd.txt", cv::FileStorage::READ);
	
	file["depth"] >> depth;
	file["zeroPlanePixelSize"] >> pixelSize;
	file["zeroPlaneDistance"] >> refDistance;
	depth = depth * 0.001f;

	float fx = refDistance * 0.5f / pixelSize, // default
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

	auto plane = makePtr<RgbdPlane>();
	plane->setThreshold(0.025f);
	Mat mask;
	vector<Vec4f> coeffs;
	//(*plane)(points3d, frame->normals, mask, coeffs);
	(*plane)(points3d, mask, coeffs);
	cout<<coeffs.size()<<endl;

	vector<int> pixelCounts(coeffs.size(), 0);
	Mat_<Vec3f> colorMask(frame->depth.rows, frame->depth.cols);
	for (int i = 0; i < mask.rows; ++i)
	{
		for (int j = 0; j < mask.cols; ++j)
		{
			if (mask.at<uchar>(i, j) != 255)
			{
				pixelCounts[mask.at<uchar>(i, j)] += 1;

				colorMask.at<Vec3f>(i, j) = hueToRgb(mask.at<uchar>(i, j) * 0.1f);//Vec3f(mask.at<uchar>(i, j) * 0.1f, 1.f - mask.at<uchar>(i, j) * 0.1f, 1);
			}
		}
	}
	imshow("planes", colorMask);

	int minArea = 400;
	int curLabel = 0;
	Mat colorLabels = Mat_<Vec3f>(mask.rows, mask.cols);
	Mat aggregatedLabels = Mat_<int>::ones(mask.rows, mask.cols) * -1;
	Mat aggregatedStats;
	Mat aggregatedCentroids;
	int maxPlaneNum = 3;
	for(int l = 0; l < maxPlaneNum + 1; l++) {
		Mat labels, stats, centroids;
		Mat src(mask.rows, mask.cols, CV_8U);
		uchar curIndex = 1;
		for (int i = 0; i < mask.rows; i++)  {
			for (int j = 0; j < mask.cols; j++)  {
				src.at<uchar>(i, j) = 0;
				if(depth.at<float>(i, j) == 0) {
					continue;
				}
				if(l == maxPlaneNum) {
					src.at<uchar>(i, j) = (mask.at<uchar>(i, j) >= l);
				} else {
					src.at<uchar>(i, j) = (mask.at<uchar>(i, j) == l);
				}
			}
		}

		int numLabels = connectedComponentsWithStats(src, labels, stats, centroids, 8);
		if(!aggregatedStats.data) {
			aggregatedStats = stats;
			aggregatedCentroids = centroids;
		} else {
			aggregatedStats.push_back(stats);
			aggregatedCentroids.push_back(centroids);
		}

		for (int i = 0; i < mask.rows; i++)  {
			for (int j = 0; j < mask.cols; j++)  {
				if(src.at<uchar>(i, j) == 0) continue;
				aggregatedLabels.at<int>(i, j) = labels.at<int>(i, j) + curLabel;
			}
		}
		curLabel += numLabels;
	}

	int newLabelNum = 0;
	vector<int> newLabels(aggregatedStats.rows, -1);
	for(int i = 0; i < aggregatedStats.rows; i++) {
		if(aggregatedStats.at<int>(i, CC_STAT_AREA) < minArea) {
			continue;
		}
		//if(depth.at<float>(aggregatedStats.at<int>(i, CC_STAT_TOP), aggregatedStats.at<int>(i, CC_STAT_LEFT)) == 0) {
		//	continue;
		//}
		newLabels.at(i) = newLabelNum;
		newLabelNum++;
	}
	cout << newLabelNum << endl;
	vector<vector<Point3f> > meshes(newLabelNum);
	vector<vector<int> > indices(newLabelNum);
	Mat indexMap(mask.rows, mask.cols, CV_32S);
	for (int i = 0; i < mask.rows; i++)  {
		for (int j = 0; j < mask.cols; j++)  {
			if(aggregatedLabels.at<int>(i, j) < 0 || newLabels.at(aggregatedLabels.at<int>(i, j)) < 0) {
				continue;
			}
			auto v = hueToRgb((float)newLabels.at(aggregatedLabels.at<int>(i, j)) / newLabelNum);
			colorLabels.at<Vec3f>(i, j) = v;
			meshes.at(newLabels.at(aggregatedLabels.at<int>(i, j))).push_back(points3d.at<Vec3f>(i, j));
			indexMap.at<int>(i, j) = meshes.at(newLabels.at(aggregatedLabels.at<int>(i, j))).size();
		}
	}
	for (int i = 0; i < mask.rows; i++)  {
		for (int j = 0; j < mask.cols; j++)  {
			if(aggregatedLabels.at<int>(i, j) < 0 || newLabels.at(aggregatedLabels.at<int>(i, j)) < 0) {
				continue;
			}
			if(i + 1 < mask.rows &&
				j + 1 < mask.cols &&
				aggregatedLabels.at<int>(i, j) == aggregatedLabels.at<int>(i + 1, j) &&
				aggregatedLabels.at<int>(i, j) == aggregatedLabels.at<int>(i, j + 1) &&
				aggregatedLabels.at<int>(i, j) == aggregatedLabels.at<int>(i + 1, j + 1)) {
					indices.at(newLabels.at(aggregatedLabels.at<int>(i, j))).push_back(indexMap.at<int>(i, j));
					indices.at(newLabels.at(aggregatedLabels.at<int>(i, j))).push_back(indexMap.at<int>(i+1, j));
					indices.at(newLabels.at(aggregatedLabels.at<int>(i, j))).push_back(indexMap.at<int>(i, j+1));
					indices.at(newLabels.at(aggregatedLabels.at<int>(i, j))).push_back(indexMap.at<int>(i, j+1));
					indices.at(newLabels.at(aggregatedLabels.at<int>(i, j))).push_back(indexMap.at<int>(i+1, j));
					indices.at(newLabels.at(aggregatedLabels.at<int>(i, j))).push_back(indexMap.at<int>(i+1, j+1));
			}
		}
	}
	imshow("clustered", colorLabels);
	//Mat colorLabelsSave;
	//colorLabels = colorLabels * 255;
	//colorLabels.convertTo(colorLabelsSave, CV_8UC3);
	//imwrite("clustered.png", colorLabelsSave);
	imshow("p", points3d);

	waitKey(0);

	for(int i = 0; i < meshes.size(); i++) {
		ofstream myfile;
		myfile.open (to_string(i) + "mesh.obj");
		for(int j = 0; j < meshes.at(i).size(); j++) {
			auto & v = meshes.at(i).at(j);
			myfile << "v " << to_string(v.x) << " " << to_string(v.y) << " " << to_string(v.z) << endl;
		}
		for(int j = 0; j < indices.at(i).size(); j+=3) {
			myfile << "f " << to_string(indices.at(i).at(j)) << "// " << to_string(indices.at(i).at(j+1)) << "// " << to_string(indices.at(i).at(j+2)) << "//" << endl;
		}
		myfile.close();
	}
	return 0;
}
