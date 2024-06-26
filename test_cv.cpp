#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>  
#include <opencv2/imgproc/imgproc.hpp>  
#include <opencv2/core/core.hpp> 
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/flann/flann.hpp>
#include <opencv2/core/ocl.hpp> 
#include <iostream>
#include <string>
 
#define SRC_IMG "moon.jpg"
#define TMP_IMG "moon-2.jpg"
 
using namespace cv;
using namespace cv::ocl;
using namespace std;
 
void initOpenCL();
void runMatchGrayUseCpu(int method);
void runMatchGrayUseGpu(int method);
 
int main(int argc, char **argv){
 
	initOpenCL();
	int method = TM_SQDIFF;
    std::cout << "main method TM_SQDIFF: " << TM_SQDIFF << std::endl;

	runMatchGrayUseCpu(method);
	runMatchGrayUseGpu(method);
	return 0;
}
 
void initOpenCL(){
 
	// launch OpenCL environment...  
	std::vector<cv::ocl::PlatformInfo> plats;
	cv::ocl::getPlatfomsInfo(plats);
	const cv::ocl::PlatformInfo *platform = &plats[0];
	std::cout << "Platform name: " << platform->name().c_str() << std::endl;
	std::cout << "OpenCL CL_PLATFORM_VERSION: " << platform->version().c_str() << std::endl;
	std::cout << "OpenCL CL_PLATFORM_VENDOR: " << platform->vendor().c_str() << std::endl;
	std::cout << "OpenCL deviceNumber: " << platform->deviceNumber() << std::endl;
	cv::ocl::Device current_device;
	platform->getDevice(current_device, 0);
	std::cout << "Device name: " << current_device.name().c_str() << std::endl;
	//current_device.set(0);
	//std::cout << "Set device to 0 " << std::endl;
	
	bool is_have_opencl		= cv::ocl::haveOpenCL();
	std::cout << "is_have_opencl:"		<< is_have_opencl	<< std::endl;
	bool is_have_svm		= cv::ocl::haveSVM();
	std::cout << "is_have_svm:"			<< is_have_svm		<< std::endl;
	bool is_use_opencl		= cv::ocl::useOpenCL();
	std::cout << "is_use_opencl:"		<< is_use_opencl	<< std::endl;
	bool is_have_amd_blas	= cv::ocl::haveAmdBlas();
	std::cout << "is_have_amd_blas:"	<< is_have_amd_blas << std::endl;
	bool is_have_amd_fft	= cv::ocl::haveAmdFft();
	std::cout << "is_have_amd_fft:"		<< is_have_amd_fft	<< std::endl;
 
	cv::ocl::setUseOpenCL(true);
}
 
void runMatchGrayUseCpu(int method){
 
	std::cout << "===Test Match Template Use CPU===" << " method: " << method << std::endl;
 
	double t = 0.0;
 
	cv::Mat src = cv::imread(SRC_IMG, 1);
	cv::Mat tmp = cv::imread(TMP_IMG, 1);
 
	cv::Mat gray_src, gray_tmp;
	if (src.channels() == 1) gray_src = src;
    else cv::cvtColor(src, gray_src, COLOR_RGB2GRAY);
	if (tmp.channels() == 1) gray_tmp = tmp;
	else cv::cvtColor(tmp, gray_tmp, COLOR_RGB2GRAY);
 

    std::cout << "gray_src.cols:" << gray_src.cols << " gray_src.rows:" << gray_src.rows << std::endl;
    std::cout << "gray_tmp.cols:" << gray_tmp.cols << " gray_tmp.rows:" << gray_tmp.rows << std::endl;
	int result_cols = gray_src.cols - gray_tmp.cols + 1;
	int result_rows = gray_src.rows - gray_tmp.rows + 1;
    std::cout << "result_cols:" << result_cols << " result_rows:" << result_rows << std::endl;

	cv::Mat result = cv::Mat(result_cols, result_rows, CV_32FC1);
 
	t = (double)cv::getTickCount();
	cv::matchTemplate(gray_src, gray_tmp, result, method);
	t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
 
	cv::Point point;
	double minVal, maxVal;
	cv::Point minLoc, maxLoc;
	cv::minMaxLoc(result, &minVal, &maxVal, &minLoc, &maxLoc, cv::Mat());
	switch (method){
 
	case TM_SQDIFF:
		point = minLoc;
		break;
	case TM_SQDIFF_NORMED:
		point = minLoc;
		break;
	case TM_CCORR:
	case TM_CCOEFF:
		point = maxLoc;
		break;
	case TM_CCORR_NORMED:
	case TM_CCOEFF_NORMED:
	default:
		point = maxLoc;
		break;
	}
	
 
	std::cout << "CPU time :" << t << " second" << std::endl;
	std::cout << "obj.x :" << point.x << " obj.y :" << point.y << std::endl;
	std::cout << " " << std::endl;
}
 
void runMatchGrayUseGpu(int method){
 
	std::cout << "===Test Match Template Use GPU===" << std::endl;
 
	double t = 0.0;
 
	cv::UMat src = cv::imread(SRC_IMG, 1).getUMat(cv::ACCESS_RW);
	cv::UMat tmp = cv::imread(TMP_IMG, 1).getUMat(cv::ACCESS_RW);
	cv::UMat gray_src, gray_tmp;
	
	
	if (src.channels() == 1) gray_src = src;
    else cv::cvtColor(src, gray_src, COLOR_RGB2GRAY);
	if (tmp.channels() == 1) gray_tmp = tmp;
	else cv::cvtColor(tmp, gray_tmp, COLOR_RGB2GRAY);
 
	int result_cols = gray_src.cols - gray_tmp.cols + 1;
	int result_rows = gray_src.rows - gray_tmp.rows + 1;
	cv::UMat result = cv::UMat(result_cols, result_rows, CV_32FC1);
 
 
	t = (double)cv::getTickCount();
	cv::matchTemplate(gray_src, gray_tmp, result, method);
	t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
 
	cv::Point point;
	double minVal, maxVal;
	cv::Point minLoc, maxLoc;
	cv::minMaxLoc(result, &minVal, &maxVal, &minLoc, &maxLoc, cv::UMat());
	
	switch (method){
 
	case TM_SQDIFF:
		point = minLoc;
		break;
	case TM_SQDIFF_NORMED:
		point = minLoc;
		break;
	case TM_CCORR:
	case TM_CCOEFF:
		point = maxLoc;
		break;
	case TM_CCORR_NORMED:
	case TM_CCOEFF_NORMED:
	default:
		point = maxLoc;
		break;
	}
	
	std::cout << "GPU time :" << t << " second" << std::endl;
	std::cout << "obj.x :" << point.x << " obj.y :" << point.y << std::endl;
	std::cout << " " << std::endl;
}
