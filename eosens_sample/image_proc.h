/**
* @brief�@�摜����
*/
#pragma once



// Define if images are to be saved.
// '0'- no; '1'- yes.
#define saveImages 0
// Define if video is to be recorded.
// '0'- no; '1'- yes.
#define recordVideo 0

// Include files to use the PYLON API.
#include <pylon/PylonIncludes.h>
#ifdef PYLON_WIN_BUILD
#    include <pylon/PylonGUI.h>
#include "Pylon_with_OpenCV.h"
#endif
// Namespace for using pylon objects.
using namespace Pylon;

// Namespace for using OpenCV objects.
using namespace cv;

// Namespace for using cout.
using namespace std;

// Number of images to be grabbed.
static const uint32_t c_countOfImagesToGrab = 100000;


/**
* @struct ImageProc
* @brief�@�摜�����N���X
*/
struct ImageProc {
private:
	spsc_queue<dispData_t> &queDispData_; //! �\���f�[�^�̃��b�N�t���[�L���[
	spsc_queue<sendData_t> &queSend_; //! ���M�f�[�^�̃��b�N�t���[�L���[
	spsc_queue<recvData_t> &queRecv_; //! ��M�f�[�^�̃��b�N�t���[�L���[

	bool isFinished_; //! �I���t���O
	const unsigned int height_; //! �摜�̍���
	const unsigned int width_; //! �摜�̕�

public:
	/**
	* @brief �R���X�g���N�^
	* @param[out] queDispData �\���f�[�^
	* @param[out] queSend dSpace�ւ̑��M�f�[�^
	* @param[in] queRecv dSpace����̎�M�f�[�^
	* @param[in] width �摜�̕�
	* @param[in] height �摜�̍���
	*/
	ImageProc(decltype(queDispData_) &queDispData, decltype(queSend_) &queSend, decltype(queRecv_) &queRecv, const unsigned int width, const unsigned int height)
		: queDispData_(queDispData)
		, queSend_(queSend), queRecv_(queRecv)
		, width_(width), height_(height), isFinished_(false)
	{}

	/**
	* @brief �摜�����i����̐F�̉摜�d�S���g���b�L���O�j
	* @param[in] isSaveImage �摜��ۑ����� ���O�ɕۑ���̃f�B���N�g�����쐬�̂���
	*
	* @note OpenMP�̃X���b�h��(omp_set_num_threads)�͊����ƂɍœK�Ȓl���قȂ邽�߁C�œK�Ȃ��̂�T���Đݒ肵�Ă�������
	*/
	int operator()(const std::atomic<bool> &isSaveImage) const
	{
		//臒l��
		int momThresh = 1; //! �����Ă���ʐς��������ꍇ�͖���
		//���F���E���^���{�[��
		//const int minThreshBall[3] = { 90,130,90 }; //! HSV臒l�̉���
		//const int maxThreshBall[3] = { 110,240,255 };  //! HSV臒l�̏��
		const int maxSaveNum = 500; //! �f�o�b�O�p�ɉ摜���o�͂���t���[����
		//�X�e�b�J�[
		//const int minThreshBall[3] = { 110,120,0 }; //! HSV臒l�̉���
		//const int maxThreshBall[3] = { 128,255,255 };  //! HSV臒l�̏��
		//���F�X�e�b�J�[
		//const int minThreshBall[3] = { 30,200,100 };//! HSV臒l�̉���
		//const int maxThreshBall[3] = { 36,255,255 };  //! HSV臒l�̏��
		//���[�r�b�N�L���[�u�̉��F
		//const int minThreshBall[3] = { 25,180,90}; //! HSV臒l�̉���
		//const int maxThreshBall[3] = { 35,255,180 };  //! HSV臒l�̏��
		//�I�����W�w�T�b�N
		const int minThreshBall[3] = { 1,180,40}; //! HSV臒l�̉���
		const int maxThreshBall[3] = { 25,255,150 };  //! HSV臒l�̏��
		const int h = height_;
		const int w = width_;
		const int threads = 16;

		mat1_t srcImage = mat1_t::zeros(h, w);
		mat3_t rgbImage = mat3_t::zeros(h, w);
		mat1_t binImage = mat1_t::zeros(h, w);
		mat1_t maskImage = mat1_t::zeros(h, w);
		cv::Point2d cog;
		double isDetected = 1; //0:�d�S�ʒu��dSpace�֑���Ȃ��C1:����
		cv::Rect roi = cv::Rect(0, 0, w, h); //�d�S�v�Z��ROI�Ή���������͑S��f����

		//�摜�ۑ��֌W
		std::array<mat1_t, maxSaveNum> saveImage;
		for (auto sImage : saveImage) {
			sImage = mat1_t::zeros(h, w);
		}
		int saveCount = 0;

		sendData_t objInfo;
		dispData_t dispData;

		//EoSens cam(width_, height_);
		//auto cam = std::make_unique<EoSens>(width_, height_);

		//basler cam�̏���
		int exitCode = 0;

		// Automagically call PylonInitialize and PylonTerminate to ensure the pylon runtime system
		// is initialized during the lifetime of this object.
		Pylon::PylonAutoInitTerm autoInitTerm;

		// Create an instant camera object with the camera device found first.
		CInstantCamera camera(CTlFactory::GetInstance().CreateFirstDevice());

		// Print the model name of the camera.
		cout << "Using device " << camera.GetDeviceInfo().GetVendorName() << " " << camera.GetDeviceInfo().GetModelName() << endl;

		// Get a camera nodemap in order to access camera parameters.
		GenApi::INodeMap& nodemap = camera.GetNodeMap();
		// Open the camera before accessing any parameters.
		camera.Open();
		// Create pointers to access the camera Width and Height parameters.
		GenApi::CIntegerPtr width = nodemap.GetNode("Width");
		GenApi::CIntegerPtr height = nodemap.GetNode("Height");

		// The parameter MaxNumBuffer can be used to control the count of buffers
		// allocated for grabbing. The default value of this parameter is 10.
		camera.MaxNumBuffer = 10;

		// Create a pylon ImageFormatConverter object.
		CImageFormatConverter formatConverter;
		// Specify the output pixel format.
		formatConverter.OutputPixelFormat = PixelType_RGB8packed;
		// Create a PylonImage that will be used to create OpenCV images later.
		CPylonImage pylonImage;
		// Declare an integer variable to count the number of grabbed images
		// and create image file names with ascending number.
		int grabbedImages = 0;

		// Create an OpenCV video creator.
		VideoWriter cvVideoCreator;
		// Create an OpenCV image.
		Mat openCvImage;

		// Define the video file name.
		std::string videoFileName = "openCvVideo.avi";

		// Define the video frame size.
		cv::Size frameSize = Size((int)width->GetValue(), (int)height->GetValue());

		// Set the codec type and the frame rate. You have 3 codec options here.
		// The frame rate should match or be lower than the camera acquisition frame rate.
		cvVideoCreator.open(videoFileName, CV_FOURCC('D', 'I', 'V', 'X'), 30, frameSize, true);
		//cvVideoCreator.open(videoFileName, CV_FOURCC('M','P','4','2'), 20, frameSize, true); 
		//cvVideoCreator.open(videoFileName, CV_FOURCC('M','J','P','G'), 20, frameSize, true);

		// Start the grabbing of c_countOfImagesToGrab images.
		// The camera device is parameterized with a default configuration which
		// sets up free-running continuous acquisition.
		camera.StartGrabbing(c_countOfImagesToGrab, GrabStrategy_LatestImageOnly);

		// This smart pointer will receive the grab result data.
		CGrabResultPtr ptrGrabResult;

		// Camera.StopGrabbing() is called automatically by the RetrieveResult() method
		// when c_countOfImagesToGrab images have been retrieved.



		//Lookup table �쐬
		std::array<mat1_t, 256> Lut; //! RGB����HSV臒l�œ�l���܂ł�lookup table
		for (int i = 0; i < Lut.size(); i++) {
			Lut[i] = mat1_t::zeros(256, 256);
		}
		generateLUTRGB2HSVBin(Lut, minThreshBall, maxThreshBall);
		omp_set_num_threads(16); // �X���b�h���͍ŋ߂�PC�Ȃ��{�I�ɂ�16�ʂł悢���C�Â�PC��PC�ɂ���Ă̓X���b�h�𑽂�����ƒx���Ȃ�

		//���C������
		for (unsigned long fCount = 0; !isFinished_; ) {

			//�摜�Ǎ���
			/*if (cam->getFrame(srcImage) != 0) {
				continue;
			}*/
			//basler �摜�ǂݍ��݁@RGB
			// Wait for an image and then retrieve it. A timeout of 5000 ms is used.
			camera.RetrieveResult(5000, ptrGrabResult, TimeoutHandling_ThrowException);

			// Image grabbed successfully?
			if (ptrGrabResult->GrabSucceeded())
			{
				// Access the image data.
				//cout << "SizeX: " << ptrGrabResult->GetWidth() << endl;
				//cout << "SizeY: " << ptrGrabResult->GetHeight() << endl;

				// Convert the grabbed buffer to a pylon image.
				formatConverter.Convert(pylonImage, ptrGrabResult);

				// Create an OpenCV image from a pylon image.
				rgbImage = cv::Mat(ptrGrabResult->GetHeight(), ptrGrabResult->GetWidth(), CV_8UC3, (uint8_t *)pylonImage.GetBuffer());
			}

			fCount++;

			//�摜����
			//�F��ԁ@Bayer -> RGB
			//pvcore::__convertColorCPU((uint8_t *)pylonImage.GetBuffer(), rgbImage.data, w, h, w, 3 * w, bayerbg2rgb_8u, threads);
			//���b�N�A�b�v�e�[�u����p������l���@RGB(->HSV->�F���o)-> binary
#pragma omp parallel for
			for (int i_y = 0; i_y < h; i_y++) {
				auto rgbPtr = rgbImage[i_y];
				auto binPtr = binImage[i_y];
#pragma omp parallel for
				for (int i_x = 0; i_x < w; i_x++) {
					auto pixC = rgbPtr[i_x];
					binPtr[i_x] = Lut[pixC[0]][pixC[2]][pixC[1]];
				}
			}


			cv::morphologyEx(binImage, maskImage, cv::MORPH_OPEN, cv::Mat());// , cv::Point(-1, -1), 2); //�m�C�Y�����@�I�[�v�j���O

			//�d�S�v�Z
			CALC_COG_TBB COGTbb(maskImage.data, w, h, threads, roi);
			tbb::parallel_reduce(tbb::blocked_range<size_t>(0, threads, 1),
				COGTbb);// , tbb::auto_partitioner());

			if (COGTbb.mom0 > momThresh) {
				cog = cv::Point2d(COGTbb.mom10, COGTbb.mom01) / COGTbb.mom0;
				isDetected = 1;
			}
			else {
				cog = cv::Point2d(0, 0);
				isDetected = 0;
			}

			//�d�S�ʒu���𑗐M�p�z��Ɋi�[���đ��M�X���b�h�ɓn��
			objInfo[0] = isDetected;
			objInfo[1] = cog.x;
			objInfo[2] = cog.y;
			objInfo[3] = 1; //dummy
			queSend_.push(objInfo);

			//�\���f�[�^���i�[���ĕ\���X���b�h�֓n��
			//100�񂲂�
			if (fCount %  10 == 0) {
				dispData.frameCount = fCount;
				maskImage.copyTo(dispData.image);
				dispData.centroid = cog;
				queDispData_.push(dispData);
			}
			//�摜�o�͂̂��߃o�b�t�@�ɕۑ�
			if (isSaveImage && saveCount < maxSaveNum) {
				srcImage.copyTo(saveImage[saveCount]);
				saveCount++;
			}
		}

		//�摜�o��
		//�ۑ���̃f�B���N�g�������O�ɏ���
		if (saveCount != 0) {
			const std::string dir = "./img/";
			mat3_t outputImage = mat3_t::zeros(height_, width_);
			bool isOutputGrayscale = (saveImage[0].channels() == 1);
			std::cout << "�摜�ۑ�";
			for (int i = 0; i < saveCount; i++) {
				if (isOutputGrayscale) {
					cv::cvtColor(saveImage[i], outputImage, cv::COLOR_GRAY2BGR);
				}
				else {
					saveImage[i].copyTo(outputImage);
				}
				cv::imwrite(dir + std::to_string(i) + ".bmp", outputImage);
			}
			std::cout << "�I��" << std::endl;
		}

		return 0;
	}

	/**
	* @brief RGB����HSV�ɕϊ�����臒l�œ�l�����郋�b�N�A�b�v�e�[�u���̍쐬
	*
	* @param [out] Lut ���b�N�A�b�v�e�[�u���@256x256��Mat��256��array�����O�ɗp��
	* @param [in] minThresh HSV�̓�l��臒l�̉����l
	* @param [in] maxThresh HSV�̓�l��臒l�̏���l
	* @retval 0 ���b�N�A�b�v�e�[�u����������
	*/
	int generateLUTRGB2HSVBin(std::array<mat1_t, 256> &Lut, const int(&minThresh)[3], const int(&maxThresh)[3]) const {

		for (int i_r = 0; i_r < 256; i_r++) {
			for (int i_g = 0; i_g < 256; i_g++) {
				for (int i_b = 0; i_b < 256; i_b++) {
					//RGB->HSV���v�Z
					//S�̌v�Z�Ŏ����Ōv�Z���Ăق�������double
					double maxC = std::max({ i_r, i_g, i_b });
					double minC = std::min({ i_r, i_g, i_b });

					unsigned char hsv[3];

					if (maxC == minC) {
						hsv[0] = 0; //H
						hsv[1] = static_cast<unsigned char>(maxC); //S
						hsv[2] = 0; //V
					}
					else {

						hsv[2] = static_cast<unsigned char>(maxC); //V
						hsv[1] = static_cast<unsigned char>(((maxC - minC) / maxC) * 255); //S
																						   //H
						int hue;
						if (maxC == minC) {
							hue = 0;
						}
						else if (maxC == i_r) {
							hue = 60 * (i_g - i_b) / (maxC - minC);
						}
						else if (maxC == i_g) {
							hue = 60 * (i_b - i_r) / (maxC - minC) + 120;
						}
						else {
							hue = 60 * (i_r - i_g) / (maxC - minC) + 240;
						}
						//H�͈̔͂�0����360
						if (hue < 0) {
							hue += 360;
						}
						else if (hue > 360) {
							hue -= 360;
						}
						//OpenCV�̎����ɍ��킹�āCH�͈̔͂�0����180�ɐݒ�
						hsv[0] = static_cast<unsigned char>(hue / 2);
					}

					//H臒l�œ�l��(0 or 255)
					if (minThresh[0] < maxThresh[0]) {
						if (hsv[0] >= minThresh[0] && hsv[0] <= maxThresh[0] && hsv[1] >= minThresh[1] && hsv[1] <= maxThresh[1] && hsv[2] >= minThresh[2] && hsv[2] <= maxThresh[2]) {
							Lut[i_r].at<unsigned char>(i_b, i_g) = 255;
						}
						else {
							Lut[i_r].at<unsigned char>(i_b, i_g) = 0;
						}
					}
					else {
						if (hsv[0] >= minThresh[0] || hsv[0] <= maxThresh[0] && hsv[1] >= minThresh[1] && hsv[1] <= maxThresh[1] && hsv[2] >= minThresh[2] && hsv[2] <= maxThresh[2]) {
							Lut[i_r].at<unsigned char>(i_b, i_g) = 255;
						}
						else {
							Lut[i_r].at<unsigned char>(i_b, i_g) = 0;
						}
					}
				}
			}
		}

		return 0;
	}
	/**
	* @brief �����̏I��
	*/
	void finishProc() {
		isFinished_ = true;
	}

	/**
	* @brief �I���t���O�̊m�F
	*/
	bool isFinishProc() {
		return isFinished_;
	}
};