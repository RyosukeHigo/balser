/**
* @brief　画像処理
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
* @brief　画像処理クラス
*/
struct ImageProc {
private:
	spsc_queue<dispData_t> &queDispData_; //! 表示データのロックフリーキュー
	spsc_queue<sendData_t> &queSend_; //! 送信データのロックフリーキュー
	spsc_queue<recvData_t> &queRecv_; //! 受信データのロックフリーキュー

	bool isFinished_; //! 終了フラグ
	const unsigned int height_; //! 画像の高さ
	const unsigned int width_; //! 画像の幅

public:
	/**
	* @brief コンストラクタ
	* @param[out] queDispData 表示データ
	* @param[out] queSend dSpaceへの送信データ
	* @param[in] queRecv dSpaceからの受信データ
	* @param[in] width 画像の幅
	* @param[in] height 画像の高さ
	*/
	ImageProc(decltype(queDispData_) &queDispData, decltype(queSend_) &queSend, decltype(queRecv_) &queRecv, const unsigned int width, const unsigned int height)
		: queDispData_(queDispData)
		, queSend_(queSend), queRecv_(queRecv)
		, width_(width), height_(height), isFinished_(false)
	{}

	/**
	* @brief 画像処理（既定の色の画像重心をトラッキング）
	* @param[in] isSaveImage 画像を保存する 事前に保存先のディレクトリを作成のこと
	*
	* @note OpenMPのスレッド数(omp_set_num_threads)は環境ごとに最適な値が異なるため，最適なものを探して設定してください
	*/
	int operator()(const std::atomic<bool> &isSaveImage) const
	{
		//閾値等
		int momThresh = 1; //! 見えている面積が小さい場合は無視
		//黄色いウレタンボール
		//const int minThreshBall[3] = { 90,130,90 }; //! HSV閾値の下限
		//const int maxThreshBall[3] = { 110,240,255 };  //! HSV閾値の上限
		const int maxSaveNum = 500; //! デバッグ用に画像を出力するフレーム数
		//青ステッカー
		//const int minThreshBall[3] = { 110,120,0 }; //! HSV閾値の下限
		//const int maxThreshBall[3] = { 128,255,255 };  //! HSV閾値の上限
		//黄色ステッカー
		//const int minThreshBall[3] = { 30,200,100 };//! HSV閾値の下限
		//const int maxThreshBall[3] = { 36,255,255 };  //! HSV閾値の上限
		//ルービックキューブの黄色
		//const int minThreshBall[3] = { 25,180,90}; //! HSV閾値の下限
		//const int maxThreshBall[3] = { 35,255,180 };  //! HSV閾値の上限
		//オレンジ指サック
		const int minThreshBall[3] = { 1,180,40}; //! HSV閾値の下限
		const int maxThreshBall[3] = { 25,255,150 };  //! HSV閾値の上限
		const int h = height_;
		const int w = width_;
		const int threads = 16;

		mat1_t srcImage = mat1_t::zeros(h, w);
		mat3_t rgbImage = mat3_t::zeros(h, w);
		mat1_t binImage = mat1_t::zeros(h, w);
		mat1_t maskImage = mat1_t::zeros(h, w);
		cv::Point2d cog;
		double isDetected = 1; //0:重心位置をdSpaceへ送らない，1:送る
		cv::Rect roi = cv::Rect(0, 0, w, h); //重心計算がROI対応だが今回は全画素処理

		//画像保存関係
		std::array<mat1_t, maxSaveNum> saveImage;
		for (auto sImage : saveImage) {
			sImage = mat1_t::zeros(h, w);
		}
		int saveCount = 0;

		sendData_t objInfo;
		dispData_t dispData;

		//EoSens cam(width_, height_);
		//auto cam = std::make_unique<EoSens>(width_, height_);

		//basler camの処理
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



		//Lookup table 作成
		std::array<mat1_t, 256> Lut; //! RGBからHSV閾値で二値化までのlookup table
		for (int i = 0; i < Lut.size(); i++) {
			Lut[i] = mat1_t::zeros(256, 256);
		}
		generateLUTRGB2HSVBin(Lut, minThreshBall, maxThreshBall);
		omp_set_num_threads(16); // スレッド数は最近のPCなら基本的には16位でよいが，古いPCはPCによってはスレッドを多くすると遅くなる

		//メイン処理
		for (unsigned long fCount = 0; !isFinished_; ) {

			//画像読込み
			/*if (cam->getFrame(srcImage) != 0) {
				continue;
			}*/
			//basler 画像読み込み　RGB
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

			//画像処理
			//色空間　Bayer -> RGB
			//pvcore::__convertColorCPU((uint8_t *)pylonImage.GetBuffer(), rgbImage.data, w, h, w, 3 * w, bayerbg2rgb_8u, threads);
			//ルックアップテーブルを用いた二値化　RGB(->HSV->色抽出)-> binary
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


			cv::morphologyEx(binImage, maskImage, cv::MORPH_OPEN, cv::Mat());// , cv::Point(-1, -1), 2); //ノイズ除去　オープニング

			//重心計算
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

			//重心位置等を送信用配列に格納して送信スレッドに渡し
			objInfo[0] = isDetected;
			objInfo[1] = cog.x;
			objInfo[2] = cog.y;
			objInfo[3] = 1; //dummy
			queSend_.push(objInfo);

			//表示データを格納して表示スレッドへ渡し
			//100回ごと
			if (fCount %  10 == 0) {
				dispData.frameCount = fCount;
				maskImage.copyTo(dispData.image);
				dispData.centroid = cog;
				queDispData_.push(dispData);
			}
			//画像出力のためバッファに保存
			if (isSaveImage && saveCount < maxSaveNum) {
				srcImage.copyTo(saveImage[saveCount]);
				saveCount++;
			}
		}

		//画像出力
		//保存先のディレクトリを事前に準備
		if (saveCount != 0) {
			const std::string dir = "./img/";
			mat3_t outputImage = mat3_t::zeros(height_, width_);
			bool isOutputGrayscale = (saveImage[0].channels() == 1);
			std::cout << "画像保存";
			for (int i = 0; i < saveCount; i++) {
				if (isOutputGrayscale) {
					cv::cvtColor(saveImage[i], outputImage, cv::COLOR_GRAY2BGR);
				}
				else {
					saveImage[i].copyTo(outputImage);
				}
				cv::imwrite(dir + std::to_string(i) + ".bmp", outputImage);
			}
			std::cout << "終了" << std::endl;
		}

		return 0;
	}

	/**
	* @brief RGBからHSVに変換して閾値で二値化するルックアップテーブルの作成
	*
	* @param [out] Lut ルックアップテーブル　256x256のMatの256のarrayを事前に用意
	* @param [in] minThresh HSVの二値化閾値の下限値
	* @param [in] maxThresh HSVの二値化閾値の上限値
	* @retval 0 ルックアップテーブル生成成功
	*/
	int generateLUTRGB2HSVBin(std::array<mat1_t, 256> &Lut, const int(&minThresh)[3], const int(&maxThresh)[3]) const {

		for (int i_r = 0; i_r < 256; i_r++) {
			for (int i_g = 0; i_g < 256; i_g++) {
				for (int i_b = 0; i_b < 256; i_b++) {
					//RGB->HSVを計算
					//Sの計算で実数で計算してほしいためdouble
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
						//Hの範囲は0から360
						if (hue < 0) {
							hue += 360;
						}
						else if (hue > 360) {
							hue -= 360;
						}
						//OpenCVの実装に合わせて，Hの範囲は0から180に設定
						hsv[0] = static_cast<unsigned char>(hue / 2);
					}

					//H閾値で二値化(0 or 255)
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
	* @brief 処理の終了
	*/
	void finishProc() {
		isFinished_ = true;
	}

	/**
	* @brief 終了フラグの確認
	*/
	bool isFinishProc() {
		return isFinished_;
	}
};