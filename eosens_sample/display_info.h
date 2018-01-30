/**
* @brief　画像，重心位置，処理速度の表示
*/

#pragma once

/**
* @struct DispInfo
* @brief 情報表示クラス
*/
struct DispInfo {
private:
	spsc_queue<dispData_t> &queDispInfo_; //! 表示データのロックフリーキュー
	const unsigned int height_; //! 表示画像の高さ
	const unsigned int width_; //! 表示画像の幅

public:
	/**
	* @brief コンストラクタ
	* @param[in] queDispInfo 表示データ 
	* @param[in] width 画像の幅
	* @param[in] height 画像の高さ
	*/
	DispInfo(decltype(queDispInfo_) &queDispInfo, const unsigned int width, const unsigned int height)
		: queDispInfo_(queDispInfo)
		, width_(width), height_(height)
	{}

	/**
	* @brief 画像，トラッキング位置，処理速度表示
	*
	* @param[out] isSaveImage 画像を保存する(s)
	*
	* @note 画像が表示されたウインドウをフォーカスした状態でボタンを押すことで下記の様に動作．
	* - "q": プログラム終了
	* - "s": 画像を保存開始(デバッグ用)
	* - "S": 画像保存を停止
	*/
	const int operator()(std::atomic<bool> &isSaveImage) const
	{
		const std::string winName = "image";
		cv::namedWindow(winName);

		dispData_t dispData;
		mat1_t binImage = mat1_t::zeros(height_, width_);
		mat3_t dispImage = mat3_t::zeros(height_, width_);
		unsigned long frameCount = 0, prevFrameCount = 0;
		const auto red = cv::Scalar(71, 99, 255);

		using timeT = std::chrono::high_resolution_clock;
		auto startT = timeT::now();

		cv::Point2d cog;
		std::string winTitle = "";
		long count = 0;
		while (true) {
			//データ読み込み
			if (!queDispInfo_.pop(dispData)) {
				std::this_thread::yield();
				continue;
			}

			//表示
			frameCount = dispData.frameCount;
			cog = dispData.centroid;

			//重心位置表示のため，二値化画像の場合はカラー画像へ変換
			if (dispData.image.channels() == 1) {
				dispData.image.copyTo(binImage);
				cv::cvtColor(binImage, dispImage, cv::COLOR_GRAY2BGR);
			}
			else if (dispData.image.channels() == 3) {
				dispData.image.copyTo(dispImage);
			}
			cv::circle(dispImage, cog, 5, red, 2);
			cv::setWindowTitle(winName, winTitle);
			cv::imshow(winName, dispImage);
			auto key = cv::waitKey(1);
			if (key == 'q') {
				break;
			}
			else if (key == 's') {
				isSaveImage = true;
			}
			else if (key == 'S') {
				isSaveImage = false;
			}
			if (count % 60 == 0) {
				const auto elapsed_ms = std::chrono::duration_cast<std::chrono::microseconds>(timeT::now() - startT).count() / 1000;
				startT = timeT::now();
				auto fps = (frameCount - prevFrameCount) * 1000.0 / elapsed_ms;
				std::cout << std::to_string(frameCount) + "frame," + std::to_string(fps) + "[fps], COG: (" + std::to_string(cog.x) + ", " + std::to_string(cog.y) + ")" << std::endl;
				prevFrameCount = frameCount;
			}
			count++;
		}
		return 0;
	}
};