/**
* @brief�@�摜�C�d�S�ʒu�C�������x�̕\��
*/

#pragma once

/**
* @struct DispInfo
* @brief ���\���N���X
*/
struct DispInfo {
private:
	spsc_queue<dispData_t> &queDispInfo_; //! �\���f�[�^�̃��b�N�t���[�L���[
	const unsigned int height_; //! �\���摜�̍���
	const unsigned int width_; //! �\���摜�̕�

public:
	/**
	* @brief �R���X�g���N�^
	* @param[in] queDispInfo �\���f�[�^ 
	* @param[in] width �摜�̕�
	* @param[in] height �摜�̍���
	*/
	DispInfo(decltype(queDispInfo_) &queDispInfo, const unsigned int width, const unsigned int height)
		: queDispInfo_(queDispInfo)
		, width_(width), height_(height)
	{}

	/**
	* @brief �摜�C�g���b�L���O�ʒu�C�������x�\��
	*
	* @param[out] isSaveImage �摜��ۑ�����(s)
	*
	* @note �摜���\�����ꂽ�E�C���h�E���t�H�[�J�X������ԂŃ{�^�����������Ƃŉ��L�̗l�ɓ���D
	* - "q": �v���O�����I��
	* - "s": �摜��ۑ��J�n(�f�o�b�O�p)
	* - "S": �摜�ۑ����~
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
			//�f�[�^�ǂݍ���
			if (!queDispInfo_.pop(dispData)) {
				std::this_thread::yield();
				continue;
			}

			//�\��
			frameCount = dispData.frameCount;
			cog = dispData.centroid;

			//�d�S�ʒu�\���̂��߁C��l���摜�̏ꍇ�̓J���[�摜�֕ϊ�
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