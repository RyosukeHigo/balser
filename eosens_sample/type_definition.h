/**
* @brief �^��`
*/
#pragma once

using mat1_t = cv::Mat_<unsigned char>; //! �O���C�X�P�[���摜�̌^
using mat3_t = cv::Mat_<cv::Vec3b>; //! �J���[�摜�̌^
using sendData_t = std::array<double, 4>; //! ���M�f�[�^�̌^
using recvData_t = std::array<double, 3>; //! ��M�f�[�^�̌^

//! @brief �\���f�[�^�̌^
struct dispData_t {
	unsigned long frameCount;
	mat1_t image;
	cv::Point2d centroid;
};

constexpr std::size_t capacity = 3; //! spsc_queue�̃T�C�Y
template <typename T>
using spsc_queue = boost::lockfree::spsc_queue<T, boost::lockfree::capacity<capacity>>; //! �X���b�h�ԒʐM�p�̃��b�N�t���[�L���[�̌^