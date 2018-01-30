/**
* @brief 型定義
*/
#pragma once

using mat1_t = cv::Mat_<unsigned char>; //! グレイスケール画像の型
using mat3_t = cv::Mat_<cv::Vec3b>; //! カラー画像の型
using sendData_t = std::array<double, 4>; //! 送信データの型
using recvData_t = std::array<double, 3>; //! 受信データの型

//! @brief 表示データの型
struct dispData_t {
	unsigned long frameCount;
	mat1_t image;
	cv::Point2d centroid;
};

constexpr std::size_t capacity = 3; //! spsc_queueのサイズ
template <typename T>
using spsc_queue = boost::lockfree::spsc_queue<T, boost::lockfree::capacity<capacity>>; //! スレッド間通信用のロックフリーキューの型