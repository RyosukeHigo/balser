/**
* @mainpage
* 
* ## 概要
* Eosensの画像中から設定した色(HSV)で二値化を行い，その重心位置をdSpaceへ送信
*
* ## 必要なライブラリ
* - 環境に合わせてプロパティのディレクトリを設定してください
* - opencv: 3.0以上
*	- download: https://sourceforge.net/projects/opencvlibrary/files/opencv-win/
* - TBB
*	- download: https://github.com/01org/tbb/releases
* - boost
*	- download: https://sourceforge.net/projects/boost/files/boost-binaries/
* - SiliconSoftware SDK
*
* ## 操作
* - 画像が表示されているウインドウでキーを押すことで下記の様に動作
*	- q: プログラムの終了
*   - s: 画像を保存開始(ImageProcの"maxSaveNum"で保存フレーム数設定，事前に保存先ディレクトリを作成)
*   - S: 画像保存を停止
*
* ## 主要なファイル
* - eosens_sample.cpp: 本ファイル　メインファイル
* - communicate_dspce.h: dSpaceとの通信関係
* - display_info.h: 画像等の表示
* - image_proc.h: 画像処理
* - type_definition.h: 型や構造体の定義
*/

#include "stdafx.h"

#define CVLIBVER CVAUX_STR(CV_MAJOR_VERSION) CVAUX_STR(CV_MINOR_VERSION) CVAUX_STR(CV_SUBMINOR_VERSION)
#ifdef NDEBUG
#define CVLIB(name) "opencv_" CVAUX_STR(name) CVLIBVER
#else
#define CVLIB(name) "opencv_" CVAUX_STR(name) CVLIBVER "d"
#endif
#pragma comment(lib, CVLIB(world))


int main()
{
	//定数類
	//dSpaceとの通信
	const std::string destIP = "10.1.196.179";
	const std::string destPort = "50006";
	const std::string srcPort = "52001";
	//画像処理
	const unsigned int width = 608;
	const unsigned int height = 538;

	//スレッド間でやり取りするデータ（データやり取りの向きは一方向限定）
	spsc_queue<sendData_t> queSend; //! dSpaceへの送信データ 画像処理->通信
	spsc_queue<recvData_t> queRecv; //! dSpaceからの受信データ 通信->画像処理
	spsc_queue<dispData_t> queDisp; //! 状態表示のデータ 画像処理->表示
	std::atomic<bool> isSaveImage = false; //! デバッグ用に画像を出力　プログラム終了時に書出し　表示->画像処理
	
	auto comDspacePtr = std::make_unique<ComDspace>(queSend, queRecv, destIP, destPort, srcPort);
	auto imageProcPtr = std::make_unique<ImageProc>(queDisp, queSend, queRecv, width, height);
	auto dispInfoPtr = std::make_unique<DispInfo>(queDisp, width, height);
	
	std::thread sendThread(&ComDspace::sendData, std::ref(*comDspacePtr));
	std::thread imageProcThread(std::ref(*imageProcPtr), std::ref(isSaveImage));
	std::thread dispThread(std::ref(*dispInfoPtr), std::ref(isSaveImage));

	std::this_thread::sleep_for(std::chrono::milliseconds(10));
	std::cout << "finish loading!!" << std::endl;

	// mainスレッドがjoin()待ちで占有してしまうともったいないので100ミリ秒ごとに状態を確認する
	while (true) {
		if (dispThread.joinable()) {
			break;
		}
		std::this_thread::sleep_for(std::chrono::milliseconds(100));
	}

	dispThread.join();
	//表示スレッドを停止したら他のスレッドも停止する
	comDspacePtr->finishProc();
	imageProcPtr->finishProc();

	sendThread.join();
	imageProcThread.join();
    return 0;
}

