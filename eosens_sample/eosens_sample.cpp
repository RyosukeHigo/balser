/**
* @mainpage
* 
* ## �T�v
* Eosens�̉摜������ݒ肵���F(HSV)�œ�l�����s���C���̏d�S�ʒu��dSpace�֑��M
*
* ## �K�v�ȃ��C�u����
* - ���ɍ��킹�ăv���p�e�B�̃f�B���N�g����ݒ肵�Ă�������
* - opencv: 3.0�ȏ�
*	- download: https://sourceforge.net/projects/opencvlibrary/files/opencv-win/
* - TBB
*	- download: https://github.com/01org/tbb/releases
* - boost
*	- download: https://sourceforge.net/projects/boost/files/boost-binaries/
* - SiliconSoftware SDK
*
* ## ����
* - �摜���\������Ă���E�C���h�E�ŃL�[���������Ƃŉ��L�̗l�ɓ���
*	- q: �v���O�����̏I��
*   - s: �摜��ۑ��J�n(ImageProc��"maxSaveNum"�ŕۑ��t���[�����ݒ�C���O�ɕۑ���f�B���N�g�����쐬)
*   - S: �摜�ۑ����~
*
* ## ��v�ȃt�@�C��
* - eosens_sample.cpp: �{�t�@�C���@���C���t�@�C��
* - communicate_dspce.h: dSpace�Ƃ̒ʐM�֌W
* - display_info.h: �摜���̕\��
* - image_proc.h: �摜����
* - type_definition.h: �^��\���̂̒�`
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
	//�萔��
	//dSpace�Ƃ̒ʐM
	const std::string destIP = "10.1.196.179";
	const std::string destPort = "50006";
	const std::string srcPort = "52001";
	//�摜����
	const unsigned int width = 608;
	const unsigned int height = 538;

	//�X���b�h�Ԃł���肷��f�[�^�i�f�[�^�����̌����͈��������j
	spsc_queue<sendData_t> queSend; //! dSpace�ւ̑��M�f�[�^ �摜����->�ʐM
	spsc_queue<recvData_t> queRecv; //! dSpace����̎�M�f�[�^ �ʐM->�摜����
	spsc_queue<dispData_t> queDisp; //! ��ԕ\���̃f�[�^ �摜����->�\��
	std::atomic<bool> isSaveImage = false; //! �f�o�b�O�p�ɉ摜���o�́@�v���O�����I�����ɏ��o���@�\��->�摜����
	
	auto comDspacePtr = std::make_unique<ComDspace>(queSend, queRecv, destIP, destPort, srcPort);
	auto imageProcPtr = std::make_unique<ImageProc>(queDisp, queSend, queRecv, width, height);
	auto dispInfoPtr = std::make_unique<DispInfo>(queDisp, width, height);
	
	std::thread sendThread(&ComDspace::sendData, std::ref(*comDspacePtr));
	std::thread imageProcThread(std::ref(*imageProcPtr), std::ref(isSaveImage));
	std::thread dispThread(std::ref(*dispInfoPtr), std::ref(isSaveImage));

	std::this_thread::sleep_for(std::chrono::milliseconds(10));
	std::cout << "finish loading!!" << std::endl;

	// main�X���b�h��join()�҂��Ő�L���Ă��܂��Ƃ��������Ȃ��̂�100�~���b���Ƃɏ�Ԃ��m�F����
	while (true) {
		if (dispThread.joinable()) {
			break;
		}
		std::this_thread::sleep_for(std::chrono::milliseconds(100));
	}

	dispThread.join();
	//�\���X���b�h���~�����瑼�̃X���b�h����~����
	comDspacePtr->finishProc();
	imageProcPtr->finishProc();

	sendThread.join();
	imageProcThread.join();
    return 0;
}

