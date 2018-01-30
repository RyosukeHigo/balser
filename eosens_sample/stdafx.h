// stdafx.h : �W���̃V�X�e�� �C���N���[�h �t�@�C���̃C���N���[�h �t�@�C���A�܂���
// �Q�Ɖ񐔂������A�����܂�ύX����Ȃ��A�v���W�F�N�g��p�̃C���N���[�h �t�@�C��
// ���L�q���܂��B
//

#pragma once

#include "targetver.h"

#include <iostream>
#include <string>
#include <memory>
#include <functional>
#include <vector>
#include <array>
#include <chrono>
#include <thread>
#include <atomic>
#include <tchar.h>

#include <omp.h>

#include <opencv2/opencv.hpp>
#include <boost/lockfree/spsc_queue.hpp>
#include <boost/asio.hpp>

#include "mypvcore.h"

//#include "EoSens.h"
#include "calcCOG.h"

#include "type_definition.h"

#include "image_proc.h"
#include "display_info.h"
#include "comunicate_dspace.h"



// TODO: �v���O�����ɕK�v�Ȓǉ��w�b�_�[�������ŎQ�Ƃ��Ă�������
