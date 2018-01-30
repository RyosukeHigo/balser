/**
* @brief ��N���X����쐬�̉摜�������C�u����(pvcore)���g�p���邽�߂̐ݒ�֌W
*/

#ifndef _MYPVCORE_H_
#define _MYPVCORE_H_

#ifndef USE_SSE
#define USE_SSE (1)
#endif //USE_SSE
#define USE_TBB (1)

//pvcore�{��
#include <pvcore\common.h>
#include <pvcore\colorconversion.h>
#include <pvcore\sse_helpers.h>

//TBB�֘A
#include <tbb/task_scheduler_init.h>
#include <tbb/blocked_range.h>
#include <tbb/parallel_for.h>
#include <tbb/parallel_reduce.h>

#endif //_MYPVCORE_H_