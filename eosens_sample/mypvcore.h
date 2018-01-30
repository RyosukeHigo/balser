/**
* @brief 二クラスさん作成の画像処理ライブラリ(pvcore)を使用するための設定関係
*/

#ifndef _MYPVCORE_H_
#define _MYPVCORE_H_

#ifndef USE_SSE
#define USE_SSE (1)
#endif //USE_SSE
#define USE_TBB (1)

//pvcore本体
#include <pvcore\common.h>
#include <pvcore\colorconversion.h>
#include <pvcore\sse_helpers.h>

//TBB関連
#include <tbb/task_scheduler_init.h>
#include <tbb/blocked_range.h>
#include <tbb/parallel_for.h>
#include <tbb/parallel_reduce.h>

#endif //_MYPVCORE_H_