/**
* @brief　TBBとSSEを用いた重心位置計算
*/
#ifndef _CALCCOG_H_
#define _CALCCOG_H_

#include "mypvcore.h"
#include <opencv2/opencv.hpp>

/**
* @class CALC_COG_TBB
* @brief TBBを用いた重心位置計算クラス
*/
class CALC_COG_TBB{

	const unsigned char *src;
	unsigned int width;
	unsigned int height;
	unsigned int threads;
	unsigned int xstart;
	unsigned int xstop;
	unsigned int ystart;
	unsigned int ystop;
	
private:
	void _calc_cog(unsigned int _start, unsigned int _stop);

public:
	int mom0;
	int mom10;
	int mom01;

	void operator()(const tbb::blocked_range<size_t>& r){

		// Let's make the last thread do the least work
		unsigned int blockSize = (height + threads - 1) / threads;
//		if (blockSize % 2 == 1) blockSize += 1;
		unsigned int start = (unsigned int)r.begin()*blockSize;
		if (start >= height) return;
		unsigned int stop = (unsigned int)r.end()*blockSize;
		if (stop > height) {
			stop = height;
		}
		//	次の条件を満たすスレッドは処理をしなくてもよい
		if (stop <= ystart){ return; }
		if (ystop <= start){ return; }
		//	次の条件を満たすスレッドは、処理の範囲を狭める必要がある。
		if (start < ystart){ start = ystart; }
		if (ystop < ystop){ stop = ystop; }

		_calc_cog(start, stop);
	}

	CALC_COG_TBB(const unsigned char* _src, unsigned int _width, unsigned int _height, unsigned int _threads, cv::Rect& roi) :
		src(_src), width(_width), height(_height), threads(_threads), xstart(roi.tl().x), xstop(roi.br().x), ystart(roi.tl().y), ystop(roi.br().y), mom0(0), mom10(0), mom01(0){
	}

	//分割コンストラクタ、分割先でも初期化
	CALC_COG_TBB(CALC_COG_TBB &SplitTbb, tbb::split) : src(SplitTbb.src), width(SplitTbb.width), height(SplitTbb.height), threads(SplitTbb.threads), xstart(SplitTbb.xstart), xstop(SplitTbb.xstop), ystart(SplitTbb.ystart), ystop(SplitTbb.ystop), mom0(0), mom10(0), mom01(0){}

	//join関数
	void join(const CALC_COG_TBB &SplitTbb)
	{
		mom0 += SplitTbb.mom0;
		mom10 += SplitTbb.mom10;
		mom01 += SplitTbb.mom01; //結合
	}
};
#endif //_CALCCOG_H_