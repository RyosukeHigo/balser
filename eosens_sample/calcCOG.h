/**
* @brief�@TBB��SSE��p�����d�S�ʒu�v�Z
*/
#ifndef _CALCCOG_H_
#define _CALCCOG_H_

#include "mypvcore.h"
#include <opencv2/opencv.hpp>

/**
* @class CALC_COG_TBB
* @brief TBB��p�����d�S�ʒu�v�Z�N���X
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
		//	���̏����𖞂����X���b�h�͏��������Ȃ��Ă��悢
		if (stop <= ystart){ return; }
		if (ystop <= start){ return; }
		//	���̏����𖞂����X���b�h�́A�����͈̔͂����߂�K�v������B
		if (start < ystart){ start = ystart; }
		if (ystop < ystop){ stop = ystop; }

		_calc_cog(start, stop);
	}

	CALC_COG_TBB(const unsigned char* _src, unsigned int _width, unsigned int _height, unsigned int _threads, cv::Rect& roi) :
		src(_src), width(_width), height(_height), threads(_threads), xstart(roi.tl().x), xstop(roi.br().x), ystart(roi.tl().y), ystop(roi.br().y), mom0(0), mom10(0), mom01(0){
	}

	//�����R���X�g���N�^�A������ł�������
	CALC_COG_TBB(CALC_COG_TBB &SplitTbb, tbb::split) : src(SplitTbb.src), width(SplitTbb.width), height(SplitTbb.height), threads(SplitTbb.threads), xstart(SplitTbb.xstart), xstop(SplitTbb.xstop), ystart(SplitTbb.ystart), ystop(SplitTbb.ystop), mom0(0), mom10(0), mom01(0){}

	//join�֐�
	void join(const CALC_COG_TBB &SplitTbb)
	{
		mom0 += SplitTbb.mom0;
		mom10 += SplitTbb.mom10;
		mom01 += SplitTbb.mom01; //����
	}
};
#endif //_CALCCOG_H_