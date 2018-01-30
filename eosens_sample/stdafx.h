// stdafx.h : 標準のシステム インクルード ファイルのインクルード ファイル、または
// 参照回数が多く、かつあまり変更されない、プロジェクト専用のインクルード ファイル
// を記述します。
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



// TODO: プログラムに必要な追加ヘッダーをここで参照してください
