/**
* @brief　dSpaceとの通信(送受信)
*/

#pragma once

/**
* @struct comDspace
* @brief　dSpaceとの通信クラス
*/
struct ComDspace{

private:
	using Udp = boost::asio::ip::udp;
	spsc_queue<sendData_t> &queSend_; //! 送信データのロックフリーキュー
	spsc_queue<recvData_t> &queRecv_; //! 受信データのロックフリーキュー
	const std::string destIP_; //! 送信先(dSpace)IPアドレス
	const std::string destPort_; //! 送信先のポート
	const std::string srcPort_; //! 送信元のポート（受信用）

	bool isFinished_; //! 処理を終了するフラグ

public:
	/**
	* @brief コンストラクタ
	* @param[in] queSend 送信データ
	* @param[out] queRecv 受信データ
	* @param[in] destIP 送信先(dSpace)IPアドレス
	* @param[in] destPort 送信先(dSpace)ポート
	* @param[in] srcPort 受信(PC)ポート
	*/
	ComDspace(spsc_queue<sendData_t> &queSend, spsc_queue<recvData_t> &queRecv, const std::string &destIP, const std::string &destPort, const std::string &srcPort)
		: queSend_(queSend), queRecv_(queRecv)
		, destIP_(destIP), destPort_(destPort), srcPort_(srcPort)
		, isFinished_(false)
	{
	}

	/**
	* @brief dspaceへデータ送信
	*/
	void sendData() const
	{
		sendData_t objInfo;

		boost::asio::io_service io_service;
		Udp::resolver resolver(io_service);
		Udp::resolver::query query(Udp::v4(), destIP_, destPort_);
		Udp::endpoint receiver_endpoint = *resolver.resolve(query);
		Udp::socket socket(io_service);
		socket.open(Udp::v4());
		while (!isFinished_) {
			if (!queSend_.pop(objInfo)) {
				std::this_thread::yield();
				continue;
			}
			socket.send_to(boost::asio::buffer(objInfo), receiver_endpoint);
		}
		//受信をとめるため，自分に送る
		boost::asio::ip::udp::endpoint destination(
			boost::asio::ip::address::from_string("127.0.0.1"), std::stoi(srcPort_));
		socket.send_to(boost::asio::buffer(objInfo.data(), objInfo.size()), destination);
	}

	/**
	* @brief dspaceからデータ受信
	*/
	void recieveData()
	{

		boost::asio::io_service io_service;
		Udp::socket socket(io_service, Udp::endpoint(Udp::v4(), std::stoi(srcPort_)));
		Udp::endpoint remote_endpoint;
		boost::system::error_code error;
		recvData_t recv_buf;
		size_t len;

		while (!isFinished_) {
			len = socket.receive_from(boost::asio::buffer(recv_buf), remote_endpoint, 0, error);

			queRecv_.push(recv_buf);
		}
	}

	/**
	* @brief　処理を終了する
	*/
	void finishProc() {
		isFinished_ = true;
	}

	/**
	* @brief　終了フラグの確認
	*/
	bool isFinishProc() {
		return isFinished_;
	}
};