/**
* @brief�@dSpace�Ƃ̒ʐM(����M)
*/

#pragma once

/**
* @struct comDspace
* @brief�@dSpace�Ƃ̒ʐM�N���X
*/
struct ComDspace{

private:
	using Udp = boost::asio::ip::udp;
	spsc_queue<sendData_t> &queSend_; //! ���M�f�[�^�̃��b�N�t���[�L���[
	spsc_queue<recvData_t> &queRecv_; //! ��M�f�[�^�̃��b�N�t���[�L���[
	const std::string destIP_; //! ���M��(dSpace)IP�A�h���X
	const std::string destPort_; //! ���M��̃|�[�g
	const std::string srcPort_; //! ���M���̃|�[�g�i��M�p�j

	bool isFinished_; //! �������I������t���O

public:
	/**
	* @brief �R���X�g���N�^
	* @param[in] queSend ���M�f�[�^
	* @param[out] queRecv ��M�f�[�^
	* @param[in] destIP ���M��(dSpace)IP�A�h���X
	* @param[in] destPort ���M��(dSpace)�|�[�g
	* @param[in] srcPort ��M(PC)�|�[�g
	*/
	ComDspace(spsc_queue<sendData_t> &queSend, spsc_queue<recvData_t> &queRecv, const std::string &destIP, const std::string &destPort, const std::string &srcPort)
		: queSend_(queSend), queRecv_(queRecv)
		, destIP_(destIP), destPort_(destPort), srcPort_(srcPort)
		, isFinished_(false)
	{
	}

	/**
	* @brief dspace�փf�[�^���M
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
		//��M���Ƃ߂邽�߁C�����ɑ���
		boost::asio::ip::udp::endpoint destination(
			boost::asio::ip::address::from_string("127.0.0.1"), std::stoi(srcPort_));
		socket.send_to(boost::asio::buffer(objInfo.data(), objInfo.size()), destination);
	}

	/**
	* @brief dspace����f�[�^��M
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
	* @brief�@�������I������
	*/
	void finishProc() {
		isFinished_ = true;
	}

	/**
	* @brief�@�I���t���O�̊m�F
	*/
	bool isFinishProc() {
		return isFinished_;
	}
};