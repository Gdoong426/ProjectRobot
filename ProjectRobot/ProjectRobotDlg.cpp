
// ProjectRobotDlg.cpp : ��@��
//

#include "stdafx.h"
#include "ProjectRobot.h"
#include "ProjectRobotDlg.h"
#include "afxdialogex.h"
#include "Resource.h"


#include <stdio.h>
#include <stdlib.h>
#include <WinGDI.h>
//#include "opencv2/features2d/features2d.hpp"
#include "opencv2/nonfree/features2d.hpp"
#include "opencv2/opencv_modules.hpp"
#include "core.hpp"
#include "opencv2/legacy/legacy.hpp"
#include "opencv2/nonfree/nonfree.hpp"
#include "opencv2/video/background_segm.hpp"
#include "opencv2/video/tracking.hpp"
#include "opencv2/contrib/contrib.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/video.hpp"
#include "opencv2/objdetect/objdetect.hpp"
#include "iostream"
#include "fstream"
#include "sstream"
#include "cmath"
//
//#include "opencv2/highgui/highgui.hpp"
//#include "opencv2/imgproc/imgproc.hpp"
//#include "opencv2/objdetect/objdetect.hpp"
//#include "opencv2/video/background_segm.hpp"
//#include "opencv2/features2d/features2d.hpp"
//#include "opencv2/opencv_modules.hpp"
//#include "opencv2/core/utility.hpp"
//#include "opencv2/video/tracking.hpp"
//#include "videoio.hpp"
//#include <cv.h>
//#include "core.hpp"
//#include "stdio.h"
//#include "stdlib.h"
//
//#include "iostream"
//#include "fstream"
//#include "sstream"

bool redRobotDir = true; // red moving direction
bool greenRobotDir = true; // green moving direction

float redDirAngle; // red robot face direction
float greenDirAngle; // green robot face direction


#ifdef _DEBUG
#define new DEBUG_NEW
#endif


// �� App About �ϥ� CAboutDlg ��ܤ��
using namespace std;
using namespace cv;

class CAboutDlg : public CDialogEx
{
public:
	CAboutDlg();

// ��ܤ�����
#ifdef AFX_DESIGN_TIME
	enum { IDD = IDD_ABOUTBOX };
#endif

	protected:
	virtual void DoDataExchange(CDataExchange* pDX);    // DDX/DDV �䴩

// �{���X��@
protected:
	DECLARE_MESSAGE_MAP()
};

CAboutDlg::CAboutDlg() : CDialogEx(IDD_ABOUTBOX)
{
}

void CAboutDlg::DoDataExchange(CDataExchange* pDX)
{
	CDialogEx::DoDataExchange(pDX);
}

BEGIN_MESSAGE_MAP(CAboutDlg, CDialogEx)
END_MESSAGE_MAP()


// CProjectRobotDlg ��ܤ��



CProjectRobotDlg::CProjectRobotDlg(CWnd* pParent /*=NULL*/)
	: CDialogEx(IDD_PROJECTROBOT_DIALOG, pParent)
{
	m_hIcon = AfxGetApp()->LoadIcon(IDR_MAINFRAME);
}

void CProjectRobotDlg::DoDataExchange(CDataExchange* pDX)
{
	CDialogEx::DoDataExchange(pDX);
}

BEGIN_MESSAGE_MAP(CProjectRobotDlg, CDialogEx)
	ON_WM_SYSCOMMAND()
	ON_WM_PAINT()
	ON_WM_QUERYDRAGICON()
	ON_BN_CLICKED(IDC_BUTTON1, &CProjectRobotDlg::OnBnClickedButton1)
	ON_BN_CLICKED(IDC_BUTTON4, &CProjectRobotDlg::OnBnClickedButton4)
	ON_BN_CLICKED(IDC_BUTTON5, &CProjectRobotDlg::OnBnClickedButton5)
	ON_BN_CLICKED(IDC_BUTTON3, &CProjectRobotDlg::OnBnClickedButton3)
	ON_EN_CHANGE(IDC_EDIT1, &CProjectRobotDlg::OnEnChangeEdit1)
	ON_EN_CHANGE(IDC_EDIT2, &CProjectRobotDlg::OnEnChangeEdit2)
	ON_EN_CHANGE(IDC_EDIT3, &CProjectRobotDlg::OnEnChangeEdit3)
	ON_EN_CHANGE(IDC_EDIT4, &CProjectRobotDlg::OnEnChangeEdit4)
	ON_BN_CLICKED(IDC_BUTTON7, &CProjectRobotDlg::OnBnClickedButton7)
	ON_BN_CLICKED(IDC_BUTTON8, &CProjectRobotDlg::OnBnClickedButton8)
END_MESSAGE_MAP()


// CProjectRobotDlg �T���B�z�`��

BOOL CProjectRobotDlg::OnInitDialog()
{
	CDialogEx::OnInitDialog();

	// �N [����...] �\����[�J�t�Υ\����C

	// IDM_ABOUTBOX �����b�t�ΩR�O�d�򤧤��C
	ASSERT((IDM_ABOUTBOX & 0xFFF0) == IDM_ABOUTBOX);
	ASSERT(IDM_ABOUTBOX < 0xF000);

	CMenu* pSysMenu = GetSystemMenu(FALSE);
	if (pSysMenu != NULL)
	{
		BOOL bNameValid;
		CString strAboutMenu;
		bNameValid = strAboutMenu.LoadString(IDS_ABOUTBOX);
		ASSERT(bNameValid);
		if (!strAboutMenu.IsEmpty())
		{
			pSysMenu->AppendMenu(MF_SEPARATOR);
			pSysMenu->AppendMenu(MF_STRING, IDM_ABOUTBOX, strAboutMenu);
		}
	}

	// �]�w����ܤ�����ϥܡC�����ε{�����D�������O��ܤ���ɡA
	// �ج[�|�۰ʱq�Ʀ��@�~
	SetIcon(m_hIcon, TRUE);			// �]�w�j�ϥ�
	SetIcon(m_hIcon, FALSE);		// �]�w�p�ϥ�
	AllocConsole();
	freopen("CONOUT$", "w", stdout);


	// TODO: �b���[�J�B�~����l�]�w

	return TRUE;  // �Ǧ^ TRUE�A���D�z�ﱱ��]�w�J�I
}

void CProjectRobotDlg::OnSysCommand(UINT nID, LPARAM lParam)
{
	if ((nID & 0xFFF0) == IDM_ABOUTBOX)
	{
		CAboutDlg dlgAbout;
		dlgAbout.DoModal();
	}
	else
	{
		CDialogEx::OnSysCommand(nID, lParam);
	}
}

// �p�G�N�̤p�ƫ��s�[�J�z����ܤ���A�z�ݭn�U�C���{���X�A
// �H�Kø�s�ϥܡC���ϥΤ��/�˵��Ҧ��� MFC ���ε{���A
// �ج[�|�۰ʧ������@�~�C

void CProjectRobotDlg::OnPaint()
{
	if (IsIconic())
	{
		CPaintDC dc(this); // ø�s���˸m���e

		SendMessage(WM_ICONERASEBKGND, reinterpret_cast<WPARAM>(dc.GetSafeHdc()), 0);

		// �N�ϥܸm����Τ�ݯx��
		int cxIcon = GetSystemMetrics(SM_CXICON);
		int cyIcon = GetSystemMetrics(SM_CYICON);
		CRect rect;
		GetClientRect(&rect);
		int x = (rect.Width() - cxIcon + 1) / 2;
		int y = (rect.Height() - cyIcon + 1) / 2;

		// �yø�ϥ�
		dc.DrawIcon(x, y, m_hIcon);
	}
	else
	{
		CDialogEx::OnPaint();
	}
}

// ���ϥΪ̩즲�̤p�Ƶ����ɡA
// �t�ΩI�s�o�ӥ\����o�����ܡC
HCURSOR CProjectRobotDlg::OnQueryDragIcon()
{
	return static_cast<HCURSOR>(m_hIcon);
}



void CProjectRobotDlg::OnBnClickedButton1()
{
	Mat frame;
	Mat fgMaskMOG;
	Mat fgMaskMOG2;
	Mat fgMaskGMG;
	Mat contourImg;

	//Ptr <BackgroundSubtractor> pMOG;
	//Ptr <BackgroundSubtractor> pMOG2;
	//Ptr <BackgroundSubtractor> pGMG;

	//pMOG = new BackgroundSubtractorMOG();
	//pMOG2 = new BackgroundSubtractorMOG2();
	//pGMG = new BackgroundSubtractorGMG();

	BackgroundSubtractorMOG pMOG;
	BackgroundSubtractorMOG2 pMOG2;
	BackgroundSubtractorGMG pGMG;

	Mat element = getStructuringElement(MORPH_RECT, Size(5, 5), Point(1, 1));
	
	VideoCapture cap(-1);
	//VideoCapture cap("VideoTest.avi");
	while (true)
	{
		Mat cameraFrame;
		if (!cap.read(frame))
		break;
		/*pMOG->operator()(frame, fgMaskMOG);
		pMOG2->operator()(frame, fgMaskMOG2);
		pGMG->operator()(frame, fgMaskGMG);*/

		pMOG(frame, fgMaskMOG);
		pMOG2(frame, fgMaskMOG2);
		pGMG(frame, fgMaskGMG);

		//morphologyEx(fgMaskGMG, fgMaskGMG, CV_MOP_OPEN, element);
		
		threshold(fgMaskMOG, contourImg, 128, 255, CV_THRESH_BINARY);

		vector<vector<Point>> contours;
		findContours(contourImg, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);

		vector<Rect> output;
		vector<vector<Point>>::iterator itc = contours.begin();

		while (itc != contours.end()) {
			Rect mr = boundingRect(Mat(*itc));
			rectangle(frame, mr, Scalar(255, 0, 0), 2);
			++itc;
		}

		imshow("original", frame);
		imshow("MOG", fgMaskMOG);
		imshow("MOG2", fgMaskMOG2);
		imshow("GMG", fgMaskGMG);

		if (waitKey(33) == 27) {
			destroyWindow("original");
			destroyWindow("MOG");
			destroyWindow("MOG2");
			destroyWindow("GMG");
			break;
		}
	}

}



void CProjectRobotDlg::OnBnClickedButton4()
{
	VideoCapture capture(-1);
	if (!capture.isOpened()) {
		printf("Error finding camera...");
	}
	Size videoSize = Size((int)capture.get(CV_CAP_PROP_FRAME_WIDTH), (int)capture.get(CV_CAP_PROP_FRAME_HEIGHT));
	VideoWriter writer;
	writer.open("VideoTest.avi", CV_FOURCC('M', 'J', 'P', 'G'), 30, videoSize);
	namedWindow("show image", 0);

	while (true) {
		Mat frame;
		capture >> frame;
		if (!frame.empty()) {
			writer.write(frame);
			imshow("show image", frame);
			if (waitKey(33) == 27) {
				break;
			}
		}
	}
}

// Lower red hue range
int iLowH = 0;		int iHighH = 10;	
int iLowS = 100;	int iHighS = 255;	
int iLowV = 100;	int iHighV = 255;	

// Higher red hue range
int uLowH = 245;	int uHighH = 255;	
int uLowS = 100;	int uHighS = 255;
int uLowV = 100;	int uHighV = 255;

// green hue range
int gLowH = 40;		int gHighH = 100;	
int gLowS = 90;	int gHighS = 255;
int gLowV = 90;	int gHighV = 255;

Point lastPosition_red;
Point postPosition_red;
Point lastPosition_green;
Point postPosition_green;

void imgprocessing(Mat &imgThreshold) {
	// Blurring the image to wipe the noise
	medianBlur(imgThreshold, imgThreshold, 7);
	GaussianBlur(imgThreshold, imgThreshold, Size(5, 5), 0, 0, 4);
	erode(imgThreshold, imgThreshold, Mat());
	dilate(imgThreshold, imgThreshold, Mat());

}

float findRobotDirection(bool dir, Mat frame, Point postPosition, Point lastPosition) {
	float x1 = 1.00, y1 = 0.00;
	float x2 = (float)postPosition.x - (float)lastPosition.x;
	float y2 = (float)postPosition.y - (float)lastPosition.y;
	float dot = x1*x2 + y1*y2;
	float det = x1*y2 - y1*x2;
	float dirAngle = atan2(det,dot);


	int length = 20;
	Point point0, point1, point2;
	if (dir == true) { // if foward
		point0.x = (int)round(postPosition.x + 30 * cos(dirAngle + CV_PI));
		point0.y = (int)round(postPosition.y + 30 * sin(dirAngle + CV_PI));
		point1.x = (int)round(postPosition.x + length*cos(dirAngle + 160*CV_PI / 180.0));
		point1.y = (int)round(postPosition.y + length*sin(dirAngle + 160*CV_PI / 180.0));
		point2.x = (int)round(postPosition.x + length*cos(dirAngle - 160*CV_PI / 180.0));
		point2.y = (int)round(postPosition.y + length*sin(dirAngle - 160*CV_PI / 180.0));
		// drawing 
		//line(frame, point1, postPosition, Scalar(255, 255, 255), 1);
		//line(frame, point2, postPosition, Scalar(255, 255, 255), 1);
		//line(frame, point0, postPosition, Scalar(255, 255, 255), 1);
		
	}
	if (dir != true) { //if backward
		dirAngle = dirAngle + CV_PI;
		point0.x = (int)round(lastPosition.x + 30 * cos(dirAngle + CV_PI));
		point0.y = (int)round(lastPosition.y + 30 * sin(dirAngle + CV_PI));
		point1.x = (int)round(lastPosition.x + length*cos(dirAngle + 160 * CV_PI / 180.0));
		point1.y = (int)round(lastPosition.y + length*sin(dirAngle + 160 * CV_PI / 180.0));
		point2.x = (int)round(lastPosition.x + length*cos(dirAngle - 160 * CV_PI / 180.0));
		point2.y = (int)round(lastPosition.y + length*sin(dirAngle - 160 * CV_PI / 180.0));
		//line(frame, point1, lastPosition, Scalar(255, 255, 255), 1);
		//line(frame, point2, lastPosition, Scalar(255, 255, 255), 1);
		//line(frame, point0, lastPosition, Scalar(255, 255, 255), 1);
	}
	return dirAngle;
}



void findContoursandMomentum( Mat &ThreshImg, Mat &img, Point &lastPosition, Point &postPosition, bool &RobotDir, float &DirAngle, Rect &colorbound) { // for find the contours and minimum enclosing circle of the picture
	int largest_area = 0;
	int largest_area_index = 0;
	Mat canny, oriThresh;
	vector<vector<Point>> contours;
	vector<Vec4i> hierarchy;

	oriThresh = ThreshImg.clone();
	Canny(oriThresh, canny, 50, 255, 3);
	findContours(oriThresh, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_NONE);
	
	vector<vector<Point>> contours_poly(contours.size());
	vector<Point2f> center(contours.size());
	vector<Rect> boundRect(contours.size());
	

	for (int i = 0; i < contours.size(); i++) {
		approxPolyDP(Mat(contours[i]), contours_poly[i], 3, true);
		double a = contourArea(contours[i]);
		boundRect[i] = boundingRect(Mat(contours_poly[i]));

		
		if (a > largest_area) {
			largest_area = a;
			largest_area_index = i;
		}
	}
	
	//drawContours(img, contours, largest_area_index, Scalar(255, 255, 255), 2, 8, hierarchy);
	if (boundRect.size() != 0) {
		Rect biggerRect = boundRect[largest_area_index];
		int newX = (boundRect[largest_area_index].tl().x - boundRect[largest_area_index].br().x) * 3 / 10;
		int newY = (boundRect[largest_area_index].tl().y - boundRect[largest_area_index].br().y) * 3 / 10;

		Point inflationPoint(newX, newY);
		Size inflationSize(-newX * 2, -newY * 2);
		biggerRect += inflationPoint;
		biggerRect += inflationSize;
		Point center((boundRect[largest_area_index].tl().x + boundRect[largest_area_index].br().x) / 2, (boundRect[largest_area_index].tl().y + boundRect[largest_area_index].br().y) / 2);
		int radius = norm(boundRect[largest_area_index].tl() - boundRect[largest_area_index].br());
		//rectangle(img, boundRect[largest_area_index].tl(), boundRect[largest_area_index].br(), Scalar(255, 255, 255), 2);
		//rectangle(img, biggerRect.tl(), biggerRect.br(), Scalar(255, 0, 0), 2);
		//circle(img, center, radius, Scalar(0, 255, 0), 2);
		
	}

	// find the momentom of the img;
	if (contours.size() != 0) {
		Moments oMoments = moments(contours[largest_area_index]);
		double dM01 = oMoments.m01;
		double dM10 = oMoments.m10;
		double dArea = oMoments.m00;
		if (dArea > 1000) {
			postPosition.x = dM10 / dArea;
			postPosition.y = dM01 / dArea;

			if (lastPosition.x >= 0 && lastPosition.y >= 0 && postPosition.x >= 0 && postPosition.y >= 0) {
				//line(img, postPosition, lastPosition, Scalar(0, 0, 255), 2);
			}
			DirAngle = findRobotDirection(RobotDir, img, postPosition, lastPosition);
			lastPosition.x = postPosition.x;
			lastPosition.y = postPosition.y;
		}
	}
	
}

void drawOptFlowMap(const Mat& flow, Mat& cflowmap, int step, const Scalar& color) {
	for (int y = 0; y < cflowmap.rows; y += step)

		for (int x = 0; x < cflowmap.cols; x += step)

		{
			const Point2f& fxy = flow.at< Point2f>(y, x);

			line(cflowmap, Point(x, y), Point(cvRound(x + fxy.x), cvRound(y + fxy.y)),
				color);
			circle(cflowmap, Point(cvRound(x + fxy.x), cvRound(y + fxy.y)), 1, color, -1);

		}

}

void findOpticalFlow( Mat &frame,Mat &prevGray, Mat &nowGray) {
	//find optical flow;
	float s = 1.5;
	Mat temp1, temp2, temp3;
	resize(prevGray, temp1, Size(frame.size().width / s, frame.size().height / s));
	resize(nowGray, temp2, Size(frame.size().width / s, frame.size().height / s));
	resize(frame, temp3, Size(frame.size().width / s, frame.size().height / s));
	Mat flow;
	

	//imshow("flow", temp1);
	calcOpticalFlowFarneback(prevGray, nowGray, flow, 0.5, 4, 15, 3, 7, 1.2, 0);
	drawOptFlowMap(flow, temp3, 10, CV_RGB(0, 255, 0));
	imshow("flow", temp3);

	///  The following is for background subtraction.
	
}

void backgroundsubtract( Mat frame, Mat prevImg, Mat &diff) {
	Mat img = frame.clone();
	Mat prev = prevImg.clone();
	absdiff(img, prev, diff);
	threshold(diff, diff, 10, 255, THRESH_BINARY);
	medianBlur(diff, diff, 5);

	
	
}


void rgb2hsvimg(Mat ori, Mat &red, Mat &green) {

	Mat imgHSV;
	cvtColor(ori, imgHSV, CV_BGR2HSV);
	Mat imgThreshold_u, imgThreshold_l;
	inRange(imgHSV, Scalar(iLowH, iLowS, iLowV), Scalar(iHighH, iHighS, iHighV), imgThreshold_l);
	inRange(imgHSV, Scalar(uLowH, uLowS, uLowV), Scalar(uHighH, uHighS, uHighV), imgThreshold_u);
	inRange(imgHSV, Scalar(gLowH, gLowS, gLowV), Scalar(gHighH, gHighS, gHighV), green);

	addWeighted(imgThreshold_l, 1.0, imgThreshold_u, 1.0, 0, red);

}





void CProjectRobotDlg::OnBnClickedButton5()
{
	// Color Tracking
	VideoCapture cap("VideoTest.avi");
	//VideoCapture cap(-1);
	Rect redbound, greenbound;
	bool num = true;


	if (!cap.isOpened()) {
		cout << "Can't open the web cam...";
		exit(-1);
	}
	
	Mat frame ,img, prev;
	Mat gray, prevGray;
	Mat diff_r, diff_g, diff; // robot movement image
	Mat imgThreshold_g; // create Mat for red and green HUE image
	Mat imgThreshold_r; // the combined red image
	Mat prevGreen, prevRed;


	BackgroundSubtractorGMG pGMG;

	vector<Point2f> cornersRed, cornersRed_prev, cornersRed_temp;
	vector<Point2f> cornersGreen, cornersGreen_prev, cornersGreen_temp;
	cap >> prev;
	cvtColor(prev, prevGray, CV_RGB2GRAY);
	rgb2hsvimg(prev, prevRed, prevGreen);
	while (true) {
		cap >> img;
		frame = img.clone();

		if (frame.empty()) {
			cout << "Error reading frame from web cam..." << endl;
			destroyAllWindows;
			waitKey(0);
			break;
		}
		rgb2hsvimg(frame, imgThreshold_r, imgThreshold_g);

		// denoise and process the image;
		imgprocessing(imgThreshold_r);
		imgprocessing(imgThreshold_g);

		//find the biggest contours in the image and calculate the moments of the threshold image;
		// and find out robots current face direction
		findContoursandMomentum(imgThreshold_g, frame, lastPosition_green, postPosition_green, greenRobotDir,greenDirAngle, greenbound);
		findContoursandMomentum(imgThreshold_r, frame, lastPosition_red, postPosition_red, redRobotDir, redDirAngle, redbound);

		// optical flow
		cvtColor(frame, gray, CV_RGB2GRAY);
		/*cvtColor(prev, prevGray, CV_RGB2GRAY);
		findOpticalFlow(frame, prevGray, gray);*/

		// background subtraction

		backgroundsubtract(gray, prevGray, diff);
		backgroundsubtract(imgThreshold_g, prevGreen, diff_g);
		backgroundsubtract(imgThreshold_r, prevRed, diff_r);

		vector<vector<Point>> contours;
		vector<Vec4i> hierarchy;
		findContours(diff, contours, hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);

		vector<vector<Point>> hull(contours.size());
		for (int i = 0; i < contours.size(); i++) {
			convexHull(Mat(contours[i]), hull[i], false);
		}


		vector<Rect> output;
		vector<vector<Point>>::iterator itc = contours.begin();
		
		
		while (itc != contours.end()) {
			Rect mr = boundingRect(Mat(*itc));
			rectangle(frame, mr, Scalar(255, 0, 0), 2);
			++itc;
		}

		printf("red robot face angle: %d  ", (int)(redDirAngle * 180 / CV_PI));
		printf("green robot face angle: %d  \n", (int)(greenDirAngle * 180 / CV_PI));
		
		// Show x and y coordinates of every robots.
		CString red_x, red_y,green_x, green_y;
		red_x.Format(_T("%d"), postPosition_red.x);
		red_y.Format(_T("%d"), postPosition_red.y);
		green_x.Format(_T("%d"), postPosition_green.x);
		green_y.Format(_T("%d"), postPosition_green.y);

		GetDlgItem(IDC_EDIT1)->SetWindowTextW(red_x);
		GetDlgItem(IDC_EDIT2)->SetWindowTextW(red_y);
		GetDlgItem(IDC_EDIT3)->SetWindowTextW(green_x);
		GetDlgItem(IDC_EDIT4)->SetWindowTextW(green_y);

		// Show images

		//imshow("Smooth", Smooth);
		imshow("Red Threshold", imgThreshold_r);
		imshow("Green Threshold", imgThreshold_g);
		imshow("Original", frame);
		imshow("GreenM", diff_g);
		imshow("RedM", diff_r);
		imshow("move", diff);

		prevGray = gray.clone();
		prev = img.clone();
		prevGreen = imgThreshold_g.clone();
		prevRed = imgThreshold_r.clone();

		if (waitKey(33) == 27) {
			destroyWindow("Red Threshold");
			destroyWindow("Green Threshold");
			destroyWindow("Original");
			destroyWindow("Movement");
			destroyWindow("GreenM");
			destroyWindow("RedM");
			destroyWindow("move");

			break;
		}
	}

}



void CProjectRobotDlg::OnBnClickedButton3()
{
	vector<Rect> robot;
	Mat frame, gray_frame;
	String Robot_cascade_name = "cascade_1.xml";
	//VideoCapture capture("VIDEO0032.mp4");
	//VideoCapture capture("VIDEO0063.mp4");
	VideoCapture capture("VideoTest.avi");
	capture >> frame;

	CascadeClassifier robot_cascade;
	while (1) {
		capture >> frame;
		if (!capture.isOpened()) {
			break;
		}
		cvtColor(frame, gray_frame, CV_RGB2GRAY);
		equalizeHist(gray_frame, gray_frame);
		if (!robot_cascade.load(Robot_cascade_name)) {
			printf("--(!)Error loading robot cscade\n");
			break;
		}

		robot_cascade.detectMultiScale(gray_frame, robot, 1.1, 20, 0 | CV_HAAR_SCALE_IMAGE, Size(70, 70));
		for (size_t i = 0; i < robot.size(); i++) {
			Mat faceROI = gray_frame(robot[i]);
			int x = robot[i].x;
			int y = robot[i].y;
			int h = y + robot[i].height;
			int w = x + robot[i].width;
			rectangle(
				frame, Point(x, y), Point(w, h), Scalar(255, 0, 0), 2, 8, 0
			);
		}
		namedWindow("Robot", 2);
		imshow("Robot", frame);
		int c = waitKey(2);
		if (c == 27) {
			destroyWindow("Robot");
			break;
		}
	}

}

vector<Mat> RobotImg;

void ImgPreprocessing(Mat imgHSV) {
	
	Mat img = imgHSV.clone();
	// Use HSV color space to seperate different color
	RobotImg.erase(RobotImg.begin());
	Mat imgThreshold_ru, imgThreshold_rl, imgThreshold_r;
	inRange(img, Scalar(iLowH, iLowS, iLowV), Scalar(iHighH, iHighS, iHighV), imgThreshold_rl);
	inRange(img, Scalar(uLowH, uLowS, uLowV), Scalar(uHighH, uHighS, uHighV), imgThreshold_ru);
	addWeighted(imgThreshold_rl, 1.0, imgThreshold_ru, 1.0, 0, RobotImg[0]);
	imshow("red", RobotImg[0]);

	RobotImg.erase(RobotImg.begin() + 1);
	Mat imgThreshold_g;
	inRange(img, Scalar(gLowH, gLowS, gLowV), Scalar(gHighH, gHighS, gHighV), imgThreshold_g); // Green
	imshow("green", imgThreshold_g);
	RobotImg.push_back(imgThreshold_g);


	//printf("RobotImg Size: %d \n", RobotImg.size());
	
	//imshow("greed", imgThreshold_g);
	// denoise the image
	/*for (int num = 0; num < RobotImg.size(); num++) {
		medianBlur(RobotImg[num], RobotImg[num], 7);
		GaussianBlur(RobotImg[num], RobotImg[num], Size(5, 5), 0, 0, 4);
	}*/
	
}

void CProjectRobotDlg::OnBnClickedButton7()
{
	// color tracking version 2
	VideoCapture cap("VideoTest.avi");
	if (!cap.isOpened()) {
		cout << "Can't open the web cam...";
		exit(-1);
	}

	Mat frame;
	cap >> frame;
	while (true) {
		cap >> frame;

		if (frame.empty()) {
			cout << "Error reading frame from web cam..." << endl;
		}

		Mat imgHSV;
		cvtColor(frame, imgHSV, CV_BGR2HSV);
		imshow("HSV", imgHSV);
		 // Store the seperate kind of colors of robot image
		RobotImg.reserve(2);
		ImgPreprocessing(imgHSV);

		// try to find the color circle
		//vector<Vec3f> circles_r, circles_g;
		//HoughCircles(RobotImg[0], circles_r, CV_HOUGH_GRADIENT, 1, 20,
		//	50, 30, 0, 0); // Change the last two parameters
		//				   // (min_radius & max_radius) to detect larger circles
		//HoughCircles(RobotImg[1], circles_g, CV_HOUGH_GRADIENT, 1, 20,
		//	50, 30, 0, 0);

		//// Draw red circles
		//for (size_t i = 0; i < circles_r.size(); i++) {

		//	Point center_r(cvRound(circles_r[i][0]), cvRound(circles_r[i][1]));
		//	int radius_r = cvRound(circles_r[i][2]);

		//	circle(frame, center_r, 3, Scalar(0, 0, 255), -1, 8, 0);
		//	circle(frame, center_r, radius_r, Scalar(0, 0, 255), 2, 8, 0);
		//}
		//// Draw green circles
		//for (size_t i = 0; i < circles_g.size(); i++) {

		//	Point center_g(cvRound(circles_g[i][0]), cvRound(circles_g[i][1]));
		//	int radius_g = cvRound(circles_g[i][2]);

		//	circle(frame, center_g, 3, Scalar(0, 255, 0), -1, 8, 0);
		//	circle(frame, center_g, radius_g, Scalar(0, 255, 0), 3, 8, 0);
		//}

		//// Calculate the moments of the threshold image;
		//Moments oMoments_r = moments(RobotImg[0]);
		//Moments oMoments_g = moments(RobotImg[1]);

		//double dM01_r = oMoments_r.m01;		double dM10_r = oMoments_r.m10;		double dArea_r = oMoments_r.m00;
		//double dM01_g = oMoments_g.m01;		double dM10_g = oMoments_g.m10;		double dArea_g = oMoments_g.m00;

		//if (dArea_r > 1000) {
		//	int posX_r = dM10_r / dArea_r;
		//	int posY_r = dM01_r / dArea_r;

		//	if (iLastX_r >= 0 && iLastY_r >= 0 && posX_r >= 0 && posY_r >= 0) {
		//		line(frame, Point(posX_r, posY_r), Point(iLastX_r, iLastY_r), Scalar(0, 0, 255), 2);
		//	}

		//	iLastX_r = posX_r;
		//	iLastY_r = posY_r;
		//}
		//if (dArea_g > 1000) {
		//	int posX_g = dM10_g / dArea_g;
		//	int posY_g = dM01_g / dArea_g;

		//	if (iLastX_g >= 0 && iLastY_g >= 0 && posX_g >= 0 && posY_g >= 0) {
		//		line(frame, Point(posX_g, posY_g), Point(iLastX_g, iLastY_g), Scalar(0, 255, 0), 2);
		//	}

		//	iLastX_g = posX_g;
		//	iLastY_g = posY_g;
		//}
		//CString red_x, red_y, green_x, green_y;
		//red_x.Format(_T("%d"), iLastX_r);
		//red_y.Format(_T("%d"), iLastY_r);
		//green_x.Format(_T("%d"), iLastX_g);
		//green_y.Format(_T("%d"), iLastY_g);

		//GetDlgItem(IDC_EDIT1)->SetWindowTextW(red_x);
		//GetDlgItem(IDC_EDIT2)->SetWindowTextW(red_y);
		//GetDlgItem(IDC_EDIT3)->SetWindowTextW(green_x);
		//GetDlgItem(IDC_EDIT4)->SetWindowTextW(green_y);
		//imshow("Smooth", Smooth);
		//imshow("Red Threshold", RobotImg[0]);
		//imshow("Green Threshold", RobotImg[1]);
		imshow("Original", frame);

		if (waitKey(33) == 27) {
			//destroyWindow("Red Threshold");
			//destroyWindow("Green Threshold");
			destroyWindow("Original");
			destroyWindow("HSV");

			break;
		}
	}

}



void CProjectRobotDlg::OnEnChangeEdit1()
{
	// TODO:  �p�G�o�O RICHEDIT ����A����N���|
	// �ǰe���i���A���D�z�мg CDialogEx::OnInitDialog()
	// �禡�M�I�s CRichEditCtrl().SetEventMask()
	// ���㦳 ENM_CHANGE �X�� ORed �[�J�B�n�C

	// TODO:  �b���[�J����i���B�z�`���{���X
}


void CProjectRobotDlg::OnEnChangeEdit2()
{
	// TODO:  �p�G�o�O RICHEDIT ����A����N���|
	// �ǰe���i���A���D�z�мg CDialogEx::OnInitDialog()
	// �禡�M�I�s CRichEditCtrl().SetEventMask()
	// ���㦳 ENM_CHANGE �X�� ORed �[�J�B�n�C

	// TODO:  �b���[�J����i���B�z�`���{���X
}


void CProjectRobotDlg::OnEnChangeEdit3()
{
	// TODO:  �p�G�o�O RICHEDIT ����A����N���|
	// �ǰe���i���A���D�z�мg CDialogEx::OnInitDialog()
	// �禡�M�I�s CRichEditCtrl().SetEventMask()
	// ���㦳 ENM_CHANGE �X�� ORed �[�J�B�n�C

	// TODO:  �b���[�J����i���B�z�`���{���X
}


void CProjectRobotDlg::OnEnChangeEdit4()
{
	// TODO:  �p�G�o�O RICHEDIT ����A����N���|
	// �ǰe���i���A���D�z�мg CDialogEx::OnInitDialog()
	// �禡�M�I�s CRichEditCtrl().SetEventMask()
	// ���㦳 ENM_CHANGE �X�� ORed �[�J�B�n�C

	// TODO:  �b���[�J����i���B�z�`���{���X
}




void CProjectRobotDlg::OnBnClickedButton8()
{
	// TODO: �b���[�J����i���B�z�`���{���X
	float s = 1.5;	//global variables
	Mat GetImg;
	Mat prvs, next; //current frame
	VideoCapture stream1(-1);//0 is the id of video device.0 if you have only one camera  
	
	//VideoCapture stream1("VideoTest.avi");
	
	if (!(stream1.read(GetImg))) //get one frame form video
		printf("Error reading webcam");
	resize(GetImg, prvs, Size(GetImg.size().width / s, GetImg.size().height / s));
	cvtColor(prvs, prvs, CV_BGR2GRAY);

	//unconditional loop  
	while (true) {
		if (!(stream1.read(GetImg))) //get one frame form video  
			break;
		//Resize
		resize(GetImg, next, Size(GetImg.size().width / s, GetImg.size().height / s));

		cvtColor(next, next, CV_BGR2GRAY);

		///////////////////////////////////////////////////////////////////

		Mat flow;
		calcOpticalFlowFarneback(prvs, next, flow, 0.5, 3, 15, 3, 5, 1.2, 0);
		Mat cflow;
		cvtColor(prvs, cflow, CV_GRAY2BGR);
		drawOptFlowMap(flow, cflow, 10, CV_RGB(0, 255, 0));
		imshow("OpticalFlowFarneback", cflow);

		imshow("prvs", prvs);
		imshow("next", next);
		if (waitKey(5) >= 0)

			break;
		prvs = next.clone();

	}
}