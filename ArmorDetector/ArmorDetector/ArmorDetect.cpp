#include <iostream> // for standard I/O
#include <string> // for strings
#include <iomanip> // for controlling float print precision
#include <sstream> // string to number conversion
#include <opencv2/core/core.hpp> // Basic OpenCV structures (cv::Mat, Scalar)
#include <opencv2/imgproc/imgproc.hpp> // Gaussian Blur
#include <opencv2/highgui/highgui.hpp> // OpenCV window I/O
#include <Windows.h>

using namespace cv;
using namespace std;

//�ο�http://www.linuxidc.com/Linux/2014-04/100001.htm
//-----------------------------------------------
//����Ƶ���ļ�����Ϊ����ID����ɴ�����ͷ��Ĭ������ͷΪ0��
//VideoCapture capture("../video.avi"); // ����1
//capture.open("../video.avi");			// ����2 

//----------------------------------------------
//��ȡ��Ƶ֡
//capture.read(frame);		//����һ
//capture.grab();			//������
//capture.retrieve(frame);

//capture >> frame;			//������

//-----------------------------------------------
//��ȡ��Ƶ�Ĳ���
//double rate = capture.get(CV_CAP_PROP_FPS); // ��ȡ
//long nFrame = static_cast<long>(capture.get(CV_CAP_PROP_FRAME_COUNT)); // ��ȡ��֡��



#define T_ANGLE_THRE 10
#define T_SIZE_THRE 5

void ColorAdjust(Mat& src, Mat& dst, double alpha, double beta);
void GetDiffImage(Mat& src1, Mat& src2, Mat& dst, int threshold);

int main()
{
	VideoCapture video;
	video.open("RedCar.avi");
	if (!video.isOpened())
	{
		cout << "��Ƶδ�򿪣�\n";
		cin.get();
		cin.get();
		return -1;
	}

	int frame_count = video.get(CV_CAP_PROP_FRAME_COUNT);
	cout << "֡����" << frame_count << endl;

	Mat img, img_gray, hsv, img_h, threshold_output;
	int gray_thresh = 180, hue_thresh = 20;
	for (int i = 0; i < frame_count; i++)
	{
		video >> img;
		if (img.empty())
		{
			cout << "δ����ͼ��\n";
			cin.get();
			cin.get();
			return -1;
		}
		cout << "��" << i << "֡\n";
		//��˹�˲�
		GaussianBlur(img, img, Size(11, 11), 0);
		//ColorAdjust(img, img, 1, -120);//.............��������

		
		cvtColor(img, img_gray, CV_BGR2GRAY);
		cvtColor(img, hsv, CV_BGR2HSV);
		img_h.create(hsv.size(), hsv.depth());
		int fromto[2] = { 0, 0 };
		mixChannels(&hsv, 1, &img_h, 1, fromto, 1);

		vector<vector<Point> > contours;
		vector<Vec4i> hierarchy;
		/// Detect edges using Threshold
		threshold(img_gray, threshold_output, gray_thresh, 255, THRESH_BINARY);
		threshold(img_h, img_h, hue_thresh, 255, THRESH_BINARY);
		//adaptiveThreshold(img_gray, threshold_output, 255, CV_ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY, 31, 10);		//����Ӧ��ֵЧ����������
		imshow("thresh", threshold_output);
		//�� ��
		int erosion_size = 3;
		Mat element = getStructuringElement(MORPH_ELLIPSE,
			Size(2 * erosion_size + 1, 2 * erosion_size + 1),
			Point(erosion_size, erosion_size));
		erode(threshold_output, threshold_output, element, Point(-1, -1), 1);
		dilate(threshold_output, threshold_output, element, Point(-1, -1), 1);
//		erode(img_h, img_h, element, Point(-1, -1), 1);
//		dilate(img_h, img_h, element, Point(-1, -1), 1);
//		element = getStructuringElement(MORPH_ELLIPSE,
//			Size(2 * 2 + 1, 2 * 2 + 1),
//			Point(2, 2));
//		dilate(img_h, img_h, element, Point(-1, -1), 1);
		imshow("opened", threshold_output);
		/// Find contours
		findContours(threshold_output, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0));

		vector<RotatedRect> minRect;

		for (int i = 0; i < contours.size(); i++)
		{
			//������򲻹��������
			double aver_x = 0, diff_x = 0;
			for (int p = 0; p < contours[i].size(); p++)
			{
				aver_x += contours[i][p].x;
			}
			aver_x /= contours[i].size();
			for (int p = 0; p < contours[i].size(); p++)
			{
				diff_x += abs(contours[i][p].x - aver_x);
			}
			diff_x /= contours[i].size();
			//����ڰ�����
			RotatedRect r = minAreaRect(Mat(contours[i]));
			Point2f rect_points[4];
			r.points(rect_points);
			vector<Point2f> rp;
			rp.push_back(rect_points[0]);
			rp.push_back(rect_points[1]);
			rp.push_back(rect_points[2]);
			rp.push_back(rect_points[3]);
			double ca = contourArea(contours[i]);
			double ra = contourArea(rp);

			//�����ҵ��ľ���
			if (diff_x < 10 && ca/ra > 0.8 && ra > 200)
			{
				minRect.push_back(r);
			}
		}
		//Draw contours + rotated rects + ellipses
		Mat drawing = Mat::zeros(threshold_output.size(), CV_8UC3);
		for (int i = 0; i < contours.size(); i++)
		{
			Scalar color = Scalar(0, 255, 0);
			//	// contour
			drawContours(drawing, contours, i, color, 1, 8, vector<Vec4i>(), 0, Point());
			//	// ellipse
			//	//ellipse(drawing, minEllipse[i], color, 2, 8);
			// rotated rectangle
			if (i < minRect.size())
			{
				Point2f rect_points[4]; minRect[i].points(rect_points);
				for (int j = 0; j < 4; j++)
					line(drawing, rect_points[j], rect_points[(j + 1) % 4], Scalar(255, 0, 0), 1, 8);
			}
		}

		//���ҵ�����ת���ν���ʶ��
		//���������������̫С��
		//for (int i = 0; i < minRect.size(); i++)
		//{
		//	//if (abs(minRect[i].size.height - minRect[i].size.width) < 10)
		//	//{
		//		//minRect.erase(minRect.begin() + i);
		//		//vector<RotatedRect>::iterator it = find(minRect.begin(), minRect.end(), minRect[i]);
		//		//minRect.erase(it);
		//	//}
		//}
		//for (vector<RotatedRect>::iterator it = minRect.begin(); it != minRect.end(); it++)
		//{
		//	if (abs((*it).size.height - (*it).size.width) < 50 || (*it).size.width * (*it).size.height < 200)
		//	{
		//		//vector<RotatedRect>::iterator rm = it;
		//		//minRect.erase(rm);		//erase֮��it�Ǹ�Ұָ��  

		//		it = minRect.erase(it); //�ᵼ�±�ɾ������һ��������
		//		it--;
		//	}
		//	if (it == minRect.end()) //Ҫ���Ƶ��������ܳ����������� !!!!!!!!!!!!!!!!!!!!!!!
		//	{
		//		break;
		//	}
		//}
		////������֡
		cout << "�ҵ�" << minRect.size() << "������\n";
		if (minRect.size() == 0)
			continue;
		
		//for (int i = 0; i< minRect.size(); i++)
		//{
		//	Scalar color = Scalar(255, 255, 255);
		//	Point2f rect_points[4]; minRect[i].points(rect_points);
		//	for (int j = 0; j < 4; j++)
		//		line(drawing, rect_points[j], rect_points[(j + 1) % 4], color, 1, 8);
		//}

 		for (int i = 0; i < minRect.size() - 1; i++)
		{
			for (int j = i + 1; j < minRect.size(); j++)
			{
				Rect r1 = minRect[i].boundingRect();
				Rect r2 = minRect[j].boundingRect();
				
				//-90~...(CV_PI/180)
				if (//abs(abs(minRect[i].angle) - 90) < 20 && abs(abs(minRect[j].angle) - 90) < 20 &&	//�Ƕȶ��ӽ���ֱ...0��90 ...ת����
					abs(max(r1.height,r1.width) - max(r2.width, r2.height)) < 50 &&
					abs(min(r1.width, r1.height) - min(r2.width, r2.height)) < 50 &&				//��״����...������ ��͸�
					abs(r1.y - r2.y) < 30 &&															//���Ͻ�������ӽ�
					abs(minRect[i].center.y - minRect[j].center.y) < 30	&&							//����������ӽ�
					r1.area() > 500 && r2.area() > 500
					//abs(abs(sin(CV_PI / 180.0 * minRect[i].angle)) - 1) < 0.05 || abs(abs(sin(CV_PI / 180.0 * minRect[i].angle)) - 0) < 0.05 &&
					//abs(abs(sin(CV_PI / 180.0 * minRect[j].angle)) - 1) < 0.05 || abs(abs(sin(CV_PI / 180.0 * minRect[j].angle)) - 0) < 0.05 
					)
				{
					Point ArmorCenter;
					ArmorCenter.x = minRect[i].center.x + minRect[j].center.x;
					ArmorCenter.x /= 2;
					ArmorCenter.y = minRect[i].center.y + minRect[j].center.y;
					ArmorCenter.y /= 2;

					rectangle(img, r1, Scalar(255, 0, 0), 3);
					rectangle(img, r2, Scalar(255, 0, 0), 3);
					circle(img, ArmorCenter, 3, Scalar(0, 0, 255), 3);
				}
			}
		}

		imshow("Contours", drawing);
		imshow("vedio", img);
		waitKey(50);
	}

}

void ColorAdjust(Mat& src, Mat& dst, double alpha, double beta)
{
	for (int y = 0; y < src.rows; y++)
	{
		for (int x = 0; x < src.cols; x++)
		{
			for (int c = 0; c < 3; c++)
			{
				dst.at<Vec3b>(y, x)[c] =
					saturate_cast<uchar>(alpha*(src.at<Vec3b>(y, x)[c]) + beta);
			}
		}
	}
}

void GetDiffImage(Mat& src1, Mat& src2, Mat& dst, int threshold)
{
	for (int y = 0; y < src1.rows; y++)
	{
		for (int x = 0; x < src1.cols; x++)
		{
			if (src1.at<uchar>(y, x) - src2.at<uchar>(y, x) > threshold)
				dst.at<uchar>(y, x) = 255;
			else
				dst.at<uchar>(y, x) = 0;
		}
	}
}