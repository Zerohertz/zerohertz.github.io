---
title: Line Detection
date: 2018-08-16 04:17:00
categories:
- Etc.
tags:
- C, C++
---
```C++
#include <iostream>
#include <raspicam/raspicam_cv.h>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
 
using namespace std;
using namespace cv; 
typedef struct {
	unsigned char r;
	unsigned char g;
	unsigned char b;
} Color;
typedef struct PosList {
	int x;
	int y;
	struct PosList *next;
} PosList;
int colordiff(Color a, Color b)
{	int dr, dg, db;
	dr = (int)((a.r < b.r) ? (b.r - a.r) : (a.r - b.r));
	dg = (int)((a.g < b.g) ? (b.g - a.g) : (a.g - b.g));
	db = (int)((a.b < b.b) ? (b.b - a.b) : (a.b - b.b));
	return dr + dg + db;
}PosList* newnode(int x, int y)
{	PosList *pos;
	pos = (PosList *)malloc(sizeof(PosList));
	pos->x = x;
	pos->y = y;
	pos->next = NULL;
	return pos;
}void delnode(PosList **pos)
{	free(*pos);
	*pos = NULL;
}void pl_push(PosList **list, PosList *pos)
{	pos->next = *list;
	*list = pos;
}PosList* pl_pop(PosList **list)
{	PosList *pos;
	pos = *list;
	*list = (*list)->next;
	return pos;
}void dellist(PosList **list)
{	PosList *a, *b;
	a = *list;
	while (a != NULL) {
		b = a->next;
		delnode(&a);
		a = b;
	}
}int contains(PosList *list, int x, int y)
{	while (list != NULL) {
		if (list->x == x && list->y == y)
			return 1;
		list = list->next;
	}
	return 0;
}void rgrow(IplImage *source, IplImage *dest, Color color, int threshold)
{	PosList *list_n;
	PosList *node_r;
	PosList *list_r;
	Color curcolor;
	int x, y;
	int sx, sy;
	int dx, dy;
	int offset;
	int mindiff, curdiff;
	mindiff = 255 * 3;
	for (y = 0; y < source->height; y++) {
		for (x = 0; x < source->width; x++) {
			offset = y * source->width * 3 + x * 3;
			curcolor.b = source->imageData[offset];
			curcolor.g = source->imageData[offset + 1];
			curcolor.r = source->imageData[offset + 2];
			curdiff = colordiff(color, curcolor);
			if (curdiff < mindiff) {
				sx = x;
				sy = y;
				mindiff = curdiff;
			}
			dest->imageData[y * dest->width + x] = 0;
		}
	}
	list_n = newnode(sx, sy);
	list_r = NULL;
		int* map = new int[source->width*source->height];
	for (int i = 0; i<source->width*source->height; i++)
		map[i] = 0;
	map[sy*source->width + sx] = 1;
	while (list_n != NULL) 
	{
		sx = list_n->x;
		sy = list_n->y;
		pl_push(&list_r, pl_pop(&list_n));
		for (dy = -1; dy <= 1; dy++) 
		{
			for (dx = -1; dx <= 1; dx++) 
			{
				if (dx == 0 && dy == 0)
					continue;
				if (dx != 0 && dy != 0)
					continue;
				if (sx + dx == -1 || sx + dx == source->width ||
					sy + dy == -1 || sy + dy == source->height)
					continue;
				if (map[(sy + dy)*source->width + sx + dx] == 1)
					continue;
				offset = (sy + dy) * source->width * 3 + (sx + dx) * 3;
				curcolor.b = source->imageData[offset];
				curcolor.g = source->imageData[offset + 1];
				curcolor.r = source->imageData[offset + 2];
				curdiff = colordiff(color, curcolor);
				if (curdiff <= threshold)
				{
					pl_push(&list_n, newnode(sx + dx, sy + dy));
					map[(sy + dy)*source->width + sx + dx] = 1;
				}
			}
		}
	}
	delete [] map;
	node_r = list_r;
	while (node_r != NULL) {
		dest->imageData[node_r->y * dest->width + node_r->x] = 255;
		node_r = node_r->next;
	}
	dellist(&list_r);
}float sum1(std::vector<float>* x, std::vector<float>* y, float yCurr)
{	float sum = 0;
	for (int i = 0; i < x->size(); i++)
	{
		sum += ((*x)[i] * ((*y)[i] - yCurr));
	}
	return sum / x->size()*-2;
}float sum2(std::vector<float>* x, std::vector<float>* y, float yCurr)
{	float sum = 0;
	for (int i = 0; i < x->size(); i++)
	{
		sum += ((*y)[i] - yCurr);
	}
	return sum / x->size()*-2;
}void linearRegression(std::vector<float>* x, std::vector<float>* y, int nbData, float& b0, float& b1)
{	float xave = 0;
	float yave = 0;
	for (int i = 0; i < nbData; i++)
	{
		xave += (*x)[i];
		yave += (*y)[i];
	}
	xave /= (float)nbData;
	yave /= (float)nbData;
	float a1 = 0;
	float a2 = 0;
	for (int i = 0; i < nbData; i++)
	{
		a1 += ((*x)[i] - xave)*((*x)[i] - xave);
		a2 = a2 + ((*x)[i] - xave)*((*y)[i] - yave);
	}
	b1 = a2 / a1;
	b0 = yave - b1*xave;
}void rgrow(Mat *source, Mat *dest, Color color, int threshold)
{	PosList *list_n;
	PosList *node_r;
	PosList *list_r;
	Color curcolor;
	int x, y;
	int sx, sy;
	int dx, dy;
	int offset;
	int mindiff, curdiff;
	int width = source->cols;
	int height = source->rows;
	mindiff = 255 * 3;
	for (y = 0; y < height; y++) {
		for (x = 0; x < width; x++) {
			offset = y * width * 3 + x * 3;
			curcolor.b = source->data[offset];
			curcolor.g = source->data[offset + 1];
			curcolor.r = source->data[offset + 2];
			curdiff = colordiff(color, curcolor);
			if (curdiff < mindiff) {
				sx = x;
				sy = y;
				mindiff = curdiff;
			}
			dest->data[y * width + x] = 0;
		}
	}
	list_n = newnode(sx, sy);
	list_r = NULL;
	int* map = new int[width*height];
	for (int i = 0; i<width*height; i++)
		map[i] = 0;
	map[sy*width + sx] = 1;
	while (list_n != NULL)
	{
		sx = list_n->x;
		sy = list_n->y;
		pl_push(&list_r, pl_pop(&list_n));
		for (dy = -1; dy <= 1; dy++)
		{
			for (dx = -1; dx <= 1; dx++)
			{
				if (dx == 0 && dy == 0)
					continue;
				if (dx != 0 && dy != 0)
					continue;
				if (sx + dx == -1 || sx + dx == width ||
					sy + dy == -1 || sy + dy == height)
					continue;
				if (map[(sy + dy)*width + sx + dx] == 1)
					continue;
				offset = (sy + dy) * width * 3 + (sx + dx) * 3;
				curcolor.b = source->data[offset];
				curcolor.g = source->data[offset + 1];
				curcolor.r = source->data[offset + 2];
				curdiff = colordiff(color, curcolor);
				if (curdiff <= threshold)
				{
					pl_push(&list_n, newnode(sx + dx, sy + dy));
					map[(sy + dy)*width + sx + dx] = 1;
				}
			}
		}
	}
	delete[] map;
	node_r = list_r;
	while (node_r != NULL) {
		dest->data[node_r->y * width + node_r->x] = 255;
		node_r = node_r->next;
	}
	dellist(&list_r);
}void findCenterLine(Mat* img, Vec4i& l)
{	int imageHeight=img->rows;
	int imageWidth=img->cols;
	Mat dist(img->rows, img->cols, CV_8UC1);
	Mat dist1(img->rows, img->cols, CV_8UC1);
	distanceTransform(*img, dist, DIST_C, CV_DIST_MASK_PRECISE);
	normalize(dist, dist, 0.0, 1.0, NORM_MINMAX);
	std::vector<float> array;
	if (dist.isContinuous()) 
	{
		array.assign((float*)dist.datastart, (float*)dist.dataend);
	}
	else 
	{
		for (int i = 0; i < dist.rows; ++i) {
			array.insert(array.end(), dist.ptr<float>(i), dist.ptr<float>(i) + dist.cols);
		}
	}
	double max = 0;
	for (int i = 0; i < imageHeight*imageWidth; i++)
	{
		if (array[i]>max)
		{
			max = array[i];
		}
	}
		std::vector<float> x;
	std::vector<float> y;
	for (int i = 0; i < imageHeight; i++)
	{
		for (int j = 0; j < imageWidth; j++)
		{
			if (array[i*imageWidth+j]>max*0.1)
			{
				dist1.data[i*imageWidth + j]=255;
				x.push_back((float)j);
				y.push_back((float)i);
			}
			else
			{
				dist1.data[i*imageWidth + j] = 0;
			}
		}
	}
	float b0, b1;
	linearRegression(&x, &y, x.size(), b0, b1);
	//printf("%f %f\n",b0,b1);
	l[0] = 0; l[1] = b0 + b1*l[0];
	if(l[1]<=0)
	{
		l[1]=0;
		l[0]=(l[1]-b0)/b1;
	}
	if(l[1]>=imageHeight)
	{
		l[1]=imageHeight;
		l[0]=(l[1]-b0)/b1;
	}
		l[2] = imageWidth; l[3] = b0 + b1*l[2];
	if(l[3]<=0)
	{
		l[3]=0;
		l[2]=(l[3]-b0)/b1;
	}
	if(l[3]>=imageHeight)
	{
		l[3]=imageHeight;
		l[2]=(l[3]-b0)/b1;
	}
	imshow("Dist", dist);
	//imshow("Dist1", dist1);
}
int main (void)
{	int imageWidth = 640;
	int imageHeight = 480;

    raspicam::RaspiCam_Cv Camera;
    Camera.set( CV_CAP_PROP_FORMAT, CV_8UC3);
    Camera.set( CV_CAP_PROP_FRAME_WIDTH, imageWidth);
    Camera.set( CV_CAP_PROP_FRAME_HEIGHT, imageHeight);
		Mat orgImg;
	Mat cvEdge;
	Mat filteredImg(imageHeight, imageWidth, CV_8UC1);

    if (!Camera.open()) {cerr<<"Error opening the camera"<<endl;return -1;}
		while(1)
	{
		Camera.grab();
		Camera.retrieve(orgImg);
		CV_Assert(orgImg.data);
    	blur(orgImg, orgImg, Size(10,10));
		int threshold = 250;
		Color color; color.r = 255; color.g = 0; color.b = 0;
		rgrow(&orgImg, &filteredImg, color, threshold);
		
		//Canny Edge detector
		Canny(orgImg, cvEdge, 10, 10*3, 3);
		//Fine center line
		Vec4i lineSeg;
		findCenterLine(&filteredImg, lineSeg);
		line(orgImg, Point(lineSeg[0], lineSeg[1]), Point(lineSeg[2], lineSeg[3]), Scalar(255, 0, 255), 3, CV_AA); 
		
		imshow("Org", orgImg);
		imshow("Filtered", filteredImg);
		imshow("Edge", cvEdge);
		
		if ( waitKey(20) == 27 )break; //ESC키 누르면 종료
	}
 
    Camera.release();
}
```
