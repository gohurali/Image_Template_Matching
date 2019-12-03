// Author: Gohur Ali
// Version: 11202019
// transformer.h
// Transformer Header file that outlines the
// functions to be implemented that perform
// image contrasting and template image matching
// on an imput search image
#ifndef TRANSFORMER_H
#define TRANSFORMER_H
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <vector>
#include <iostream>
#include <stdio.h>
class Transformer {
public:

	// enhance_contrast
	// Preconditions:		Average B,G,R color spaces must be 
	//						previously calculated in order to 
	//						apply a contrast to the input image
	// Postconditions:		The image has a contrast applied to it
	cv::Mat enhance_contrast(cv::Mat&,std::vector<int>);

	// enhance_color
	// Preconditions:		Must have the current pixel and its colors
	//						the average color and the k-factor to be 
	//						able to compute the constrasted pixel.
	// Postconditions:		Computes the diff between the current and
	//						average pixel and multiplies that with the
	//						k-factor. There are checks to ensure pixel
	//						values stay within the unsigned char range
	int enhance_color(int, int, float);

	// average_BGR
	// Preconditions:		Must be able to provide the current image
	//						and the total number of pixels for each 
	//						color space, which should be the same. Keep
	//						in mind this is a simple calculation of rows
	//						mulitplied by cols.
	// Postconditions:		Computes the the sum of all the pixel values
	//						for each color space and then divides by the
	//						total number of pixels for each color space
	//						thus returning the average pixel for each color
	std::vector<int> average_BGR(cv::Mat&, std::vector<int> bgr_totals);

	// edge_compare
	// Preconditions:		Must have read int the template image and the 
	//						search image. These images are to be resized,
	//						smoothed, and edge detected prior to being 
	//						passed in as parameters to the function
	// Postconditions:		By utilizing a sliding window approach, we 
	//						run the template image over the search image
	//						and calculate the sum of matching edge pixels
	//						and return a pair of all total matches for each
	//						stride of the window and a matrix format of where
	//						each match total was found
	std::pair<std::vector<int>, cv::Mat> edge_compare(cv::Mat&, cv::Mat&);

	// show_match
	// Preconditions:		Must have the pair output from edge_compare
	//						function and both search and template image
	// Postconditions:		Creates an image of where the template is found
	//						in the search image.
	cv::Mat show_match(cv::Mat&, cv::Mat&, std::pair<std::vector<int>, cv::Mat>&,bool);
};
#endif