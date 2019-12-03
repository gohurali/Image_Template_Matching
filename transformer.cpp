#include "transformer.h"

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
std::vector<int> Transformer::average_BGR(cv::Mat& img,std::vector<int> bgr_totals) {
	float total_r = 0.0;
	float total_g = 0.0;
	float total_b = 0.0;
	for (int row = 0; row < img.rows; row++) {
		for (int col = 0; col < img.cols; col++) {
			int current_b = img.at<cv::Vec3b>(row, col)[0];
			int current_g = img.at<cv::Vec3b>(row, col)[1];
			int current_r = img.at<cv::Vec3b>(row, col)[2];
			total_b += current_b;
			total_g += current_g;
			total_r += current_r;
		}
	}
	int ave_b = static_cast<int>(total_b / bgr_totals[0]);
	int ave_g = static_cast<int>(total_g / bgr_totals[1]);
	int ave_r = static_cast<int>(total_r / bgr_totals[2]);
	std::vector<int> averages = { ave_b,ave_g,ave_r};
	return averages;
}

// enhance_color
// Preconditions:		Must have the current pixel and its colors
//						the average color and the k-factor to be 
//						able to compute the constrasted pixel.
// Postconditions:		Computes the diff between the current and
//						average pixel and multiplies that with the
//						k-factor. There are checks to ensure pixel
//						values stay within the unsigned char range
int Transformer::enhance_color(int current_color, int average_color, float factor) {
	float diff = (current_color - average_color) * factor;
	float new_val = average_color + diff;
	int new_val_rounded = static_cast<int>(new_val);
	new_val_rounded = new_val_rounded < 0 ? 0 : new_val_rounded;
	new_val_rounded = new_val_rounded > 255 ? 255 : new_val_rounded;
	return new_val_rounded;
}

// enhance_contrast
// Preconditions:		Average B,G,R color spaces must be 
//						previously calculated in order to 
//						apply a contrast to the input image
// Postconditions:		The image has a contrast applied to it
cv::Mat Transformer::enhance_contrast(cv::Mat& img,std::vector<int>averages) {
	for (int row = 0; row < img.rows; row++) {
		for (int col = 0; col < img.cols; col++) {
			int current_b = img.at<cv::Vec3b>(row, col)[0];
			int current_g = img.at<cv::Vec3b>(row, col)[1];
			int current_r = img.at<cv::Vec3b>(row, col)[2];
			int new_b = this->enhance_color(current_b, averages[0], 1.5);
			int new_g = this->enhance_color(current_g, averages[1], 1.5);
			int new_r = this->enhance_color(current_r, averages[2], 1.5);
			img.at<cv::Vec3b>(row, col)[0] = new_b;
			img.at<cv::Vec3b>(row, col)[1] = new_g;
			img.at<cv::Vec3b>(row, col)[2] = new_r;
		}
	}
	return img;
}


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
std::pair<std::vector<int>,cv::Mat> Transformer::edge_compare(cv::Mat& template_im, cv::Mat& search_im) {
	cv::Mat match_sum_img(search_im.rows, search_im.cols,CV_32S,cv::Scalar(0));
	std::vector<int> match_counts;
	for (int s_row = 0; s_row < search_im.rows - template_im.rows; s_row++) {
		for (int s_col = 0; s_col < search_im.cols - template_im.cols; s_col++) {
			int t_rows = template_im.rows;
			int t_cols = template_im.cols;
			int total_matches = 0;
			int s_row_over = s_row;
			int s_col_over = s_col;
			for (int t_row = 0, s_row_over = s_row; t_row < t_rows; t_row++, s_row_over++) {
				for (int t_col = 0, s_col_over = s_col; t_col < t_cols; t_col++, s_col_over++) {
					int curr_s_pxl = search_im.at<uchar>(s_row_over, s_col_over);
					int curr_t_pxl = template_im.at<uchar>(t_row, t_col);
					if (curr_s_pxl == 255 && curr_t_pxl == 255) {
						total_matches++;
					}
				}
			}
			match_counts.push_back(total_matches);
			match_sum_img.at<int>(s_row, s_col) = total_matches;
		}
	}
	std::pair<std::vector<int>, cv::Mat> output = { match_counts,match_sum_img };
	return output;
}

// show_match
// Preconditions:		Must have the pair output from edge_compare
//						function and both search and template image
// Postconditions:		Creates an image of where the template is found
//						in the search image.
cv::Mat Transformer::show_match(cv::Mat& search_im, cv::Mat& template_im, std::pair<std::vector<int>, cv::Mat>& info, bool view_only_roi) {
	// Get the maximum count
	int max_match_count = *std::max_element(info.first.begin(), info.first.end());
	int m_row = 0;
	int m_col = 0;
	cv::Mat output(search_im.rows, search_im.cols, CV_8UC3, cv::Scalar(0));
	for (int s_row = 0; s_row < search_im.rows; s_row++) {
		for (int s_col = 0; s_col < search_im.cols; s_col++) {
			int curr_count = info.second.at<int>(s_row, s_col);
			// only look at the location where we find the max
			if (curr_count == max_match_count) {
				// Draw contents of whats in search im

				if (view_only_roi) {
					// Shows only the patch of where the match is located
					for (int si_row = s_row; si_row < s_row + template_im.rows; si_row++) {
						for (int si_col = s_col; si_col < s_col + template_im.cols; si_col++) {
							output.at<cv::Vec3b>(si_row, si_col)[2] = search_im.at<uchar>(si_row, si_col);
						}
					}
				}
				else {
					// Shows the entire search image
					for (int si_row = 0; si_row < search_im.rows; si_row++) {
						for (int si_col = 0; si_col < search_im.cols; si_col++) {
							output.at<cv::Vec3b>(si_row, si_col)[2] = search_im.at<uchar>(si_row, si_col);
						}
					}
				}

				// Draw contents of what in the template im
				for (int si_row = s_row, t_row = 0; t_row < template_im.rows; si_row++, t_row++) {
					for (int si_col = s_col, t_col = 0; t_col < template_im.cols; si_col++, t_col++) {
						output.at<cv::Vec3b>(si_row, si_col)[1] = template_im.at<uchar>(t_row, t_col);
					}
				}
			}
			if (s_col % template_im.cols) {
				m_col += template_im.cols;
			}
		}
		if (s_row % template_im.rows) {
			m_row += template_im.rows;
		}
	}
	return output;
}