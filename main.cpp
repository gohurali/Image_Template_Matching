#include "transformer.h"
void show_image(cv::Mat& img, int x_window, int y_window, int delay) {
	cv::namedWindow("output", cv::WINDOW_NORMAL);
	cv::resizeWindow("output", img.cols / x_window, img.rows / y_window);
	cv::imshow("output", img);
	cv::waitKey(delay);
}

int main() {
	cv::Mat img = cv::imread("background.jpg");
	
	Transformer tf;
	std::vector<int> total_bgr = {img.rows * img.cols, img.rows * img.cols, img.rows * img.cols };
	printf("B: %d -- G: %d -- R: %d\n",total_bgr[0],total_bgr[1],total_bgr[2]);
	std::vector<int> ave_bgr = tf.average_BGR(img, total_bgr);
	printf("B: %d -- G: %d -- R: %d\n", ave_bgr[0], ave_bgr[1], ave_bgr[2]);
	cv::Mat enhanced_img = tf.enhance_contrast(img, ave_bgr);
	cv::imwrite("prob1_output.jpg", enhanced_img);

	// Problem 2
	cv::Mat big_img = cv::imread("prob2_large.png");
	cv::Mat small_img = cv::imread("prob2_small.png");
	cv::resize(big_img, big_img, { 75,100 });//{ 50,80 });
	cv::resize(small_img, small_img, { 30,30 });//{ 25,25 });

	// Obtain edges
	cv::Mat big_edges = big_img.clone();
	cv::Mat small_edges = small_img.clone();
	cv::GaussianBlur(big_edges, big_edges, { 7,7 }, 100.0, 200.0);
	cv::GaussianBlur(small_edges, small_edges, { 7,7 }, 100.0, 200.0);

	cv::Canny(big_img, big_edges, 120, 140);
	cv::Canny(small_img, small_edges, 120, 140);
	
	cv::imwrite("search_im.png", big_edges);
	cv::imwrite("template_im.png", small_edges);

	int edge_pxs = 0;
	for (int t_row = 0; t_row < small_edges.rows; t_row++) {
		for (int t_col = 0; t_col < small_edges.cols; t_col++) {
			int curr = small_edges.at<uchar>(t_row, t_col);
			if (curr == 255)
				edge_pxs++;
		}
	}
	printf("TOTAL EDGES IN TEMPLATE = %i\n", edge_pxs);

	//std::cout << cv::format(small_edges, cv::Formatter::FMT_PYTHON) << std::endl;
	//show_image(big_edges, 1, 1, 5000);
	//show_image(small_edges, 1, 1, 5000);
	std::pair<std::vector<int>,cv::Mat> output = tf.edge_compare(small_edges, big_edges);
	
	cv::Mat out = tf.show_match(big_edges, small_edges, output);
	printf("output siz: r = %i ---- c = %i ", out.rows, out.cols);
	show_image(out, 1, 1, 5000);
	cv::imwrite("matches.png", out);
	return 0;
}