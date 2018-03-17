/*
Copyright (c) 2018 Roman Kazantsev
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

#pragma once

#include <cstdint>
#include <memory>
#include <unordered_map>
#include "mv.hpp"

constexpr const char FILTER_NAME[] = "ME_Kazantsev";
constexpr const char FILTER_AUTHOR[] = "Roman Kazantsev";

class MotionEstimator {
public:
	/// Constructor
	MotionEstimator(int width, int height, uint8_t quality, bool use_half_pixel);

	/// Destructor
	~MotionEstimator();

	/// Copy constructor (deleted)
	MotionEstimator(const MotionEstimator&) = delete;

	/// Move constructor
	MotionEstimator(MotionEstimator&&) = default;

	/// Copy assignment (deleted)
	MotionEstimator& operator=(const MotionEstimator&) = delete;

	/// Move assignment
	MotionEstimator& operator=(MotionEstimator&&) = default;

	/**
	 * Estimate motion between two frames
	 *
	 * @param[in] cur_Y array of pixels of the current frame
	 * @param[in] prev_Y array of pixels of the previous frame
	 * @param[in] prev_Y_up array of pixels of the previous frame shifted half a pixel up,
	 *   only valid if use_half_pixel is true
	 * @param[in] prev_Y_left array of pixels of the previous frame shifted half a pixel left,
	 *   only valid if use_half_pixel is true
	 * @param[in] prev_Y_upleft array of pixels of the previous frame shifted half a pixel up left,
	 *   only valid if use_half_pixel is true
	 * @param[in] use_half_pixel whether prev_Y_{up,left,upleft} are valid
	 * @param[out] mvectors output array of motion vectors
	 */
	void Estimate(const uint8_t* cur_Y,
	              const uint8_t* prev_Y,
	              const uint8_t* prev_Y_up,
	              const uint8_t* prev_Y_left,
	              const uint8_t* prev_Y_upleft,
	              MV* mvectors);

	/**
	 * Size of the borders added to frames by the template, in pixels.
	 * This is the most pixels your motion vectors can extend past the image border.
	 */
	static constexpr int BORDER = 16;

	/// Size of a block covered by a motion vector. Do not change.
	static constexpr int BLOCK_SIZE = 16;

private:
	/// Frame width (not including borders)
	const int width;

	/// Frame height (not including borders)
	const int height;

	/// Quality
	const uint8_t quality;

	/// Whether to use half-pixel precision
	const bool use_half_pixel;

	/// Extended frame width (including borders)
	const int width_ext;

	/// Number of blocks per X-axis
	const int num_blocks_hor;

	/// Number of blocks per Y-axis
	const int num_blocks_vert;

	/// Position of the first pixel of the frame in the extended frame
	const int first_row_offset;

	/// Array of motion vectors for the previous frame and current one
	std::vector<MV> prev_mv_map;
	std::vector<MV> curr_mv_map;

	void AdvancedSearch(MV &best_vector, std::unordered_map<ShiftDir,
		const uint8_t*> const & prev_map, const uint8_t* cur,
		int hor_offset, int vert_offset, int i, int j);

	void CrossSearch(MV &best_vector, std::unordered_map<ShiftDir,
		const uint8_t*> const & prev_map, const uint8_t* cur,
		int hor_offset, int vert_offset);

	void OrthogonalSearch(MV &best_vector, std::unordered_map<ShiftDir,
		const uint8_t*> const & prev_map, const uint8_t* cur,
		int hor_offset, int vert_offset);

	void BruteForceSearch(MV &best_vector, std::unordered_map<ShiftDir,
		const uint8_t*> const & prev_map, const uint8_t* cur,
		int hor_offset, int vert_offset);

	void MotionEstimator::CheckAndUpdateBestMV(MV &best_vector,
		ShiftDir shift_dir, const uint8_t* prev,
		const uint8_t* cur, int try_x, int try_y);

	void StoreBestMVInMap(MV const &best_vector, int const ind);
};
