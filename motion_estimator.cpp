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

#include <unordered_map>
#include "metric.hpp"
#include "motion_estimator.hpp"

MotionEstimator::MotionEstimator(int width, int height, uint8_t quality, bool use_half_pixel)
	: width(width)
	, height(height)
	, quality(quality)
	, use_half_pixel(use_half_pixel)
	, width_ext(width + 2 * BORDER)
	, num_blocks_hor((width + BLOCK_SIZE - 1) / BLOCK_SIZE)
	, num_blocks_vert((height + BLOCK_SIZE - 1) / BLOCK_SIZE)
	, first_row_offset(width_ext * BORDER + BORDER) {

	// MV map for the previous frame
	prev_mv_map.resize(num_blocks_hor * num_blocks_vert,
		MV(0, 0, ShiftDir::NONE, std::numeric_limits<long>::max()));

	// MV map for the current frame
	curr_mv_map.resize(num_blocks_hor * num_blocks_vert,
		MV(0, 0, ShiftDir::NONE, std::numeric_limits<long>::max()));
}

MotionEstimator::~MotionEstimator() {
	// PUT YOUR CODE HERE
}

void MotionEstimator::CheckAndUpdateBestMV(MV &best_vector,
	ShiftDir shift_dir, const uint8_t* prev,
	const uint8_t* cur, int try_x, int try_y) {
	auto const comp = prev + try_y * width_ext + try_x;
	long tmp_error = GetErrorSAD_16x16(cur, comp, width_ext);
	if (tmp_error < best_vector.error) {
		best_vector.error = tmp_error;
		best_vector.x = try_x;
		best_vector.y = try_y;
		best_vector.shift_dir = shift_dir;
	}
}

void MotionEstimator::StoreBestMVInMap(MV const &best_vector, int const ind) {
	curr_mv_map[ind].x = best_vector.x;
	curr_mv_map[ind].y = best_vector.y;
	curr_mv_map[ind].error = best_vector.error;
	curr_mv_map[ind].shift_dir = best_vector.shift_dir;
}

void MotionEstimator::CrossSearch(MV &best_vector,
	const std::unordered_map<ShiftDir, const uint8_t*> &prev_map,
	const uint8_t* cur, int hor_offset, int vert_offset) {
	for (const auto& prev_pair : prev_map) {
		const auto prev = prev_pair.second + vert_offset + hor_offset;
		int template_size = BORDER / 4;

		// compute log2_template_size
		int log2_template_size = 0;
		int tmp = template_size + 1;
		while (tmp >>= 1) ++log2_template_size;

		int max_num_steps = log2_template_size * 4;
		int num_steps = quality * max_num_steps / 100;

		int cur_x = 0, cur_y = 0;
		while (template_size > 0) {
			// check bottom left corner
			int tmp_x = cur_x - template_size;
			int tmp_y = cur_y - template_size;
			if (tmp_x >= -BORDER &&
				tmp_y >= -BORDER) {
				auto const comp = prev + tmp_y * width_ext + tmp_x;
				long tmp_error = GetErrorSAD_16x16(cur, comp, width_ext);
				if (tmp_error < best_vector.error) {
					best_vector.error = tmp_error;
					best_vector.x = tmp_x;
					best_vector.y = tmp_y;
					best_vector.shift_dir = prev_pair.first;
				}
				if (--num_steps <= 0) break;
			}

			// check top left corner
			tmp_x = cur_x - template_size;
			tmp_y = cur_y + template_size;
			if (tmp_x >= -BORDER &&
				tmp_y <= BORDER) {
				auto const comp = prev + tmp_y * width_ext + tmp_x;
				long tmp_error = GetErrorSAD_16x16(cur, comp, width_ext);
				if (tmp_error < best_vector.error) {
					best_vector.error = tmp_error;
					best_vector.x = tmp_x;
					best_vector.y = tmp_y;
					best_vector.shift_dir = prev_pair.first;
				}
				if (--num_steps <= 0) break;
			}

			// check top right corner
			tmp_x = cur_x + template_size;
			tmp_y = cur_y + template_size;
			if (tmp_x <= BORDER &&
				tmp_y <= BORDER) {
				auto const comp = prev + tmp_y * width_ext + tmp_x;
				long tmp_error = GetErrorSAD_16x16(cur, comp, width_ext);
				if (tmp_error < best_vector.error) {
					best_vector.error = tmp_error;
					best_vector.x = tmp_x;
					best_vector.y = tmp_y;
					best_vector.shift_dir = prev_pair.first;
				}
				if (--num_steps <= 0) break;
			}

			// check bottom right corner
			tmp_x = cur_x + template_size;
			tmp_y = cur_y - template_size;
			if (tmp_x <= BORDER &&
				tmp_y >= -BORDER) {
				auto const comp = prev + tmp_y * width_ext + tmp_x;
				long tmp_error = GetErrorSAD_16x16(cur, comp, width_ext);
				if (tmp_error < best_vector.error) {
					best_vector.error = tmp_error;
					best_vector.x = tmp_x;
					best_vector.y = tmp_y;
					best_vector.shift_dir = prev_pair.first;
				}
				if (--num_steps <= 0) break;
			}

			cur_x = best_vector.x;
			cur_y = best_vector.y;
			template_size /= 2;
		}
	}
}

void MotionEstimator::AdvancedSearch(MV &best_vector, std::unordered_map<ShiftDir,
	const uint8_t*> const & prev_map, const uint8_t* cur,
	int hor_offset, int vert_offset, int i, int j) {
	int error_threshold = 1; // after which search is ended
	int step_size = BORDER / 4; // step size for gradient descent
	int num_blocks_vert = width / BLOCK_SIZE;
	int num_blocks_hor = height / BLOCK_SIZE;

	// compute log2_template_size and number fo steps for given quality
	int log2_step_size = 0;
	int tmp = step_size + 1;
	while (tmp >>= 1) ++log2_step_size;
	int max_steps = 4 * log2_step_size + 10;
	if (use_half_pixel) max_steps += 8;
	int num_steps = std::max(5, max_steps * quality / 100);

	best_vector.x = best_vector.y = 0;
	best_vector.error = std::numeric_limits<long>::max();

	int cur_ind = i * BLOCK_SIZE + j;
	int up_index = (i - 1) * BLOCK_SIZE + j;
	int upleft_index = (i - 1) * BLOCK_SIZE + j - 1;
	int upright_index = (i - 1) * BLOCK_SIZE + j + 1;
	int left_index = i * BLOCK_SIZE + j - 1;

	// try motion vector from the previous frame
	auto prev = prev_map.at(prev_mv_map[cur_ind].shift_dir) +
		vert_offset + hor_offset;
	best_vector.x = prev_mv_map[cur_ind].x;
	best_vector.y = prev_mv_map[cur_ind].y;
	auto comp = prev + best_vector.y * width_ext + best_vector.x;
	best_vector.error = GetErrorSAD_16x16(cur, comp, width_ext);
	if (best_vector.error < error_threshold) {
		StoreBestMVInMap(best_vector, cur_ind);
		return;
	}
	if (--num_steps <= 0) {
		StoreBestMVInMap(best_vector, cur_ind);
		return;
	}

	// try MV of up block of the current frame
	if (i > 0 &&
		curr_mv_map[up_index].error != std::numeric_limits<long>::max()) {
		prev = prev_map.at(curr_mv_map[up_index].shift_dir) +
			vert_offset + hor_offset;
		int tmp_x = curr_mv_map[up_index].x;
		int tmp_y = curr_mv_map[up_index].y;
		CheckAndUpdateBestMV(best_vector, curr_mv_map[up_index].shift_dir,
			prev, cur, tmp_x, tmp_y);
	}
	if (best_vector.error < error_threshold || (--num_steps <= 0)) {
		StoreBestMVInMap(best_vector, cur_ind);
		return;
	}

	// try MV of upleft block
	if (i > 0 && j > 0 &&
		curr_mv_map[upleft_index].error != std::numeric_limits<long>::max()) {
		prev = prev_map.at(curr_mv_map[upleft_index].shift_dir) +
			vert_offset + hor_offset;
		int tmp_x = curr_mv_map[upleft_index].x;
		int tmp_y = curr_mv_map[upleft_index].y;
		CheckAndUpdateBestMV(best_vector, curr_mv_map[upleft_index].shift_dir,
			prev, cur, tmp_x, tmp_y);
	}
	if (best_vector.error < error_threshold || (--num_steps <= 0)) {
		StoreBestMVInMap(best_vector, cur_ind);
		return;
	}

	// try MV of upright block
	if (i > 0 && j < (num_blocks_hor - 1) &&
		curr_mv_map[upright_index].error != std::numeric_limits<long>::max()) {
		prev = prev_map.at(curr_mv_map[upright_index].shift_dir) +
			vert_offset + hor_offset;
		int tmp_x = curr_mv_map[upright_index].x;
		int tmp_y = curr_mv_map[upright_index].y;
		CheckAndUpdateBestMV(best_vector, curr_mv_map[upright_index].shift_dir,
			prev, cur, tmp_x, tmp_y);
	}
	if (best_vector.error < error_threshold || (--num_steps <= 0)) {
		StoreBestMVInMap(best_vector, cur_ind);
		return;
	}

	// try MV of left block
	if (j > 0 &&
		curr_mv_map[left_index].error != std::numeric_limits<long>::max()) {
		prev = prev_map.at(curr_mv_map[left_index].shift_dir) +
			vert_offset + hor_offset;
		int tmp_x = curr_mv_map[left_index].x;
		int tmp_y = curr_mv_map[left_index].y;
		CheckAndUpdateBestMV(best_vector, curr_mv_map[left_index].shift_dir,
			prev, cur, tmp_x, tmp_y);
	}
	if (best_vector.error < error_threshold || (--num_steps <= 0)) {
		StoreBestMVInMap(best_vector, cur_ind);
		return;
	}

	// gradient descent search
	int cur_x = best_vector.x, cur_y = best_vector.y;
	best_vector.shift_dir = ShiftDir::NONE;
	prev = prev_map.at(best_vector.shift_dir) +
		vert_offset + hor_offset;
	comp = prev + best_vector.y * width_ext + best_vector.x;
	best_vector.error = GetErrorSAD_16x16(cur, comp, width_ext);
	if (best_vector.error < error_threshold || (--num_steps <= 0)) {
		StoreBestMVInMap(best_vector, cur_ind);
		return;
	}

	while (step_size > 0) {
		// check bottom left corner
		int tmp_x = cur_x - step_size;
		int tmp_y = cur_y - step_size;
		if (tmp_x >= -BORDER && tmp_y >= -BORDER) {
			CheckAndUpdateBestMV(best_vector, best_vector.shift_dir,
				prev, cur, tmp_x, tmp_y);
		}
		if (best_vector.error < error_threshold) break;
		if (--num_steps <= 0) {
			StoreBestMVInMap(best_vector, cur_ind);
			return;
		}

		// check top left corner
		tmp_x = cur_x - step_size;
		tmp_y = cur_y + step_size;
		if (tmp_x >= -BORDER && tmp_y <= BORDER) {
			CheckAndUpdateBestMV(best_vector, best_vector.shift_dir,
				prev, cur, tmp_x, tmp_y);
		}
		if (best_vector.error < error_threshold) break;
		if (--num_steps <= 0) {
			StoreBestMVInMap(best_vector, cur_ind);
			return;
		}

		// check top right corner
		tmp_x = cur_x + step_size;
		tmp_y = cur_y + step_size;
		if (tmp_x <= BORDER && tmp_y <= BORDER) {
			CheckAndUpdateBestMV(best_vector, best_vector.shift_dir, prev, cur, tmp_x, tmp_y);
		}
		if (best_vector.error < error_threshold) break;
		if (--num_steps <= 0) {
			StoreBestMVInMap(best_vector, cur_ind);
			return;
		}

		// check bottom right corner
		tmp_x = cur_x + step_size;
		tmp_y = cur_y - step_size;
		if (tmp_x <= BORDER && tmp_y >= -BORDER) {
			CheckAndUpdateBestMV(best_vector, best_vector.shift_dir, prev, cur, tmp_x, tmp_y);
		}
		if (best_vector.error < error_threshold) break;

		// update current block position
		if (cur_x == best_vector.x && cur_y == best_vector.y) {
			step_size /= 2;
		}
		else {
			cur_x = best_vector.x;
			cur_y = best_vector.y;
		}
	}

	// try to find in half pixel distance
	cur_x = best_vector.x;
	cur_y = best_vector.y;
	if (use_half_pixel) {
		// left by half pixel
		prev = prev_map.at(ShiftDir::LEFT) + vert_offset + hor_offset;
		int tmp_x = cur_x;
		int tmp_y = cur_y;
		CheckAndUpdateBestMV(best_vector, ShiftDir::LEFT,
			prev, cur, tmp_x, tmp_y);
		if (best_vector.error < error_threshold || (--num_steps <= 0)) {
			StoreBestMVInMap(best_vector, cur_ind);
			return;
		}

		// right by half pixel
		prev = prev_map.at(ShiftDir::LEFT) + vert_offset + hor_offset;
		tmp_x = cur_x + 1;
		tmp_y = cur_y;
		if (tmp_x <= BORDER) {
			CheckAndUpdateBestMV(best_vector, ShiftDir::LEFT,
				prev, cur, tmp_x, tmp_y);
			if (best_vector.error < error_threshold || (--num_steps <= 0)) {
				StoreBestMVInMap(best_vector, cur_ind);
				return;
			}
		}

		// up by half pixel
		prev = prev_map.at(ShiftDir::UP) + vert_offset + hor_offset;
		tmp_x = cur_x;
		tmp_y = cur_y;
		CheckAndUpdateBestMV(best_vector, ShiftDir::UP,
			prev, cur, tmp_x, tmp_y);
		if (best_vector.error < error_threshold || (--num_steps <= 0)) {
			StoreBestMVInMap(best_vector, cur_ind);
			return;
		}

		// down by half pixel
		prev = prev_map.at(ShiftDir::UP) + vert_offset + hor_offset;
		tmp_x = cur_x;
		tmp_y = cur_y + 1;
		if (tmp_y <= BORDER) {
			CheckAndUpdateBestMV(best_vector, ShiftDir::UP,
				prev, cur, tmp_x, tmp_y);
			if (best_vector.error < error_threshold || (--num_steps <= 0)) {
				StoreBestMVInMap(best_vector, cur_ind);
				return;
			}
		}

		// up left by half pixel
		prev = prev_map.at(ShiftDir::UPLEFT) + vert_offset + hor_offset;
		tmp_x = cur_x;
		tmp_y = cur_y;
		CheckAndUpdateBestMV(best_vector, ShiftDir::UPLEFT,
			prev, cur, tmp_x, tmp_y);
		if (best_vector.error < error_threshold || (--num_steps <= 0)) {
			StoreBestMVInMap(best_vector, cur_ind);
			return;
		}

		// down right by half pixel
		prev = prev_map.at(ShiftDir::UPLEFT) + vert_offset + hor_offset;
		tmp_x = cur_x + 1;
		tmp_y = cur_y + 1;
		if (tmp_x <= BORDER && tmp_y <= BORDER) {
			CheckAndUpdateBestMV(best_vector, ShiftDir::UPLEFT,
				prev, cur, tmp_x, tmp_y);
			if (best_vector.error < error_threshold || (--num_steps <= 0)) {
				StoreBestMVInMap(best_vector, cur_ind);
				return;
			}
		}

	}

	StoreBestMVInMap(best_vector, cur_ind);
}

void MotionEstimator::OrthogonalSearch(MV &best_vector, std::unordered_map<ShiftDir,
	const uint8_t*> const & prev_map, const uint8_t* cur,
	int hor_offset, int vert_offset) {
	for (const auto& prev_pair : prev_map) {
		const auto prev = prev_pair.second + vert_offset + hor_offset;
		int template_size = BORDER / 4;

		// compute log2_template_size
		int log2_template_size = 0;
		int tmp = template_size + 1;
		while (tmp >>= 1) ++log2_template_size;

		int max_num_steps = log2_template_size * 4;
		int num_steps = quality * max_num_steps / 100;

		int cur_x = 0, cur_y = 0;
		while (template_size > 0) {
			// 1. check horizontally oriented
			int tmp_x = cur_x - template_size;
			int tmp_y = cur_y;
			if (tmp_x >= -BORDER) {
				auto const comp = prev + tmp_y * width_ext + tmp_x;
				long tmp_error = GetErrorSAD_16x16(cur, comp, width_ext);
				if (tmp_error < best_vector.error) {
					best_vector.error = tmp_error;
					best_vector.x = tmp_x;
					best_vector.y = tmp_y;
					best_vector.shift_dir = prev_pair.first;
				}
				if (--num_steps <= 0) break;
			}

			tmp_x = cur_x + template_size;
			tmp_y = cur_y;
			if (tmp_x <= BORDER) {
				auto const comp = prev + tmp_y * width_ext + tmp_x;
				long tmp_error = GetErrorSAD_16x16(cur, comp, width_ext);
				if (tmp_error < best_vector.error) {
					best_vector.error = tmp_error;
					best_vector.x = tmp_x;
					best_vector.y = tmp_y;
					best_vector.shift_dir = prev_pair.first;
				}
				if (--num_steps <= 0) break;
			}

			cur_x = best_vector.x;
			cur_y = best_vector.y;

			// 2. check vertically oriented
			tmp_x = cur_x;
			tmp_y = cur_y - template_size;
			if (tmp_y >= -BORDER) {
				auto const comp = prev + tmp_y * width_ext + tmp_x;
				long tmp_error = GetErrorSAD_16x16(cur, comp, width_ext);
				if (tmp_error < best_vector.error) {
					best_vector.error = tmp_error;
					best_vector.x = tmp_x;
					best_vector.y = tmp_y;
					best_vector.shift_dir = prev_pair.first;
				}
				if (--num_steps <= 0) break;
			}

			tmp_x = cur_x;
			tmp_y = cur_y + template_size;
			if (tmp_y <= BORDER) {
				auto const comp = prev + tmp_y * width_ext + tmp_x;
				long tmp_error = GetErrorSAD_16x16(cur, comp, width_ext);
				if (tmp_error < best_vector.error) {
					best_vector.error = tmp_error;
					best_vector.x = tmp_x;
					best_vector.y = tmp_y;
					best_vector.shift_dir = prev_pair.first;
				}
				if (--num_steps <= 0) break;
			}

			cur_x = best_vector.x;
			cur_y = best_vector.y;
			template_size /= 2;
		}
	}

}

void MotionEstimator::BruteForceSearch(MV &best_vector, std::unordered_map<ShiftDir,
	const uint8_t*> const & prev_map, const uint8_t* cur,
	int hor_offset, int vert_offset) {
	for (const auto& prev_pair : prev_map) {
		const auto prev = prev_pair.second + vert_offset + hor_offset;

		for (int y = -BORDER; y <= BORDER; ++y) {
			for (int x = -BORDER; x <= BORDER; ++x) {
				const auto comp = prev + y * width_ext + x;
				const auto error = GetErrorSAD_16x16(cur, comp, width_ext);

				if (error < best_vector.error) {
					best_vector.x = x;
					best_vector.y = y;
					best_vector.shift_dir = prev_pair.first;
					best_vector.error = error;
				}
			}
		}
	}
}

void MotionEstimator::Estimate(const uint8_t* cur_Y,
                               const uint8_t* prev_Y,
                               const uint8_t* prev_Y_up,
                               const uint8_t* prev_Y_left,
                               const uint8_t* prev_Y_upleft,
                               MV* mvectors) {
	std::unordered_map<ShiftDir, const uint8_t*> prev_map {
		{ ShiftDir::NONE, prev_Y }
	};

	if (use_half_pixel) {
		prev_map.emplace(ShiftDir::UP, prev_Y_up);
		prev_map.emplace(ShiftDir::LEFT, prev_Y_left);
		prev_map.emplace(ShiftDir::UPLEFT, prev_Y_upleft);
	}

	// find a motion vector for each block of the current frame
	for (int i = 0; i < num_blocks_vert; ++i) {
		for (int j = 0; j < num_blocks_hor; ++j) {
			const auto block_id = i * num_blocks_hor + j;
			const auto hor_offset = j * BLOCK_SIZE;
			const auto vert_offset = first_row_offset + i * BLOCK_SIZE * width_ext;
			const auto cur = cur_Y + vert_offset + hor_offset;

			MV best_vector;
			best_vector.error = std::numeric_limits<long>::max();

			AdvancedSearch(best_vector, prev_map, cur, hor_offset, vert_offset, i, j);

#if 0			
			// Split into four subvectors if the error is too large
			if (best_vector.error > 1000) {
				best_vector.Split();

				for (int h = 0; h < 4; ++h) {
					auto& subvector = best_vector.SubVector(h);
					subvector.error = std::numeric_limits<long>::max();

					const auto hor_offset = j * BLOCK_SIZE + ((h & 1) ? BLOCK_SIZE / 2 : 0);
					const auto vert_offset = first_row_offset + (i * BLOCK_SIZE + ((h > 1) ? BLOCK_SIZE / 2 : 0)) * width_ext;
					const auto cur = cur_Y + vert_offset + hor_offset;

					for (const auto& prev_pair : prev_map) {
						const auto prev = prev_pair.second + vert_offset + hor_offset;

						for (int y = -BORDER; y <= BORDER; ++y) {
							for (int x = -BORDER; x <= BORDER; ++x) {
								const auto comp = prev + y * width_ext + x;
								const auto error = GetErrorSAD_8x8(cur, comp, width_ext);

								if (error < subvector.error) {
									subvector.x = x;
									subvector.y = y;
									subvector.shift_dir = prev_pair.first;
									subvector.error = error;
								}
							}
						}
					}
				}

				if (best_vector.SubVector(0).error
				    + best_vector.SubVector(1).error
				    + best_vector.SubVector(2).error
				    + best_vector.SubVector(3).error > best_vector.error * 0.7)
					best_vector.Unsplit();
			}
#endif

			mvectors[block_id] = best_vector;
		}
	}

	// save MV map of the current frame to previous frame map
	prev_mv_map = std::move(curr_mv_map);
	
	//reinitialize current frame for the next frame
	curr_mv_map.resize(num_blocks_hor * num_blocks_vert,
		MV(0, 0, ShiftDir::NONE, std::numeric_limits<long>::max()));
}

