/*
 * Copyright (c) 2020-2022, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

/** @file   nerf_loader.h
 *  @author Alex Evans, NVIDIA
 *  @brief  Ability to load nerf datasets.
 */

#pragma once

#include <neural-graphics-primitives/bounding_box.cuh>
#include <neural-graphics-primitives/common.h>

#include <filesystem/path.h>

#include <vector>

NGP_NAMESPACE_BEGIN

// how much to scale the scene by vs the original nerf dataset; we want to fit the thing in the unit cube
static constexpr float NERF_SCALE = 0.33f;

struct TrainingImageMetadata {
	// Camera intrinsics and additional data associated with a NeRF training image
	// the memory to back the pixels and rays is held by GPUMemory objects in the NerfDataset and copied here.
	const void* pixels = nullptr;
	EImageDataType image_data_type = EImageDataType::Half;

	const float* depth = nullptr;
	const float* depth_cov = nullptr;
	const Ray* rays = nullptr;

	Lens lens = {};
	Eigen::Vector2i resolution = Eigen::Vector2i::Zero(); // TODO: when calling set_camera_intrinsics, since this is 0, principal_point becomes inf.
	Eigen::Vector2f principal_point = Eigen::Vector2f::Constant(0.5f);
	Eigen::Vector2f focal_length = Eigen::Vector2f::Constant(1000.f);
	Eigen::Vector4f rolling_shutter = Eigen::Vector4f::Zero();
	Eigen::Vector3f light_dir = Eigen::Vector3f::Constant(0.f); // TODO: replace this with more generic float[] of task-specific metadata.
};

inline std::ostream& operator<<(std::ostream& os, const TrainingImageMetadata& metadata) {
	os << "TrainingImageMetadata:\n";
	os << "\t resolution: " << metadata.resolution << '\n';
	os << "\t principal_point: " << metadata.principal_point << '\n';
	os << "\t focal_length: " << metadata.focal_length << '\n';
	os << "\t rolling_shutter: " << metadata.rolling_shutter << '\n';
	os << "\t light_dir: " << metadata.light_dir << '\n';
	return os;
}

inline size_t image_type_size(EImageDataType type) {
	switch (type) {
		case EImageDataType::None: return 0;
		case EImageDataType::Byte: return 1;
		case EImageDataType::Half: return 2;
		case EImageDataType::Float: return 4;
		default: return 0;
	}
}

inline size_t depth_type_size(EDepthDataType type) {
	switch (type) {
		case EDepthDataType::UShort: return 2;
		case EDepthDataType::Float: return 4;
		default: return 0;
	}
}

struct NerfDataset {
	std::vector<tcnn::GPUMemory<Ray>> raymemory;
	std::vector<tcnn::GPUMemory<uint8_t>> pixelmemory;
	std::vector<tcnn::GPUMemory<float>> depthmemory;
	std::vector<tcnn::GPUMemory<float>> depthcovmemory;

	std::vector<TrainingImageMetadata> metadata;
	tcnn::GPUMemory<TrainingImageMetadata> metadata_gpu;

	void update_metadata(int first = 0, int last = -1);

	std::vector<TrainingXForm> xforms;
	std::vector<std::string> paths;
	tcnn::GPUMemory<float> sharpness_data;
	Eigen::Vector2i sharpness_resolution = {0, 0};
	tcnn::GPUMemory<float> envmap_data;

	BoundingBox render_aabb = {};
	Eigen::Matrix3f render_aabb_to_local = Eigen::Matrix3f::Identity();
	Eigen::Vector3f up = {0.0f, 1.0f, 0.0f};
	Eigen::Vector3f offset = {0.0f, 0.0f, 0.0f};
	size_t n_images = 0;
	Eigen::Vector2i envmap_resolution = {0, 0};
	float scale = 1.0f;
	int aabb_scale = 1;
	bool from_mitsuba = false;
	bool is_hdr = false;
	bool wants_importance_sampling = true;
	bool has_rays = false;

	uint32_t n_extra_learnable_dims = 0;
	bool has_light_dirs = false;

	uint32_t n_extra_dims() const {
		return (has_light_dirs ? 3u : 0u) + n_extra_learnable_dims;
	}

	void set_training_image(int frame_idx,
							const Eigen::Vector2i& image_resolution,
							const void* pixels,
							const void* depth_pixels,
							const void* depth_cov_pixels,
							float depth_scale,
							float depth_cov_scale,
							bool image_data_on_gpu,
							EImageDataType image_type,
							EDepthDataType depth_type,
							EDepthDataType depth_cov_type,
							float sharpen_amount = 0.f,
							bool white_transparent = false,
							bool black_transparent = false,
							uint32_t mask_color = 0,
							const Ray* rays = nullptr);

	Eigen::Vector3f nerf_direction_to_ngp(const Eigen::Vector3f& nerf_dir) {
		Eigen::Vector3f result = nerf_dir;
		if (from_mitsuba) {
			result *= -1;
		} else {
			result = Eigen::Vector3f(result.y(), result.z(), result.x());
		}
		return result;
	}

	Eigen::Matrix<float, 3, 4> nerf_matrix_to_ngp(const Eigen::Matrix<float, 3, 4>& nerf_matrix, bool scale_columns = false) const {
		Eigen::Matrix<float, 3, 4> result = nerf_matrix;
		result.col(0) *= scale_columns ? scale : 1.f;
		result.col(1) *= scale_columns ? -scale : -1.f;
		result.col(2) *= scale_columns ? -scale : -1.f;
		result.col(3) = result.col(3) * scale + offset;

		if (from_mitsuba) {
			result.col(0) *= -1;
			result.col(2) *= -1;
		} else {
			// Cycle axes xyz<-yzx
			Eigen::Vector4f tmp = result.row(0);
			result.row(0) = (Eigen::Vector4f)result.row(1);
			result.row(1) = (Eigen::Vector4f)result.row(2);
			result.row(2) = tmp;
		}

		return result;
	}

	Eigen::Matrix<float, 3, 4> ngp_matrix_to_nerf(const Eigen::Matrix<float, 3, 4>& ngp_matrix, bool scale_columns = false) const {
		Eigen::Matrix<float, 3, 4> result = ngp_matrix;
		if (from_mitsuba) {
			result.col(0) *= -1;
			result.col(2) *= -1;
		} else {
			// Cycle axes xyz->yzx
			Eigen::Vector4f tmp = result.row(0);
			result.row(0) = (Eigen::Vector4f)result.row(2);
			result.row(2) = (Eigen::Vector4f)result.row(1);
			result.row(1) = tmp;
		}
		result.col(0) *= scale_columns ?  1.f/scale :  1.f;
		result.col(1) *= scale_columns ? -1.f/scale : -1.f;
		result.col(2) *= scale_columns ? -1.f/scale : -1.f;
		result.col(3) = (result.col(3) - offset) / scale;
		return result;
	}

	Eigen::Vector3f ngp_position_to_nerf(Eigen::Vector3f pos) const {
		if (!from_mitsuba) {
			pos = Eigen::Vector3f(pos.z(), pos.x(), pos.y());
		}
		return (pos - offset) / scale;
	}

	Eigen::Vector3f nerf_position_to_ngp(const Eigen::Vector3f &pos) const {
		Eigen::Vector3f rv = pos * scale + offset;
		return from_mitsuba ? rv : Eigen::Vector3f(rv.y(), rv.z(), rv.x());
	}

	void nerf_ray_to_ngp(Ray& ray, bool scale_direction = false) {
		ray.o = ray.o * scale + offset;
		if (scale_direction)
			ray.d *= scale;

		float tmp = ray.o[0];
		ray.o[0] = ray.o[1];
		ray.o[1] = ray.o[2];
		ray.o[2] = tmp;

		tmp = ray.d[0];
		ray.d[0] = ray.d[1];
		ray.d[1] = ray.d[2];
		ray.d[2] = tmp;
	}

};


NerfDataset load_nerf(const std::vector<filesystem::path>& jsonpaths, float sharpen_amount = 0.f);
NerfDataset create_empty_nerf_dataset(size_t n_images, float nerf_scale, Eigen::Vector3f nerf_offset, int aabb_scale, BoundingBox render_aabb, bool is_hdr = false);

inline std::ostream& operator<<(std::ostream& os, const NerfDataset& dataset) {
	os << "Nerf Dataset:\n";
	os << "Ray memory: " << dataset.raymemory.size() << '\n';
	os << "Pixel memory: " << dataset.pixelmemory.size() << '\n';
	os << "Depth memory: " << dataset.depthmemory.size() << '\n';
	os << "Depth Cov memory: " << dataset.depthcovmemory.size() << '\n';
	os << "Metadata: " << dataset.metadata.size() << '\n';
	if (!dataset.metadata.empty()) os << "\t Last Metadata: " << dataset.metadata.back() << '\n';
	//tcnn::GPUMemory<TrainingImageMetadata> metadata_gpu << '\n';
	os << "Xforms: " << dataset.xforms.size() << '\n';
	if (!dataset.xforms.empty()) os << "\t Last Xform: " << dataset.xforms.back().start << '\n';
	os << "Paths: " << dataset.paths.size() << '\n';
	//tcnn::GPUMemory<float> sharpness_data << '\n';
	os << "sharpness_resolution: " << dataset.sharpness_resolution << '\n';
	//os << tcnn::GPUMemory<float> envmap_data << '\n';
	os << "Bounding Box: " << dataset.render_aabb << '\n';
	os << "render_aabb_to_local: " << dataset.render_aabb_to_local << '\n';
	os << "up: " << dataset.up << '\n';
	os << "offset: " << dataset.offset << '\n';
	os << "n_images: " << dataset.n_images << '\n';
	os << "envmap_resolution: " << dataset.envmap_resolution << '\n';
	os << "scale: " << dataset.scale << '\n';
	os << "aabb_scale: " << dataset.aabb_scale << '\n';
	os << "from_mitsuba: " << dataset.from_mitsuba << '\n';
	os << "is_hdr: " << dataset.is_hdr << '\n';
	os << "wants_importance_sampling: " << dataset.wants_importance_sampling << '\n';
	os << "has_rays: " << dataset.has_rays << '\n';
	os << "n_extra_learnable_dims: " << dataset.n_extra_learnable_dims << '\n';
	os << "has_light_dirs: " << dataset.has_light_dirs << '\n';
	return os;
}

NGP_NAMESPACE_END
