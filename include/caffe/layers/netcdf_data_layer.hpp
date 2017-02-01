#ifdef USE_NETCDF
#ifndef CAFFE_NETCDF_DATA_LAYER_HPP_
#define CAFFE_NETCDF_DATA_LAYER_HPP_

#include <netcdf.h>

#include <string>
#include <vector>
#include <map>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/layers/base_data_layer.hpp"

namespace caffe {

	/**
	* @brief Provides data to the Net from NetCDF files.
	*
	* TODO(dox): thorough documentation for Forward and proto params.
	*/
	template <typename Dtype>
	class NetCDFDataLayer : public Layer<Dtype> {
	public:
		explicit NetCDFDataLayer(const LayerParameter& param)
			: Layer<Dtype>(param) {}
		virtual ~NetCDFDataLayer();
		virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top);
		// Data layers should be shared by multiple solvers in parallel
		virtual inline bool ShareInParallel() const { return true; }
		// Data layers have no bottoms, so reshaping is trivial.
		virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {}

		virtual inline const char* type() const { return "NetCDFData"; }
		virtual inline int ExactNumBottomBlobs() const { return 0; }
		virtual inline int MinTopBlobs() const { return 1; }

	protected:
		virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top);
		virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top);
		virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {}
		virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {}
		virtual void LoadNetCDFFileData(const char* filename);

		std::vector<std::string> netcdf_filenames_;
		std::map < string, std::vector<std::string> > netcdf_variables_;
		std::map < string, unsigned int > num_dimensions_, num_variables_;
		unsigned int num_files_;
		unsigned int current_file_;
		unsigned int current_row_;
		std::vector<shared_ptr<Blob<Dtype> > > netcdf_blobs_;
		bool first_dim_is_batched_;
		std::vector<unsigned int> data_permutation_;
		std::vector<unsigned int> file_permutation_;
	};

}  // namespace caffe

#endif  // CAFFE_NETCDF_DATA_LAYER_HPP_
#endif  // USE_NETCDF
