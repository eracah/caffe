#ifdef USE_NETCDF
#ifndef CAFFE_UTIL_NETCDF_H_
#define CAFFE_UTIL_NETCDF_H_

#include <string>
#include <vector>
#include <map>

#include <netcdf.h>

#include "caffe/blob.hpp"

namespace caffe {

	void netcdf_check_variable_helper(const int& file_id, const string& variable_name_, int& dset_id, 
										const int& min_dim, const int& max_dim, std::vector<size_t>& dims, nc_type& vtype_);

	inline void check_var_status(const int& status, const std::string& varname);

	template <typename Dtype>
	void netcdf_load_nd_dataset_helper(const int& file_id, const std::vector<string>& netcdf_variables_, std::vector<int>& dset_ids,const int& time_stride,
										const int& min_dim, const int& max_dim, std::vector<size_t>& dims, nc_type& vtype_, Blob<Dtype>* blob, const bool& transpose=false);

	
	template <typename Dtype>
	void netcdf_load_nd_dataset_transposed(const int& file_id, const std::vector<string>& netcdf_variables_,const int& time_stride, const int& min_dim, const int& max_dim, Blob<Dtype>* blob);

	int netcdf_load_int(int loc_id, const string& variable_name);
	string netcdf_load_string(int loc_id, const string& variable_name);



}  // namespace caffe

#endif   // CAFFE_UTIL_NETCDF_H_
#endif //USE_NETCDF
