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
	void netcdf_load_nd_dataset_helper(const int& file_id, const std::vector<string>& netcdf_variables_, std::vector<int>& dset_ids, 
										const int& min_dim, const int& max_dim, std::vector<size_t>& dims, nc_type& vtype_, Blob<Dtype>* blob, const bool& transpose=false);

	template <typename Dtype>
	void netcdf_load_nd_dataset(const int& file_id, const std::vector<string>& netcdf_variables_, const int& min_dim, const int& max_dim, Blob<Dtype>* blob);
	
	template <typename Dtype>
	void netcdf_load_nd_dataset_transposed(const int& file_id, const std::vector<string>& netcdf_variables_, const int& min_dim, const int& max_dim, Blob<Dtype>* blob);

	//template <typename Dtype>
	//void netcdf_save_nd_dataset(
	//    const hid_t file_id, const string& dataset_name, const Blob<Dtype>& blob,
	//    bool write_diff = false);

	int netcdf_load_int(int loc_id, const string& variable_name);
	//void netcdf_save_int(hid_t loc_id, const string& variable_name, int i);
	string netcdf_load_string(int loc_id, const string& variable_name);
	//void netcdf_save_string(hid_t loc_id, const string& variable_name, const string& s);

	//int netcdf_get_num_links(hid_t loc_id);
	//string netcdf_get_name_by_idx(hid_t loc_id, int idx);

}  // namespace caffe

#endif   // CAFFE_UTIL_NETCDF_H_
#endif //USE_NETCDF
