#ifdef USE_NETCDF
#include "caffe/util/netcdf.hpp"

#include <string>
#include <vector>

namespace caffe {

	void netcdf_check_variable_helper(const int& file_id, const string& variable_name_, int& dset_id, const int& min_dim, const int& max_dim, std::vector<size_t>& dims, nc_type& vtype_){
		//look up variable
		int status;
		status=nc_inq_varid(file_id, variable_name_.c_str(), &dset_id);
		CHECK(status != NC_EBADID) << "Bad NetCDF File-ID specified";
		
		//CHECK() << "Failed to find NetCDF variable " << variable_name_;
		// Verify that the number of dimensions is in the accepted range.
		int ndims;
		status = nc_inq_varndims(file_id, dset_id, &ndims);
		if(status != 0){
			CHECK(status != NC_EBADID) << "Failed to get variable ndims for " << variable_name_;
			CHECK(status != NC_ENOTVAR) << "Invalid path " << variable_name_;
		}
		CHECK_GE(ndims, min_dim);
		CHECK_LE(ndims, max_dim);

		// Verify that the data format is what we expect: float or double.
		std::vector<int> dimids(ndims);
		status = nc_inq_vardimid(file_id, dset_id, dimids.data());

		//query unlimited dimensions
		//int ndims_unlimited;
		//std::vector<int> dimids_unlimited;
		//status = nc_inq_unlimdims(file_id, &ndims_unlimited, dimids_unlimited.data());
		//CHECK(status != NC_ENOTNC4) << "netCDF-4 operation on netCDF-3 file performed for variable " << variable_name_;
		//for(unsigned int i=0; i<dimids_unlimited.size(); i++){
		//	std::cout << i << " " << dimids_unlimited[i] << std::endl;
		//}
		//for(unsigned int i=0; i<dimids.size(); i++){
		//	std::cout << i << " " << dimids[i] << std::endl;
		//}

		//get size of dimensions
		dims.resize(ndims);
		for(unsigned int i=0; i<ndims; ++i){
			status = nc_inq_dimlen(file_id, dimids[i], &dims[i]);
			CHECK(status != NC_EBADDIM) << "Invalid dimension " << i << " for " << variable_name_;
		}

		status = nc_inq_vartype(file_id, dset_id, &vtype_);
		switch (vtype_) {
			case NC_FLOAT:
			LOG_FIRST_N(INFO, 1) << "Datatype class for variable " << variable_name_ << ": NC_FLOAT";
			break;
			case NC_INT:
			LOG_FIRST_N(INFO, 1) << "Datatype class for variable " << variable_name_ << ": NC_INT or NC_LONG";
			break;
			case NC_DOUBLE:
			LOG_FIRST_N(INFO, 1) << "Datatype class for variable " << variable_name_ << ": NC_DOUBLE";
			break;
			default:
			LOG(FATAL) << "Unsupported Datatype " << vtype_ << " for variable " << variable_name_;
		}
	}
  
	// Verifies format of data stored in NetCDF file and reshapes blob accordingly.
	template <typename Dtype>
	void netcdf_load_nd_dataset_helper(const int& file_id, const std::vector<string>& netcdf_variables_, std::vector<int>& dset_ids, 
	const int& min_dim, const int& max_dim, std::vector<size_t>& dims, nc_type& vtype, Blob<Dtype>* blob, const bool& transpose) {
    
		// Obtain all sizes for variable 0:
		string variable_name_=netcdf_variables_[0];
		netcdf_check_variable_helper(file_id, variable_name_, dset_ids[0], min_dim, max_dim, dims, vtype);
    
		//make sure that all other dimensions for the other variables fit as well:
		for(unsigned int i=1; i<netcdf_variables_.size(); i++){
			std::vector<size_t> tmpdims;
			nc_type tmpvtype;
			variable_name_=netcdf_variables_[i];
			netcdf_check_variable_helper(file_id, variable_name_, dset_ids[i], min_dim, max_dim, tmpdims, tmpvtype);
			CHECK_EQ(tmpdims.size(),dims.size()) << "Number of dimensions of variable " << netcdf_variables_[0] << " and " << variable_name_ << " do not agree!";
			for(unsigned int d=0; d<tmpdims.size(); d++){
				CHECK_EQ(tmpdims[d],dims[d]) << "Dimension " << d << " does not agree for " << netcdf_variables_[0] << " and " << variable_name_;
			}
			CHECK_EQ(vtype,tmpvtype) << "Datatypes of variable " << netcdf_variables_[0] << " and " << variable_name_ << " do not agree!";
		}
		
		//set blob dimensions and reshape
		vector<int> blob_dims(dims.size()+1);
		//first dimension is channel dimension
		if(!transpose){
			blob_dims[0]=static_cast<int>(netcdf_variables_.size());
			for (int i = 1; i <= dims.size(); ++i) {
				blob_dims[i] = dims[i-1];
			}
		}
		else{
			blob_dims[0]=dims[0];
			blob_dims[1]=static_cast<int>(netcdf_variables_.size());
			for (int i = 2; i <= dims.size(); ++i) {
				blob_dims[i] = dims[i-1];
			}
		}
		blob->Reshape(blob_dims);
	}
	
	inline void check_var_status(const int& status, const std::string& varname){
		if(status != NC_NOERR){
			if(status == NC_ENOTVAR) std::cerr << "Variable " << varname << " not found";
			else if(status == NC_EINVALCOORDS) std::cerr << "Index exceeds dimension bound for variable " << varname;
			else if(status == NC_EEDGE) std::cerr << "Start+size exceeds dimension bound for variable " << varname;
			else if(status == NC_ERANGE) std::cerr << "SOme values are out of range for variable " << varname;
			CHECK(status) << "Failed to read double variable " << varname;
		}
	}

	template <>
	void netcdf_load_nd_dataset<float>(const int& file_id, const std::vector<string>& netcdf_variables_, 
	const int& min_dim, const int& max_dim, Blob<float>* blob) {
		
		//query the data and get some dimensions
		std::vector<int> dset_ids(netcdf_variables_.size());
		std::vector<size_t> dims;
		nc_type vtype;
		netcdf_load_nd_dataset_helper(file_id, netcdf_variables_, dset_ids, min_dim, max_dim, dims, vtype, blob);
		
		//create start vector for Hyperslab-IO:
		std::vector<size_t> start(dims.size());
		unsigned long offset=1;
		for(unsigned int i=0; i<dims.size(); i++){
			offset*=dims[i];
			start[i]=0;
		}
		
		//read the data
		if(vtype == NC_FLOAT){
			//direct read possible
			for(unsigned int i=0; i<netcdf_variables_.size(); i++){
				int status = nc_get_vara_float(file_id, dset_ids[i], start.data(), dims.data(), &(blob->mutable_cpu_data()[i*offset]));
				check_var_status(status,netcdf_variables_[i]);
			}
		}
		else if(vtype == NC_DOUBLE){
			//conversion necessary
			double* buf=new double[offset];
			for(unsigned int i=0; i<netcdf_variables_.size(); i++){	
				int status = nc_get_vara_double(file_id, dset_ids[i], start.data(), dims.data(), buf);
				check_var_status(status,netcdf_variables_[i]);
#pragma omp parallel for
				for(unsigned int k=0; k<offset; k++){
					blob->mutable_cpu_data()[k+i*offset]=static_cast<float>(buf[k]);
				}
			}
			delete [] buf;
		}
		else if( (vtype == NC_INT) || (vtype == NC_LONG) ){
			//conversion necessary
			int* buf=new int[offset];
			for(unsigned int i=0; i<netcdf_variables_.size(); i++){
				int status = nc_get_vara_int(file_id, dset_ids[i], start.data(), dims.data(), buf);
				check_var_status(status,netcdf_variables_[i]);
#pragma omp parallel for
				for(unsigned int k=0; k<offset; k++){
					blob->mutable_cpu_data()[k+i*offset]=static_cast<float>(buf[k]);
				}
			}
			delete [] buf;
		}
		else{
			DLOG(FATAL) << "Unsupported datatype";
		}
	}
	
	//this one transposes dims 0 and 1, important if the 1st dimension shouldbe batched as well.
	template <>
	void netcdf_load_nd_dataset_transposed<float>(const int& file_id, const std::vector<string>& netcdf_variables_, 
	const int& min_dim, const int& max_dim, Blob<float>* blob) {
		
		//query the data and get some dimensions
		unsigned int numvars=netcdf_variables_.size();
		std::vector<int> dset_ids(numvars);
		std::vector<size_t> dims;
		nc_type vtype;
		netcdf_load_nd_dataset_helper(file_id, netcdf_variables_, dset_ids, min_dim, max_dim, dims, vtype, blob, true);
		
		//create start vector for Hyperslab-IO:
		std::vector<size_t> start(dims.size()), count(dims);
		unsigned long offset=1;
		//starts at dim1, because dim0 will be considered singleton
		for(unsigned int i=1; i<dims.size(); i++){
			offset*=dims[i];
			start[i]=0;
		}
		count[0]=1;
		
		//read the data
		if(vtype == NC_FLOAT){
			//direct read possible
			for(unsigned int d=0; d<dims[0]; d++){
				start[0]=d;
				for(unsigned int i=0; i<numvars; i++){
					int status = nc_get_vara_float(file_id, dset_ids[i], start.data(), count.data(), &(blob->mutable_cpu_data()[offset*(i+numvars*d)]));
					check_var_status(status,netcdf_variables_[i]);
				}
			}
		}
		else if(vtype == NC_DOUBLE){
			//conversion necessary
			double* buf=new double[offset];
			for(unsigned int d=0; d<dims[0]; d++){
				start[0]=d;
				for(unsigned int i=0; i<numvars; i++){	
					int status = nc_get_vara_double(file_id, dset_ids[i], start.data(), count.data(), buf);
					check_var_status(status,netcdf_variables_[i]);
#pragma omp parallel for
					for(unsigned int k=0; k<offset; k++){
						blob->mutable_cpu_data()[k+offset*(i+numvars*d)]=static_cast<float>(buf[k]);
					}
				}
			}
			delete [] buf;
		}
		else if( (vtype == NC_INT) || (vtype == NC_LONG) ){
			//conversion necessary
			int* buf=new int[offset];
			for(unsigned int d=0; d<dims[0]; d++){
				start[0]=d;
				for(unsigned int i=0; i<numvars; i++){
					int status = nc_get_vara_int(file_id, dset_ids[i], start.data(), count.data(), buf);
					check_var_status(status,netcdf_variables_[i]);
#pragma omp parallel for
					for(unsigned int k=0; k<offset; k++){
						blob->mutable_cpu_data()[k+offset*(i+numvars*d)]=static_cast<float>(buf[k]);
					}
				}
			}
			delete [] buf;
		}
		else{
			DLOG(FATAL) << "Unsupported datatype";
		}
	}

	template <>
	void netcdf_load_nd_dataset<double>(const int& file_id, const std::vector<string>& netcdf_variables_, 
	const int& min_dim, const int& max_dim, Blob<double>* blob) {
		//query the data and get some dimensions
		std::vector<int> dset_ids(netcdf_variables_.size());
		std::vector<size_t> dims;
		nc_type vtype;
		netcdf_load_nd_dataset_helper(file_id, netcdf_variables_, dset_ids, min_dim, max_dim, dims, vtype, blob);
		
		//create start vector for Hyperslab-IO:
		std::vector<size_t> start(dims.size());
		unsigned long offset=1;
		for(unsigned int i=0; i<dims.size(); i++){
			offset*=dims[i];
			start[i]=0;
		}
		
		//read the data
		if(vtype == NC_DOUBLE){
			for(unsigned int i=0; i<netcdf_variables_.size(); i++){	
				int status = nc_get_vara_double(file_id, dset_ids[i], start.data(), dims.data(), &(blob->mutable_cpu_data()[i*offset]));
				check_var_status(status,netcdf_variables_[i]);
			}
		}
		else if(vtype == NC_FLOAT){
			//conversion necessary
			float* buf=new float[offset];
			for(unsigned int i=0; i<netcdf_variables_.size(); i++){	
				int status = nc_get_vara_float(file_id, dset_ids[i], start.data(), dims.data(), buf);
				check_var_status(status,netcdf_variables_[i]);
#pragma omp parallel for
				for(unsigned int k=0; k<offset; k++){
					blob->mutable_cpu_data()[k+i*offset]=static_cast<double>(buf[k]);
				}
			}
			delete [] buf;
		}
		else if( (vtype == NC_INT) || (vtype == NC_LONG) ){
			//conversion necessary
			int* buf=new int[offset];
			for(unsigned int i=0; i<netcdf_variables_.size(); i++){
				int status = nc_get_vara_int(file_id, dset_ids[i], start.data(), dims.data(), buf);
				check_var_status(status,netcdf_variables_[i]);
#pragma omp parallel for
				for(unsigned int k=0; k<offset; k++){
					blob->mutable_cpu_data()[k+i*offset]=static_cast<double>(buf[k]);
				}
			}
			delete [] buf;
		}
		else{
			DLOG(FATAL) << "Unsupported datatype";
		}
	}


	template <>
	void netcdf_load_nd_dataset_transposed<double>(const int& file_id, const std::vector<string>& netcdf_variables_, 
	const int& min_dim, const int& max_dim, Blob<double>* blob) {
		//query the data and get some dimensions
		unsigned int numvars=netcdf_variables_.size();
		std::vector<int> dset_ids(numvars);
		std::vector<size_t> dims;
		nc_type vtype;
		netcdf_load_nd_dataset_helper(file_id, netcdf_variables_, dset_ids, min_dim, max_dim, dims, vtype, blob, true);
		
		//create start vector for Hyperslab-IO:
		std::vector<size_t> start(dims.size()), count(dims);
		unsigned long offset=1;
		//starts at dim1, because dim0 will be considered singleton
		for(unsigned int i=1; i<dims.size(); i++){
			offset*=dims[i];
			start[i]=0;
		}
		count[0]=1;
		
		//read the data
		if(vtype == NC_DOUBLE){
			for(unsigned int d=0; d<dims[0]; d++){
				start[0]=d;
				for(unsigned int i=0; i<numvars; i++){	
					int status = nc_get_vara_double(file_id, dset_ids[i], start.data(), count.data(), &(blob->mutable_cpu_data()[offset*(i+numvars*d)]));
					check_var_status(status,netcdf_variables_[i]);
				}
			}
		}
		else if(vtype == NC_FLOAT){
			//conversion necessary
			float* buf=new float[offset];
			for(unsigned int d=0; d<dims[0]; d++){
				start[0]=d;
				for(unsigned int i=0; i<numvars; i++){	
					int status = nc_get_vara_float(file_id, dset_ids[i], start.data(), count.data(), buf);
					check_var_status(status,netcdf_variables_[i]);
#pragma omp parallel for
					for(unsigned int k=0; k<offset; k++){
						blob->mutable_cpu_data()[k+offset*(i+numvars*d)]=static_cast<double>(buf[k]);
					}
				}
			}
			delete [] buf;
		}
		else if( (vtype == NC_INT) || (vtype == NC_LONG) ){
			//conversion necessary
			int* buf=new int[offset];
			for(unsigned int d=0; d<dims[0]; d++){
				start[0]=d;
				for(unsigned int i=0; i<numvars; i++){
					int status = nc_get_vara_int(file_id, dset_ids[i], start.data(), count.data(), buf);
					check_var_status(status,netcdf_variables_[i]);
#pragma omp parallel for
					for(unsigned int k=0; k<offset; k++){
						blob->mutable_cpu_data()[k+offset*(i+numvars*d)]=static_cast<double>(buf[k]);
					}
				}
			}
			delete [] buf;
		}
		else{
			DLOG(FATAL) << "Unsupported datatype";
		}
	}

	//template <>
	//void netcdf_save_nd_dataset<float>(
	//	const hid_t file_id, const string& dataset_name, const Blob<float>& blob,
	//bool write_diff) {
	//	int num_axes = blob.num_axes();
	//	hsize_t *dims = new hsize_t[num_axes];
	//	for (int i = 0; i < num_axes; ++i) {
	//		dims[i] = blob.shape(i);
	//	}
	//	const float* data;
	//	if (write_diff) {
	//		data = blob.cpu_diff();
	//	} else {
	//		data = blob.cpu_data();
	//	}
	//	herr_t status = H5LTmake_dataset_float(
	//		file_id, dataset_name.c_str(), num_axes, dims, data);
	//	CHECK_GE(status, 0) << "Failed to make float dataset " << dataset_name;
	//	delete[] dims;
	//}
	//
	//template <>
	//void netcdf_save_nd_dataset<double>(
	//	hid_t file_id, const string& dataset_name, const Blob<double>& blob,
	//bool write_diff) {
	//	int num_axes = blob.num_axes();
	//	hsize_t *dims = new hsize_t[num_axes];
	//	for (int i = 0; i < num_axes; ++i) {
	//		dims[i] = blob.shape(i);
	//	}
	//	const double* data;
	//	if (write_diff) {
	//		data = blob.cpu_diff();
	//	} else {
	//		data = blob.cpu_data();
	//	}
	//	herr_t status = H5LTmake_dataset_double(
	//		file_id, dataset_name.c_str(), num_axes, dims, data);
	//	CHECK_GE(status, 0) << "Failed to make double dataset " << dataset_name;
	//	delete[] dims;
	//}

	string netcdf_load_string(int loc_id, const string& variable_name_) {
		// Verify that the dataset exists.
		int dset_id;
		CHECK_GT(nc_inq_varid(loc_id, variable_name_.c_str(), &dset_id),0) << "Failed to find NetCDF variable " << variable_name_;
		
		// Get size of dataset
		char* buffer;
		int status = nc_get_var_string(loc_id, dset_id, &buffer);
		string val(buffer);
		delete [] buffer;
		return val;
	}

	//void netcdf_save_string(hid_t loc_id, const string& dataset_name,
	//const string& s) {
	//	herr_t status = \
	//		H5LTmake_dataset_string(loc_id, dataset_name.c_str(), s.c_str());
	//	CHECK_GE(status, 0)
	//		<< "Failed to save string dataset with name " << dataset_name;
	//}

	int netcdf_load_int(int loc_id, const string& variable_name_) {
		int dset_id;
		CHECK_GT(nc_inq_varid(loc_id, variable_name_.c_str(), &dset_id),0) << "Failed to find NetCDF variable " << variable_name_;
		
		int val;
		int status = nc_get_var_int(loc_id, dset_id, &val);
		CHECK_GT(status, 0) << "Failed to load int variable " << variable_name_;
		return val;
	}

	//void netcdf_save_int(hid_t loc_id, const string& dataset_name, int i) {
	//	hsize_t one = 1;
	//	herr_t status = \
	//		H5LTmake_dataset_int(loc_id, dataset_name.c_str(), 1, &one, &i);
	//	CHECK_GE(status, 0)
	//		<< "Failed to save int dataset with name " << dataset_name;
	//}

	//int netcdf_get_num_links(hid_t loc_id) {
	//	H5G_info_t info;
	//	herr_t status = H5Gget_info(loc_id, &info);
	//	CHECK_GE(status, 0) << "Error while counting NetCDF links.";
	//	return info.nlinks;
	//}

	//string netcdf_get_name_by_idx(hid_t loc_id, int idx) {
	//	ssize_t str_size = H5Lget_name_by_idx(
	//		loc_id, ".", H5_INDEX_NAME, H5_ITER_NATIVE, idx, NULL, 0, H5P_DEFAULT);
	//	CHECK_GE(str_size, 0) << "Error retrieving NetCDF variable at index " << idx;
	//	char *c_str = new char[str_size+1];
	//	ssize_t status = H5Lget_name_by_idx(
	//		loc_id, ".", H5_INDEX_NAME, H5_ITER_NATIVE, idx, c_str, str_size+1,
	//	H5P_DEFAULT);
	//	CHECK_GE(status, 0) << "Error retrieving NetCDF variable at index " << idx;
	//	string result(c_str);
	//	delete[] c_str;
	//	return result;
	//}

}  // namespace caffe
#endif
