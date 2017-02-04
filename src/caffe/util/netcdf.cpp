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
	void netcdf_load_nd_dataset_helper(const int& file_id, const std::vector<string>& netcdf_variables_, std::vector<int>& dset_ids,const int& time_stride, 
	const int& min_dim, const int& max_dim, std::vector<size_t>& dims, nc_type& vtype, Blob<Dtype>* blob) {
    
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

	//this one transposes dims 0 and 1, important if the 1st dimension shouldbe batched as well.
	template <>
	void netcdf_load_nd_dataset<float>(const int& file_id, const std::vector<string>& netcdf_variables_,const int& time_stride, 
	const int& min_dim, const int& max_dim, Blob<float>* blob) {
		
		//query the data and get some dimensions
		unsigned int numvars=netcdf_variables_.size();
		std::vector<int> dset_ids(numvars);
		std::vector<size_t> dims;
		nc_type vtype;
		netcdf_load_nd_dataset_helper(file_id, netcdf_variables_, dset_ids, time_stride, min_dim, max_dim, dims, vtype, blob);
		
		size_t num_dims = dims.size();
                int  num_time_steps_in_file = dims[0];
                int xdim = dims[1];
                int ydim = dims[2];
		vector<int> blob_dims(dims.size()+1);
		int num_blob_time_steps = num_time_steps_in_file / time_stride;
                //first dimension is channel dimension
                blob_dims[0] = num_blob_time_steps;
                blob_dims[1] = numvars;
                blob_dims[2] = xdim;
                blob_dims[3] = ydim;
		blob->Reshape(blob_dims);
		//create start vector for Hyperslab-IO:
		std::vector<size_t> start(dims.size()), count(dims);
		unsigned long offset=1;
		//starts at dim1, because dim0 will be considered singleton
		for(unsigned int i=1; i<num_dims; i++){
			offset*=dims[i];
			start[i]=0;
		}
		count[0]=1;	
		//read the data
		if(vtype == NC_FLOAT){
			//direct read possible
			for(unsigned int time=0; time<num_blob_time_steps; time++){
				start[0]=time * time_stride;
				for(unsigned int var=0; var<numvars; var++){
					int status = nc_get_vara_float(file_id, dset_ids[var], start.data(), count.data(), &(blob->mutable_cpu_data()[offset*(var+numvars*time)]));
					check_var_status(status,netcdf_variables_[var]);
				}
			}
		}
		else{
			DLOG(FATAL) << "Unsupported datatype";
		}
	}


	template <>
	void netcdf_load_nd_dataset<double>(const int& file_id, const std::vector<string>& netcdf_variables_,const int& time_stride, 
	const int& min_dim, const int& max_dim, Blob<double>* blob) {
			DLOG(FATAL) << "Not implemented";
	}





}  // namespace caffe
#endif
