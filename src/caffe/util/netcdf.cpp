
#ifdef USE_NETCDF
#include "caffe/util/netcdf.hpp"
//#include "caffe/util/netcdf_error_handling.hpp"

#include <string>
#include <vector>

namespace caffe {

        //this one transposes dims 0 and 1, important if the 1st dimension shouldbe batched as well.
        template <>
        void netcdf_load_nd_dataset<float>(const int& file_id, 
                                        const std::vector<string>& netcdf_variables_,
                                        const int& time_stride,
                                        const int & xdim, 
                                        const int & ydim,
                                        const int & timedim,
                                        const int & crop_index,
					const int & crop_stride,
                                        Blob<float>* blob) {


                unsigned int numvars=netcdf_variables_.size();
                int  num_time_steps_in_file = timedim;
                int num_blob_time_steps = num_time_steps_in_file / time_stride;
                

                int tmp_blob_dims[4] = {num_blob_time_steps, numvars, xdim, ydim};
                std::vector<int> blob_dims(tmp_blob_dims, tmp_blob_dims + sizeof(tmp_blob_dims) / sizeof(int));
                blob->Reshape(blob_dims);

                //set all values to -99
                // for (int i =0; i < num_blob_time_steps * numvars * xdim * ydim; i++)
                //         blob->mutable_cpu_data()[i] =  -99.0;
                //         //CHECK_EQ(blob->cpu_data()[i], -99.0);

                // CHECK_EQ(blob->cpu_data()[0], -99);
                // CHECK_EQ(blob->cpu_data()[1000], -99);

                unsigned long offset =  xdim * ydim;
                size_t start[3] = {0, 0 , crop_index * crop_stride};
                size_t count[3] =  {1, xdim, ydim };
                //std::cerr <<  "y start: " << start[2] << " y count: " << count[2] << std::endl;
                int status;
                int var_id;
                int index;
                int cnt = 0;

                //read the data
                for(unsigned int time=0; time<num_blob_time_steps; time++){
                        
                        start[0] = time * time_stride;
                        
                        for(unsigned int var=0; var<numvars; var++){
                                
                                index = offset * (var + numvars * time);

                                status = nc_inq_varid (file_id, netcdf_variables_[var].c_str(), &var_id);
                                check_status(status,netcdf_variables_[var]);
                                
                                
                                status = nc_get_vara_float(file_id, 
                                                        var_id,
                                                        start, 
                                                        count, 
                                                        & (blob->mutable_cpu_data()[index]));
                                // cnt = 0;
                                // for (int i =0; i < offset; i++)
                                // {
                                //         if (blob->cpu_data()[index + i] == -99)
                                //                 std::cerr << "Whoa whoa " << index + i  << " was not filled "<< std::endl;
                                //         if (blob->cpu_data()[index + offset + i] != -99)
                                //                 cnt++;


                                // }

                                // std::cerr << "at time: " << time << " and var: " << var << " there were " << cnt << " extra variables filled" << std::endl;

                                check_status(status,netcdf_variables_[var]);       
                        }
                }
        }


        template <>
        void netcdf_load_nd_dataset<double>(const int& file_id,
                                        const std::vector<string>& netcdf_variables_,
                                        const int& time_stride,
                                        const int & xdim,
                                        const int & ydim,
                                        const int & timedim,
                                        const int & crop_index,
                                        const int & crop_stride,
                                        Blob<double>* blob){   
		DLOG(FATAL) << "Not implemented";
        }

        void check_status(const int& status, const std::string& name)
        {
                if (status != NC_NOERR       ){

                        if (status == NC_EBADID      )     std::cerr <<  "Not a netcdf id " << name << std::endl;
                        else if (status == NC_ENFILE      ) std::cerr <<  "Too many netcdfs open " << name << std::endl;
                        else if (status == NC_EEXIST      ) std::cerr <<  "netcdf file exists && NC_NOCLOBBER " << name << std::endl;
                        else if (status == NC_EINVAL      ) std::cerr <<  "Invalid Argument " << name << std::endl;
                        else if (status == NC_EPERM       ) std::cerr <<  "Write to read only " << name << std::endl;
                        else if (status == NC_ENOTINDEFINE) std::cerr <<  "Operation not allowed in data mode " << name << std::endl;
                        else if (status == NC_EINDEFINE   ) std::cerr <<  "Operation not allowed in define mode " << name << std::endl;
                        else if (status == NC_EINVALCOORDS) std::cerr <<  "Index exceeds dimension bound " << name << std::endl;
                        else if (status == NC_EMAXDIMS    ) std::cerr <<  "NC_MAX_DIMS exceeded " << name << std::endl;
                        else if (status == NC_ENAMEINUSE  ) std::cerr <<  "String match to name in use " << name << std::endl;
                        else if (status == NC_ENOTATT     ) std::cerr <<  "Attribute not found " << name << std::endl;
                        else if (status == NC_EMAXATTS    ) std::cerr <<  "NC_MAX_ATTRS exceeded " << name << std::endl;
                        else if (status == NC_EBADTYPE    ) std::cerr <<  "Not a netcdf data type " << name << std::endl;
                        else if (status == NC_EBADDIM     ) std::cerr <<  "Invalid dimension id or name " << name << std::endl;
                        else if (status == NC_EUNLIMPOS   ) std::cerr <<  "NC_UNLIMITED in the wrong index " << name << std::endl;
                        else if (status == NC_EMAXVARS    ) std::cerr <<  "NC_MAX_VARS exceeded " << name << std::endl;
                        else if (status == NC_ENOTVAR     ) std::cerr <<  "Variable not found " << name << std::endl;
                        else if (status == NC_EGLOBAL     ) std::cerr <<  "Action prohibited on NC_GLOBAL varid " << name << std::endl;
                        else if (status == NC_ENOTNC      ) std::cerr <<  "Not a netcdf file " << name << std::endl;
                        else if (status == NC_ESTS        ) std::cerr <<  "In Fortran, string too short " << name << std::endl;
                        else if (status == NC_EMAXNAME    ) std::cerr <<  "NC_MAX_NAME exceeded " << name << std::endl;
                        else if (status == NC_EUNLIMIT    ) std::cerr <<  "NC_UNLIMITED size already in use " << name << std::endl;
                        else if (status == NC_ENORECVARS  ) std::cerr <<  "nc_rec op when there are no record vars " << name << std::endl;
                        else if (status == NC_ECHAR       ) std::cerr <<  "Attempt to convert between text & numbers " << name << std::endl;
                        else if (status == NC_EEDGE       ) std::cerr <<  "Edge+start exceeds dimension bound " << name << std::endl;
                        else if (status == NC_ESTRIDE     ) std::cerr <<  "Illegal stride " << name << std::endl;
                        else if (status == NC_EBADNAME    ) std::cerr <<  "Attribute or variable name contains illegal characters " << name << std::endl;
                        else if (status == NC_ERANGE      ) std::cerr <<  "Math result not representable " << name << std::endl;
                        else if (status == NC_ENOMEM      ) std::cerr <<  "Memory allocation (malloc) failure " << name << std::endl;
                        else if (status == NC_EVARSIZE    ) std::cerr <<  "One or more variable sizes violate format constraints " << name << std::endl;
                        else if (status == NC_EDIMSIZE    ) std::cerr <<  "Invalid dimension size " << name << std::endl;
                        else if (status == NC_ETRUNC      ) std::cerr <<  "File likely truncated or possibly corrupted " << name << std::endl;
                        else if (status == NC_EHDFERR     ) std::cerr <<  "Error, HDF5-error while opening file " << name << std::endl;
                        else if (status == NC_ECANTREAD   ) std::cerr <<  "CANTREAD    " << name << std::endl;
                        else if (status == NC_ECANTWRITE  ) std::cerr <<  "CANTWRITE    " << name << std::endl;
                        else if (status == NC_ECANTCREATE ) std::cerr <<  "CANTCREATE  " << name << std::endl;
                        else if (status == NC_EFILEMETA   ) std::cerr <<  "FILEMETA   " << name << std::endl;
                        else if (status == NC_EDIMMETA    ) std::cerr <<  "Error in NetCDF-4 dimension data in  file" << name << std::endl;
                        else if (status == NC_EATTMETA    ) std::cerr <<  "ATTMETA   " << name << std::endl;
                        else if (status == NC_EVARMETA    ) std::cerr <<  "VARMETA  " << name << std::endl;
                        else if (status == NC_ENOCOMPOUND ) std::cerr <<  "NOCOMPOUND " << name << std::endl;
                        else if (status == NC_EATTEXISTS  ) std::cerr <<  "ATT Exists " << name << std::endl;
                        else if (status == NC_ENOTNC4     ) std::cerr <<  "Attempting netcdf-4 operation on netcdf-3 file. " << name << std::endl;
                        else if (status == NC_ESTRICTNC3  ) std::cerr <<  "Attempting netcdf-4 operation on strict nc3 netcdf-4 file. " << name << std::endl;
                        else if (status == NC_EBADGRPID   ) std::cerr <<  "Bad group id. Bad! " << name << std::endl;
                        // else if (status == NC_EBADTYPEID  ) std::cerr <<  "Bad type id. " << name << std::endl;
                        // else if (status == NC_EBADFIELDID ) std::cerr <<  "Bad field id. " << name << std::endl;
                        // else if (status == NC_EUNKNAME    ) std::cerr <<  "Unknown name " << name << std::endl;
                }
        }

}/// namespace caffe
#endif
