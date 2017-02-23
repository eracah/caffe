#ifdef USE_NETCDF
#ifndef CAFFE_UTIL_NETCDF_H_
#define CAFFE_UTIL_NETCDF_H_

#include <string>
#include <vector>
#include <map>

#include <netcdf.h>

#include "caffe/blob.hpp"

namespace caffe {


        inline void check_status(const int& status, const std::string& name);


        template <typename Dtype>
        void netcdf_load_nd_dataset(const int& file_id,
                                        const std::vector<string>& netcdf_variables_,
                                        const int& time_stride,
                                        const int & xdim,
                                        const int & ydim,
                                        const int & timedim,
                                        const int & crop_index,
                                        const int & crop_stride,
                                        Blob<Dtype>* blob);




}  // namespace caffe

#endif   // CAFFE_UTIL_NETCDF_H_
#endif //USE_NETCDF
