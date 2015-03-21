
#include <string>
#include <cstring>
#include <assert.h>
#include "mex.h"
#include "cxxnet_wrapper.h"

#define MEX_ARGS int nlhs, mxArray **plhs, int nrhs, const mxArray **prhs

typedef unsigned long long uint64;
union Ptr {
  uint64 data;
  void *ptr;
};


static mxArray* SetHandle(void *handle) {
  union Ptr bridge;
  bridge.data = 0;
  bridge.ptr = handle;
  const mwSize dims[1] = {1};
  mxArray *mx_out = mxCreateNumericArray(1, dims, mxUINT64_CLASS, mxREAL);
  uint64 *up = reinterpret_cast<uint64*>(mxGetData(mx_out));
  *up = bridge.data;
  return mx_out;
}

static void *GetHandle(const mxArray *input) {
  union Ptr bridge;
  uint64 *up = reinterpret_cast<uint64*>(mxGetData(input));
  bridge.data = *up;
  return bridge.ptr;
}

static mxArray* Ctype2Mx4DT(const cxx_real_t *ptr, cxx_uint oshape[4], cxx_uint ostride) {
  const mwSize dims[4] = {oshape[0], oshape[1], oshape[2], oshape[3]};
  mxArray *mx_out = mxCreateNumericArray(4, dims, mxSINGLE_CLASS, mxREAL);
  cxx_real_t *mx_ptr = reinterpret_cast<cxx_real_t*>(mxGetData(mx_out));
  for (cxx_uint i = 0; i < oshape[0]; ++i) {
    for (cxx_uint j = 0; j < oshape[1]; ++j) {
      for (cxx_uint k = 0; k < oshape[2]; ++k) {
        memcpy(mx_ptr, ptr, oshape[3] * sizeof(cxx_real_t));
        ptr += ostride;
        mx_ptr += oshape[3];
      }
    }
  }
  return mx_out;
}

static mxArray* Ctype2Mx2DT(const cxx_real_t *ptr, cxx_uint oshape[2], cxx_uint ostride) {
  const mwSize dims[2] = {oshape[0], oshape[1]};
  mxArray *mx_out = mxCreateNumericArray(2, dims, mxSINGLE_CLASS, mxREAL);
  cxx_real_t *mx_ptr = reinterpret_cast<cxx_real_t*>(mxGetData(mx_out));
  for (cxx_uint i = 0; i < oshape[0]; ++i) {
    memcpy(mx_ptr, ptr, oshape[1] * sizeof(cxx_real_t));
    ptr += ostride;
    mx_ptr += oshape[3];
  }
  return mx_out;
}


static mxArray* Ctype2Mx1DT(const cxx_real_t *ptr, cxx_uint len) {
  const mwSize dims[1] = {len};
  mxArray *mx_out = mxCreateNumericArray(1, dims, mxSINGLE_CLASS, mxREAL);
  cxx_real_t *mx_ptr = reinterpret_cast<cxx_real_t*>(mxGetData(mx_out));
  memcpy(mx_ptr, ptr, len * sizeof(cxx_real_t));
  return mx_out;
}

static mxArray* MXCXNIOCreateFromConfig(const char *cfg) {
  void *handle = CXNIOCreateFromConfig(cfg);
  return SetHandle(handle);
}

static mxArray* MXCXNIONext(void *handle) {
  const mwSize dims[1] = {1};
  mxArray *mx_out = mxCreateNumericArray(1, dims, mxINT32_CLASS, mxREAL);
  int *mx_ptr = reinterpret_cast<int*>(mxGetData(mx_out));
  int res = CXNIONext(handle);
  memcpy(mx_ptr, &res, sizeof(int));
  return mx_out;
}

static void MXCXNIOBeforeFirst(void *handle) {
  CXNIOBeforeFirst(handle);
}

static mxArray* MXCXNIOGetData(void *handle) {
  cxx_uint oshape[4];
  cxx_uint ostride = 0;
  const cxx_real_t *res_ptr = CXNIOGetData(handle, oshape, &ostride);
  return Ctype2Mx4DT(res_ptr, oshape, ostride);
}

static mxArray* MXCXNIOGetLabel(void *handle) {
  cxx_uint oshape[2];
  cxx_uint ostride = 0;
  const cxx_real_t *res_ptr = CXNIOGetLabel(handle, oshape, &ostride);
  return Ctype2Mx2DT(res_ptr, oshape, ostride);
}

static void MXCXNIOFree(void *handle) {
  CXNIOFree(handle);
}

static mxArray* MXCXNNetCreate(const char *device, const char *cfg) {
  void *handle = CXNNetCreate(device, cfg);
  return SetHandle(handle);
}

static void MXCXNNetFree(void *handle) {
  CXNNetFree(handle);
}

static void MXCXNNetSetParam(void *handle, const char *name, const char *val) {
  CXNNetSetParam(handle, name, val);
}

static void MXCXNNetInitModel(void *handle) {
  CXNNetInitModel(handle);
}

static void MXCXNNetSaveModel(void *handle, const char *fname) {
  CXNNetSaveModel(handle, fname);
}

static void MXCXNNetLoadModel(void *handle, const char *fname) {
  CXNNetLoadModel(handle, fname);
}

static void MXCXNNetStartRound(void *handle, int round) {
  CXNNetStartRound(handle, round);
}

static void MXCXNNetSetWeight(void *handle, const mxArray *p_weight,
                              const char *layer_name,
                              const char *wtag) {
  cxx_uint size = mxGetElementSize(p_weight) / sizeof(cxx_real_t);
  cxx_real_t *ptr = reinterpret_cast<cxx_real_t*>(mxGetData(p_weight));
  CXNNetSetWeight(handle, ptr, size, layer_name, wtag);
}


static mxArray* MXCXNNetGetWeight(void *handle,
                                  const char *layer_name,
                                  const char *wtag) {
  cxx_uint wshape[4];
  cxx_uint odim = 0;
  const cxx_real_t *res_ptr = CXNNetGetWeight(handle, layer_name, wtag, wshape, &odim);
  if (odim == 0) return NULL;
  return Ctype2Mx4DT(res_ptr, wshape, wshape[3]);
}

static void MXCXNNetUpdateIter(void *handle, void *data_handle) {
  CXNNetUpdateIter(handle, data_handle);
}

static void MXCXNNetUpdateBatch(void *handle, const mxArray *p_data,
                                const cxx_uint dshape[4],
                                const mxArray *p_label,
                                const cxx_uint lshape[2]) {
 cxx_real_t *ptr_data = reinterpret_cast<cxx_real_t*>(mxGetData(p_data));
 cxx_real_t *ptr_label = reinterpret_cast<cxx_real_t*>(mxGetData(p_label));
 CXNNetUpdateBatch(handle, ptr_data, dshape, ptr_label, lshape);
}

static mxArray* MXCXNNetPredictBatch(void *handle, const mxArray *p_data,
                                     const cxx_uint dshape[4]) {
  cxx_real_t *ptr_data = reinterpret_cast<cxx_real_t*>(mxGetData(p_data));
  cxx_uint out_size = 0;
  const cxx_real_t *ptr_res = CXNNetPredictBatch(handle, ptr_data, dshape, &out_size);
  return Ctype2Mx1DT(ptr_res, out_size);
}

static mxArray* MXCXNNetPredictIter(void *handle, void *data_handle) {
  cxx_uint out_size = 0;
  const cxx_real_t *ptr_res = CXNNetPredictIter(handle, data_handle, &out_size);
  return Ctype2Mx1DT(ptr_res, out_size);
}


static mxArray* MXCXNNetExtractIter(void *handle, void *data_handle,
                                    const char *node_name) {
  cxx_uint oshape[4];
  const cxx_real_t *ptr_res = CXNNetExtractIter(handle, data_handle, node_name, oshape);
  return Ctype2Mx4DT(ptr_res, oshape, oshape[3]);
}

static mxArray* MXCXNNetExtractBatch(void *handle,
                                     const mxArray *p_data,
                                     const cxx_uint dshape[4],
                                     const char *node_name) {
  cxx_real_t *ptr_data = reinterpret_cast<cxx_real_t*>(mxGetData(p_data));
  cxx_uint oshape[4];
  const cxx_real_t *ptr_res = CXNNetExtractBatch(handle, ptr_data, dshape, node_name, oshape);
  return Ctype2Mx4DT(ptr_res, oshape, oshape[3]);
}


static mxArray* MXCXNNetEvaluate(void *handle, void *data_handle, const char *data_name) {
  const char *ret = CXNNetEvaluate(handle, data_handle, data_name);
  return mxCreateString(ret);
}

static void MEXCXNIOCreateFromConfig(MEX_ARGS) {
  char *conf = mxArrayToString(prhs[1]);
  plhs[0] = MXCXNIOCreateFromConfig(conf);
  mxFree(conf);
}

static void MEXCXNIONext(MEX_ARGS) {
  void *handle = GetHandle(prhs[1]);
  plhs[0] = MXCXNIONext(handle);
}

static void MEXCXNIOBeforeFirst(MEX_ARGS) {
  void *handle = GetHandle(prhs[1]);
  MXCXNIOBeforeFirst(handle);
}

static void MEXCXNIOGetData(MEX_ARGS) {
  void *handle = GetHandle(prhs[1]);
  plhs[0] = MXCXNIOGetData(handle);
}

static void MEXCXNIOGetLabel(MEX_ARGS) {
  void *handle = GetHandle(prhs[1]);
  plhs[0] = MXCXNIOGetLabel(handle);
}

static void MEXCXNIOFree(MEX_ARGS) {
  void *handle = GetHandle(prhs[1]);
  MXCXNIOFree(handle);
}

static void MEXCXNNetCreate(MEX_ARGS) {
  char *dev = mxArrayToString(prhs[1]);
  char *conf = mxArrayToString(prhs[2]);
  plhs[0] = MXCXNNetCreate(dev, conf);
  mxFree(dev);
  mxFree(conf);
}

static void MEXCXNNetFree(MEX_ARGS) {
  void *handle = GetHandle(prhs[1]);
  MXCXNNetFree(handle);
}

static void MEXCXNNetSetParam(MEX_ARGS) {
  void *handle = GetHandle(prhs[1]);
  char *key = mxArrayToString(prhs[2]);
  char *val = mxArrayToString(prhs[3]);
  MXCXNNetSetParam(handle, key, val);
  mxFree(key);
  mxFree(val);
}

static void MEXCXNNetInitModel(MEX_ARGS) {
  void *handle = GetHandle(prhs[1]);
  MXCXNNetInitModel(handle);
}

static void MEXCXNNetSaveModel(MEX_ARGS) {
  void *handle = GetHandle(prhs[1]);
  char *name = mxArrayToString(prhs[2]);
  MXCXNNetSaveModel(handle, name);
  mxFree(name);
}

static void MEXCXNNetLoadModel(MEX_ARGS) {
  void *handle = GetHandle(prhs[1]);
  char *name = mxArrayToString(prhs[2]);
  MXCXNNetLoadModel(handle, name);
  mxFree(name);
}

static void MEXCXNNetStartRound(MEX_ARGS) {
  void *handle = GetHandle(prhs[1]);
  int *ptr = reinterpret_cast<int*>(mxGetData(prhs[2]));
  MXCXNNetStartRound(handle, *ptr);
}

static void MEXCXNNetSetWeight(MEX_ARGS) {
  void *handle = GetHandle(prhs[1]);
  const mxArray *p_weight = prhs[2];
  char *layer_name = mxArrayToString(prhs[3]);
  char *wtag = mxArrayToString(prhs[4]);
  MXCXNNetSetWeight(handle, p_weight, layer_name, wtag);
  mxFree(layer_name);
  mxFree(wtag);
}

static void MEXCXNNetGetWeight(MEX_ARGS) {
  void *handle = GetHandle(prhs[1]);
  char *layer_name = mxArrayToString(prhs[2]);
  char *wtag = mxArrayToString(prhs[3]);
  plhs[0] = MXCXNNetGetWeight(handle, layer_name, wtag);
  mxFree(layer_name);
  mxFree(wtag);
}

static void MEXCXNNetUpdateIter(MEX_ARGS) {
  void *handle = GetHandle(prhs[1]);
  void *data_handle = GetHandle(prhs[2]);
  MXCXNNetUpdateIter(handle, data_handle);
}

static void MEXCXNNetUpdateBatch(MEX_ARGS) {
  void *handle = GetHandle(prhs[1]);
  const mxArray *p_data = prhs[2];
  const mxArray *p_label = prhs[3];
  cxx_uint dshape[4];
  cxx_uint lshape[2];
  assert(mxGetNumberOfDimensions(p_data) == 4);
  assert(mxGetNumberOfDimensions(p_label) == 2);
  const mwSize *d_size = mxGetDimensions(p_data);
  const mwSize *l_size = mxGetDimensions(p_label);
  dshape[0] = d_size[0]; dshape[1] = d_size[1];
  dshape[2] = d_size[2]; dshape[3] = d_size[3];
  lshape[0] = l_size[0]; lshape[1] = l_size[1];
  MXCXNNetUpdateBatch(handle, p_data, dshape, p_label, lshape);
}

static void MEXCXNNetPredictBatch(MEX_ARGS) {
  void *handle = GetHandle(prhs[1]);
  const mxArray *p_data = prhs[2];
  cxx_uint dshape[4];
  assert(mxGetNumberOfDimensions(p_data) == 4);
  const mwSize *d_size = mxGetDimensions(p_data);
  dshape[0] = d_size[0]; dshape[1] = d_size[1];
  dshape[2] = d_size[2]; dshape[3] = d_size[3];
  plhs[0] = MXCXNNetPredictBatch(handle, p_data, dshape);
}

static void MEXCXNNetPredictIter(MEX_ARGS) {
  void *handle = GetHandle(prhs[1]);
  void *data_handle = GetHandle(prhs[2]);
  plhs[0] = MXCXNNetPredictIter(handle, data_handle);
}

static void MEXCXNNetExtractBatch(MEX_ARGS) {
  void *handle = GetHandle(prhs[1]);
  const mxArray *p_data = prhs[2];
  char *node_name = mxArrayToString(prhs[3]);
  assert(mxGetNumberOfDimensions(p_data) == 4);
  cxx_uint dshape[4];
  const mwSize *d_size = mxGetDimensions(p_data);
  dshape[0] = d_size[0]; dshape[1] = d_size[1];
  dshape[2] = d_size[2]; dshape[3] = d_size[3];
  plhs[0] = MXCXNNetExtractBatch(handle, p_data, dshape, node_name);
  mxFree(node_name);
}

static void MEXCXNNetExtractIter(MEX_ARGS) {
  void *handle = GetHandle(prhs[1]);
  void *data_handle = GetHandle(prhs[2]);
  char *node_name = mxArrayToString(prhs[3]);
  plhs[0] = MXCXNNetExtractIter(handle, data_handle, node_name);
  mxFree(node_name);
}

static void MEXCXNNetEvaluate(MEX_ARGS) {
  void *handle = GetHandle(prhs[1]);
  void *data_handle = GetHandle(prhs[2]);
  char *data_name = mxArrayToString(prhs[3]);
  plhs[0] = MXCXNNetEvaluate(handle, data_handle, data_name);
  mxFree(data_name);
}


// MEX Function
//

struct handle_registry {
  std::string cmd;
  void (*func)(MEX_ARGS);
};


static handle_registry handles[] = {
  {"MEXCXNIOCreateFromConfig", MEXCXNIOCreateFromConfig},
  {"MEXCXNIONext", MEXCXNIONext},
  {"MEXCXNIOBeforeFirst", MEXCXNIOBeforeFirst},
  {"MEXCXNIOGetData", MEXCXNIOGetData},
  {"MEXCXNIOGetLabel", MEXCXNIOGetLabel},
  {"MEXCXNIOFree", MEXCXNIOFree},
  {"MEXCXNNetCreate", MEXCXNNetCreate},
  {"MEXCXNNetFree", MEXCXNNetFree},
  {"MEXCXNNetSetParam", MEXCXNNetSetParam},
  {"MEXCXNNetInitModel", MEXCXNNetInitModel},
  {"MEXCXNNetSaveModel", MEXCXNNetSaveModel},
  {"MEXCXNNetLoadModel", MEXCXNNetLoadModel},
  {"MEXCXNNetStartRound", MEXCXNNetStartRound},
  {"MEXCXNNetSetWeight", MEXCXNNetSetWeight},
  {"MEXCXNNetGetWeight", MEXCXNNetGetWeight},
  {"MEXCXNNetUpdateIter", MEXCXNNetUpdateIter},
  {"MEXCXNNetUpdateBatch", MEXCXNNetUpdateBatch},
  {"MEXCXNNetPredictBatch", MEXCXNNetPredictBatch},
  {"MEXCXNNetPredictIter", MEXCXNNetPredictIter},
  {"MEXCXNNetExtractBatch", MEXCXNNetExtractBatch},
  {"MEXCXNNetExtractIter", MEXCXNNetExtractIter},
  {"MEXCXNNetEvaluate", MEXCXNNetEvaluate},
  {"NULL", NULL},
};

void mexFunction(MEX_ARGS) {
  mexLock();  // Avoid clearing the mex file.
  if (nrhs == 0) {
    mexErrMsgTxt("No API command given");
    return;
  }
  char *cmd = mxArrayToString(prhs[0]);
  bool dispatched = false;
  for (int i = 0; handles[i].func != NULL; i++) {
    if (handles[i].cmd.compare(cmd) == 0) {
      handles[i].func(nlhs, plhs, nrhs, prhs);
      dispatched = true;
      break;
    }
  }
  if (!dispatched) {
    std::string err = "Unknown command '";
    err += cmd;
    err += "'";
    mexErrMsgTxt(err.c_str());
  }
  mxFree(cmd);
}
