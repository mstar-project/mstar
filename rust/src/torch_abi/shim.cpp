// C ABI over torch's C++ API, for torch_abi/mod.rs: tensor facts and tensor
// construction with no Python call. Built by build.rs as its own shared library
// and loaded after torch is imported, because THPVariableClass is a data symbol
// that linking the extension itself to would make `import mstar_rust` need torch.
//
// Keep MstarFacts and MstarError in step with the #[repr(C)] twins in mod.rs;
// mstar_torch_abi_revision is the check that they are.
#include <Python.h>

#include <ATen/ATen.h>
#include <ATen/TensorSubclassLikeUtils.h>
#include <torch/csrc/Dtype.h>
#include <torch/csrc/autograd/python_variable.h>
#include <torch/csrc/api/include/torch/version.h>

#include <cstdio>
#include <vector>

#define EXPORT extern "C" __attribute__((visibility("default")))

namespace {

constexpr int kMaxDims = 16;
constexpr int kAbiRevision = 1;
enum : int32_t { kOk = 0, kError = 1, kOutOfMemory = 2 };

}  // namespace

extern "C" {

struct MstarFacts {
  void* data;
  int64_t nbytes;
  int32_t device_type;  // c10::DeviceType
  int32_t device_index;
  int32_t dtype;  // c10::ScalarType
  int32_t ndim;
  // Its bytes are the tensor's values: strided, on real storage, conj and neg
  // bits resolved, no subclass dispatch. Otherwise only numel-derived fields
  // and dims are meaningful and the caller takes the torch path.
  uint8_t plain;
  uint8_t contiguous;
  uint8_t pad[6];
  int64_t dims[kMaxDims];
};

struct MstarError {
  int32_t code;
  char message[508];
};

}  // extern "C"

namespace {

void set_error(MstarError* err, int32_t code, const char* what) {
  err->code = code;
  std::snprintf(err->message, sizeof(err->message), "%s", what);
}

// For the calls that may run without the GIL: the error goes back as data.
template <typename F>
auto guarded(MstarError* err, F&& f) -> decltype(f()) {
  err->code = kOk;
  try {
    return f();
  } catch (const c10::OutOfMemoryError& e) {
    set_error(err, kOutOfMemory, e.what());
  } catch (const std::exception& e) {
    set_error(err, kError, e.what());
  }
  return decltype(f())();
}

at::Tensor reclaim(void* raw) {
  return at::Tensor::wrap_tensor_impl(
      c10::intrusive_ptr<c10::TensorImpl>::reclaim(static_cast<c10::TensorImpl*>(raw)));
}

// For the calls made holding the GIL.
template <typename F>
PyObject* to_python(F&& f) {
  try {
    return THPVariable_Wrap(f());
  } catch (const std::exception& e) {
    PyErr_SetString(PyExc_RuntimeError, e.what());
    return nullptr;
  }
}

}  // namespace

EXPORT int32_t mstar_torch_abi_revision() {
  return kAbiRevision * 1000 + static_cast<int32_t>(sizeof(MstarFacts) + sizeof(MstarError));
}

EXPORT const char* mstar_torch_built_against() { return TORCH_VERSION; }

// -1: not a tensor, or the tensor has more dims than kMaxDims.
EXPORT int32_t mstar_facts(PyObject* o, MstarFacts* f) {
  if (!THPVariable_Check(o)) return -1;
  try {
    const at::Tensor& t = THPVariable_Unpack(o);
    f->ndim = static_cast<int32_t>(t.dim());
    if (f->ndim > kMaxDims) return -1;
    f->dtype = static_cast<int32_t>(t.scalar_type());
    const at::Device device = t.device();
    f->device_type = static_cast<int32_t>(device.type());
    f->device_index = device.index();
    f->plain = t.layout() == at::kStrided && t.has_storage() && !t.is_conj() && !t.is_neg() &&
               !t.is_nested() && !at::isTensorSubclassLike(t);
    if (!f->plain) {
      f->data = nullptr;
      f->nbytes = 0;
      f->contiguous = 0;
      return 0;
    }
    f->data = t.data_ptr();
    f->nbytes = t.numel() * t.element_size();
    f->contiguous = t.is_contiguous();
    const auto sizes = t.sizes();
    for (int32_t i = 0; i < f->ndim; ++i) f->dims[i] = sizes[i];
    return 0;
  } catch (const std::exception&) {
    return -1;
  }
}

// A c10::ScalarType, or -1 when `o` is not a torch.dtype.
EXPORT int32_t mstar_dtype_scalar(PyObject* o) {
  if (!THPDtype_Check(o)) return -1;
  return static_cast<int32_t>(reinterpret_cast<THPDtype*>(o)->scalar_type);
}

// The next three never touch Python, so the caller may have released the GIL.
//
// A new uninitialized tensor, as an owned TensorImpl*. It allocates under the
// calling thread's current stream, so a device block is tagged as torch's own
// factories would tag it.
EXPORT void* mstar_empty(int32_t ndim, const int64_t* dims, int32_t dtype, int32_t device_type,
                         int32_t device_index, int32_t pinned, void** data, int64_t* nbytes,
                         MstarError* err) {
  return guarded(err, [&]() -> void* {
    at::Tensor t = at::empty(at::IntArrayRef(dims, ndim),
                             at::TensorOptions()
                                 .dtype(static_cast<c10::ScalarType>(dtype))
                                 .device(c10::Device(static_cast<c10::DeviceType>(device_type),
                                                     static_cast<c10::DeviceIndex>(device_index)))
                                 .pinned_memory(pinned != 0));
    *data = t.data_ptr();
    *nbytes = t.numel() * t.element_size();
    return t.unsafeReleaseTensorImpl();
  });
}

EXPORT void mstar_drop(void* raw) {
  try {
    reclaim(raw);
  } catch (const std::exception&) {
  }
}

// `raw`'s rows along dim 0 as owned TensorImpl*s; `raw` stays owned by the caller.
EXPORT int32_t mstar_unbind(void* raw, void** out, int64_t n, MstarError* err) {
  return guarded(err, [&]() -> int32_t {
    at::Tensor t = reclaim(raw);
    // reclaim() above took ownership for the call; hand it back on every path.
    struct Release {
      at::Tensor& t;
      ~Release() { t.unsafeReleaseTensorImpl(); }
    } release{t};
    std::vector<at::Tensor> rows = t.unbind(0);
    if (static_cast<int64_t>(rows.size()) != n) throw std::runtime_error("unbind: wrong row count");
    for (int64_t i = 0; i < n; ++i) out[i] = rows[i].unsafeReleaseTensorImpl();
    return 1;
  });
}

// GIL required from here on.

// Consumes `raw`.
EXPORT PyObject* mstar_wrap(void* raw) {
  return to_python([&] { return reclaim(raw); });
}

EXPORT PyObject* mstar_narrow(PyObject* o, int64_t start, int64_t length) {
  return to_python([&] { return THPVariable_Unpack(o).narrow(0, start, length); });
}

// `dim` first, the rest in order, contiguous: the canonical sharded layout.
EXPORT PyObject* mstar_lead_contiguous(PyObject* o, int64_t dim) {
  return to_python([&] {
    const at::Tensor& t = THPVariable_Unpack(o);
    std::vector<int64_t> order{dim};
    for (int64_t i = 0; i < t.dim(); ++i) {
      if (i != dim) order.push_back(i);
    }
    return t.permute(order).contiguous();
  });
}

// The inverse: dim 0 back to position `dim`.
EXPORT PyObject* mstar_unlead_contiguous(PyObject* o, int64_t dim) {
  return to_python([&] {
    const at::Tensor& t = THPVariable_Unpack(o);
    std::vector<int64_t> order;
    for (int64_t i = 1; i <= dim; ++i) order.push_back(i);
    order.push_back(0);
    for (int64_t i = dim + 1; i < t.dim(); ++i) order.push_back(i);
    return t.permute(order).contiguous();
  });
}

EXPORT PyObject* mstar_cat(PyObject* const* tensors, int64_t n, int64_t dim) {
  return to_python([&] {
    std::vector<at::Tensor> parts;
    parts.reserve(n);
    for (int64_t i = 0; i < n; ++i) parts.push_back(THPVariable_Unpack(tensors[i]));
    return at::cat(parts, dim);
  });
}
