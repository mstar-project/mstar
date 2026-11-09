#include <torch/extension.h>
at::Tensor vox_solve(at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor,
    at::Tensor, std::vector<at::Tensor>, std::vector<at::Tensor>,
    std::vector<at::Tensor>, at::Tensor, at::Tensor, double);
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("solve", &vox_solve); }
