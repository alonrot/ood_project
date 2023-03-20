

#include <predictions.hpp>

#include <pybind11/pybind11.h>
#include <pybind11/eigen.h> // This header includes #include <Eigen/Core>
#include <pybind11/stl.h>



/* -------------------------------------------------------------------------------------------------------------- */
/* ------------------------------------------ Python Wrapper ---------------------------------------------------- */
/* -------------------------------------------------------------------------------------------------------------- */


namespace py = pybind11;

PYBIND11_MODULE(predictions_interface, m) {
    m.doc() = R"pbdoc(
          Go1 Robot Interface Python Bindings
          -----------------------
          .. currentmodule:: predictions_interface
          .. autosummary::
             :toctree: _generate
      )pbdoc";

    py::class_<Predictions>(m, "Predictions")
        .def(py::init<size_t , size_t , Eigen::MatrixXd , std::vector<Eigen::MatrixXd> , Eigen::MatrixXd , std::vector<Eigen::MatrixXd> , Eigen::MatrixXd >(),
        "Initialize the Articulated System.\n\n"
        "Do not call this method yourself. use World class to create an Articulated system.\n\n"
        "Args:\n"
        "    filename (str): path to the robot description file (URDF, etc).\n"
        "    resource_directory (str): path the resource directory. If empty, it will use the robot description folder.\n"
        "    joint_order (list[str]): specify the joint order, if we want it to be different from the URDF file.\n"
        "    options (ArticulatedSystemOption): options.",
        py::arg("dim_in"), py::arg("dim_out"), py::arg("phi_samples_all_dim"), py::arg("W_samples_all_dim"), py::arg("mean_beta_pred_all_dim"), py::arg("cov_beta_pred_chol_all_dim"), py::arg("noise_mat"))
        .def("run_all_rollouts_for_entire_trajectory", &Predictions::run_all_rollouts_for_entire_trajectory)
        .def("run_all_rollouts_from_current_state", &Predictions::run_all_rollouts_from_current_state)
        .def("run_one_rollout_from_current_state", &Predictions::run_one_rollout_from_current_state)
        .def("get_ft_sampled_all_dims", &Predictions::get_ft_sampled_all_dims)
        .def("update_features_mat", &Predictions::update_features_mat)
        .def("get_features_one_channel", &Predictions::get_features_one_channel);

    #ifdef VERSION_INFO
      m.attr("__version__") = VERSION_INFO;
    #else
      m.attr("__version__") = "dev";
    #endif

      m.attr("TEST") = py::int_(int(42));

}
