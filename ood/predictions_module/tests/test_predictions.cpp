/************************************************************************
*
* @brief prediction module
*
* @author Alonso Marco
* Contact: amarco@berkeley.edu
************************************************************************/

#include <string>
// #include <pthread.h>
// #include <boost/thread.hpp>
// #include <boost/thread/mutex.hpp>
// #include <unitree_legged_msgs/HighCmd.h>
// #include <unitree_legged_msgs/HighState.h>

// #include <data_parser.hpp>

// #include "convert.h" // Needed for ToRos()

#include <predictions.hpp>


int main(int argc, char *argv[])
{

  size_t Nomegas = 750;
  size_t dim_in = 5;
  size_t dim_out = 3;
  size_t Nrollouts = 20;
  size_t Nhor = 15;

  Eigen::MatrixXd phi_samples_all_dim;
  phi_samples_all_dim = Eigen::MatrixXd::Random(dim_out,Nomegas);
  
  Eigen::MatrixXd mean_beta_pred_all_dim;
  mean_beta_pred_all_dim = Eigen::MatrixXd::Random(dim_out,Nomegas);

  std::vector<Eigen::MatrixXd> W_samples_all_dim;
  for(int dd=0;dd<dim_out;dd++){
    W_samples_all_dim.push_back(Eigen::MatrixXd::Random(Nomegas,dim_in));
  }
  
  std::vector<Eigen::MatrixXd> cov_beta_pred_chol_all_dim;
  for(int dd=0;dd<dim_out;dd++){
    cov_beta_pred_chol_all_dim.push_back(Eigen::MatrixXd::Random(Nomegas,Nomegas));
  }

  Eigen::MatrixXd noise_mat = Eigen::MatrixXd::Random(Nrollouts,Nomegas);

  Predictions predictions(dim_in,dim_out,phi_samples_all_dim,W_samples_all_dim,mean_beta_pred_all_dim,cov_beta_pred_chol_all_dim,noise_mat,Nrollouts,Nhor);

  Eigen::MatrixXd X_in = Eigen::MatrixXd::Random(1,dim_in);

  // Tests:
  // Eigen::MatrixXd harmonics_vec = predictions.get_features_one_channel(X_in,0);
  
  // predictions.update_features_mat(X_in);
  // for(int dd=0;dd<dim_out;dd++){
  //   std::cout << "predictions.phi_mat[dd]: " << predictions.phi_mat[dd].format(predictions.clean_format) << "\n";
  // }


  // Eigen::MatrixXd ft_sampled = predictions.get_ft_sampled_all_dims(X_in,predictions.noise_mat.row(0));
  // std::cout << "ft_sampled: " << ft_sampled.format(predictions.clean_format) << "\n";
  

  // Eigen::MatrixXd x0 = Eigen::MatrixXd::Random(1,dim_out);
  // size_t Nhor = 15;
  // Eigen::MatrixXd u_traj = Eigen::MatrixXd::Random(Nhor,dim_in-dim_out);
  // Eigen::MatrixXd xtraj_sampled = predictions.run_one_rollout_from_current_state(x0,u_traj,predictions.noise_mat.row(1));
  // std::cout << "xtraj_sampled: " << xtraj_sampled.format(predictions.clean_format) << "\n";

  // Eigen::MatrixXd x0 = Eigen::MatrixXd::Random(1,dim_out);
  // size_t Nhor = 15;
  // Eigen::MatrixXd u_traj = Eigen::MatrixXd::Random(Nhor,dim_in-dim_out);
  // std::vector<Eigen::MatrixXd> xtraj_sampled_all_rollouts = predictions.run_all_rollouts_from_current_state(x0,u_traj);

  // for(int ii=0;ii<xtraj_sampled_all_rollouts.size();++ii){
  //   std::cout << "xtraj_sampled_all_rollouts[ii]: " << xtraj_sampled_all_rollouts[ii].format(predictions.clean_format) << "\n";
  // }



  size_t Nsteps = 120;
  Eigen::MatrixXd x_traj_real = Eigen::MatrixXd::Random(Nsteps,dim_out);
  Eigen::MatrixXd u_traj_real = Eigen::MatrixXd::Random(Nsteps,dim_in-dim_out);
  std::vector<std::vector<Eigen::MatrixXd>> xtraj_sampled_all_rollouts_for_entire_trajectory = predictions.run_all_rollouts_for_entire_trajectory(x_traj_real,u_traj_real,Nhor); // [Nsteps, Nrollouts, Nhor, dim_out]

  // for(int tt=0;tt<xtraj_sampled_all_rollouts_for_entire_trajectory.size();++tt){
  //   for(int rr=0;rr<xtraj_sampled_all_rollouts_for_entire_trajectory[0].size();++rr){
  //     std::cout << "xtraj_sampled_all_rollouts_for_entire_trajectory[tt][rr]: " << xtraj_sampled_all_rollouts_for_entire_trajectory[tt][rr].format(predictions.clean_format) << "\n";
  //   }
  // }




  // std::cin.ignore();



    return 0;
}

