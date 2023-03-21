import pdb
import math
import numpy as np

# export PYTHONPATH=$PYTHONPATH:/Users/alonrot/work/code_projects_WIP/ood_project/ood/predictions_module/build
from predictions_interface import Predictions



def main():

	dim_out = 3
	dim_in = 5
	Nomegas = 1000
	Nrollouts = 50


	# Eigen::MatrixXd phi_samples_all_dim;
	# phi_samples_all_dim = Eigen::MatrixXd::Random(dim_out,Nomegas);
	phi_samples_all_dim = np.random.randn(dim_out,Nomegas)
	W_samples_all_dim = np.random.randn(dim_out,Nomegas,dim_in)
	mean_beta_pred_all_dim = np.random.randn(dim_out,Nomegas)
	cov_beta_pred_chol_all_dim = np.random.randn(dim_out,Nomegas,Nomegas)
	noise_mat = np.random.randn(Nrollouts,Nomegas)

	predictions = Predictions(dim_in,dim_out,phi_samples_all_dim,W_samples_all_dim,mean_beta_pred_all_dim,cov_beta_pred_chol_all_dim,noise_mat)
	# Predictions predictions(dim_in,dim_out,phi_samples_all_dim,W_samples_all_dim,mean_beta_pred_all_dim,cov_beta_pred_chol_all_dim,noise_mat);

	X_in = np.random.randn(1,dim_in)
	harmonics_vec = predictions.get_features_one_channel(X_in,0)
	print(harmonics_vec)


	# Eigen::MatrixXd ft_sampled = predictions.get_ft_sampled_all_dims(X_in,predictions.noise_mat.row(0));
	# std::cout << "ft_sampled: " << ft_sampled.format(predictions.clean_format) << "\n";

	ft_sampled = predictions.get_ft_sampled_all_dims(X_in,noise_mat[0:1,:])
	print("ft_sampled:",ft_sampled)

	# Eigen::MatrixXd x0 = Eigen::MatrixXd::Random(1,dim_out);
	# size_t Nhor = 15;
	# Eigen::MatrixXd u_traj = Eigen::MatrixXd::Random(Nhor,dim_in-dim_out);
	# Eigen::MatrixXd xtraj_sampled = predictions.run_one_rollout_from_current_state(x0,u_traj,predictions.noise_mat.row(1));
	# std::cout << "xtraj_sampled: " << xtraj_sampled.format(predictions.clean_format) << "\n";

	# x0 = np.random.randn(1,dim_out)
	# Nhor = 15
	# u_traj = np.random.randn(Nhor,dim_in-dim_out)
	# xtraj_sampled = predictions.run_one_rollout_from_current_state(x0,u_traj,noise_mat[0:1,:]) # xtraj_sampled: [Nhor+1,dim_x]
	# print("xtraj_sampled.shape:",xtraj_sampled.shape)
	# print("xtraj_sampled:",xtraj_sampled)
	# xtraj_sampled_all_rollouts = predictions.run_all_rollouts_from_current_state(x0,u_traj) # xtraj_sampled_all_rollouts: [ [Nhor+1,dim_x], ..., [Nhor+1,dim_x] ] list of Nrollouts elements
	# print("xtraj_sampled_all_rollouts:",xtraj_sampled_all_rollouts)
	# pdb.set_trace()
	# print("xtraj_sampled_all_rollouts.shape:",xtraj_sampled_all_rollouts.shape)


	Nhor = 15
	Nsteps = 120
	x_traj_real = np.random.randn(Nsteps,dim_out)
	u_traj_real = np.random.randn(Nsteps,dim_in-dim_out)
	xtraj_sampled_all_rollouts_for_entire_trajectory = predictions.run_all_rollouts_for_entire_trajectory(x_traj_real,u_traj_real,Nhor) # [Nsteps-Nhor, Nrollouts, Nhor+1, dim_x]; [list,list,np.array,np.array]


if __name__ == "__main__":

	main()





