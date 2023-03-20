import pdb
import math
import numpy as np

# export PYTHONPATH=$PYTHONPATH:/Users/alonrot/work/code_projects_WIP/ood_project/ood/predictions_module/build


from predictions_interface import Predictions



def main():

	dim_out = 3
	dim_in = 5
	Nomegas = 1000
	Nrollouts = 10


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




if __name__ == "__main__":

	main()





