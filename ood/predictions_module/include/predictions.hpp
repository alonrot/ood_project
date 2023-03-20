


/************************************************************************
*
* @brief Trajectory prediction
* 
*
* @author Alonso Marco
* Contact: amarco@berkeley.edu
************************************************************************/

#ifndef _PREDICTIONS_H_
#define _PREDICTIONS_H_


#include <chrono>
#include <thread>
#include <array>
#include <math.h>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <iostream>
#include <vector>

// typedef Eigen::Matrix<double, 12, 1> Vector12d; // Column vector by default; https://eigen.tuxfamily.org/dox/group__TutorialMatrixClass.html

class Predictions
{
public:
    Predictions(size_t dim_in,
                size_t dim_out,
                Eigen::MatrixXd phi_samples_all_dim,
    			std::vector<Eigen::MatrixXd> W_samples_all_dim,
    			Eigen::MatrixXd mean_beta_pred_all_dim,
    			std::vector<Eigen::MatrixXd> cov_beta_pred_chol_all_dim,
                Eigen::MatrixXd noise_mat){

	    
	    this->dim_in = dim_in;
	    this->dim_out = dim_out;

        this->clean_format = Eigen::IOFormat(4, 0, ", ", "\n", "[", "]");

        this->Nomegas = this->phi_samples_all_dim.cols();


        for(int ii=0;ii<this->dim_out;++ii){
            this->phi_mat.push_back(Eigen::MatrixXd::Zero(1,this->Nomegas));
        }

        this->phi_samples_all_dim = phi_samples_all_dim;
        this->mean_beta_pred_all_dim = mean_beta_pred_all_dim;
        
        this->W_samples_all_dim = W_samples_all_dim;
        this->cov_beta_pred_chol_all_dim = cov_beta_pred_chol_all_dim;

        this->noise_mat = noise_mat;

        
        // std::cout << "this->phi_samples_all_dim: " << this->phi_samples_all_dim.format(this->clean_format) << "\n";
        // std::cout << "this->mean_beta_pred_all_dim: " << this->mean_beta_pred_all_dim.format(this->clean_format) << "\n";
        // std::cout << "this->W_samples_all_dim: " << this->W_samples_all_dim.format(this->clean_format) << "\n";
        // std::cout << "this->cov_beta_pred_chol_all_dim: " << this->cov_beta_pred_chol_all_dim.format(this->clean_format) << "\n";

    }

    ~Predictions(){
        std::cout << "Destroying Predictions class ...\n";
    }


    std::vector<std::vector<Eigen::MatrixXd>> run_all_rollouts_for_entire_trajectory( const Eigen::Ref<const Eigen::MatrixXd>& x_traj_real,
                                            const Eigen::Ref<const Eigen::MatrixXd>& u_traj_real,
                                            size_t Nhor);

    std::vector<Eigen::MatrixXd> run_all_rollouts_from_current_state(const Eigen::Ref<const Eigen::MatrixXd>& x0,
                                        const Eigen::Ref<const Eigen::MatrixXd>& u_traj);

    Eigen::MatrixXd run_one_rollout_from_current_state( const Eigen::Ref<const Eigen::MatrixXd>& x0,
                                        const Eigen::Ref<const Eigen::MatrixXd>& u_traj,
                                        const Eigen::Ref<const Eigen::MatrixXd>& vec_noise);
    Eigen::MatrixXd get_ft_sampled_all_dims(const Eigen::Ref<const Eigen::MatrixXd>& X, const Eigen::Ref<const Eigen::MatrixXd>& vec_noise);
    void update_features_mat(const Eigen::Ref<const Eigen::MatrixXd>& X);
    Eigen::MatrixXd get_features_one_channel(const Eigen::Ref<const Eigen::MatrixXd>& X, size_t ind_dim);

    size_t dim_x;
    size_t dim_u;
    size_t dim_in;
    size_t dim_out;
    size_t Nomegas;

    Eigen::MatrixXd phi_samples_all_dim; 		// [dim_out,Nomegas]
    std::vector<Eigen::MatrixXd> W_samples_all_dim; // [dim_out,Nomegas,dim_in]
    Eigen::MatrixXd mean_beta_pred_all_dim; 	// [dim_out,Nomegas]
    std::vector<Eigen::MatrixXd> cov_beta_pred_chol_all_dim; // [dim_out,Nomegas,Nomegas]

    Eigen::MatrixXd noise_mat;

    // Eigen::MatrixXd S_samples_all_dim; 			// [dim_out,Nomegas]
    // Eigen::MatrixXd dw_samples_all_dim; 		// [dim_out,Nomegas]
    // Eigen::MatrixXd chol_corr_noise_mat; 		// [dim_out,dim_out]


    // std::vector<Eigen::MatrixXd> xtraj_sampled_all_rollouts; // [Nrollouts,Nhorizon_rec,dim_x]
    // std::array<std::array<Eigen::MatrixXd>> x_traj_pred_all_vec;

    std::vector<Eigen::MatrixXd> phi_mat; // One per output channel

    // x_traj_pred_all_vec = np.zeros((Nsteps_tot,Nrollouts,Nhorizon_rec,dim_x)) # For plotting

    // Flags:

    Eigen::IOFormat clean_format;
protected:

};


#endif  // _PREDICTIONS_H_
