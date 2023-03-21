


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

/*
Nomegas = 1500
dim_in = 5
dim_out = 3
Nrollouts = 20
Nhor = 25
*/

// // Column vector by default; https://eigen.tuxfamily.org/dox/group__TutorialMatrixClass.html
// typedef Eigen::Matrix<double, 12, 1> Vector12d;
// typedef std::array<Eigen::Matrix<double, 1, 1500>,3> FeaturesMatrix; // One per output channel [dim_out,1,Nomegas]
// typedef std::array<Eigen::Matrix<double, 25, 3>,20> SampledStatesTrajectory; // [Nrollouts,Nhorizon_rec,dim_x]
// // std::array<Eigen::MatrixXd, 20> xtraj_sampled_all_rollouts; // [Nrollouts,Nhorizon_rec,dim_x]


class Predictions
{
public:
    Predictions(size_t dim_in,
                size_t dim_out,
                Eigen::MatrixXd phi_samples_all_dim,
    			std::vector<Eigen::MatrixXd> W_samples_all_dim,
    			Eigen::MatrixXd mean_beta_pred_all_dim,
    			std::vector<Eigen::MatrixXd> cov_beta_pred_chol_all_dim,
                Eigen::MatrixXd noise_mat,
                size_t Nrollouts,
                size_t Nhor){

	    
	    this->dim_in = dim_in;
	    this->dim_out = dim_out;
        this->phi_samples_all_dim = phi_samples_all_dim;
        this->W_samples_all_dim = W_samples_all_dim;
        this->mean_beta_pred_all_dim = mean_beta_pred_all_dim;
        this->cov_beta_pred_chol_all_dim = cov_beta_pred_chol_all_dim;
        this->noise_mat = noise_mat;

        this->Nrollouts = Nrollouts;
        this->Nhor = Nhor;

        assert(this->noise_mat.rows() == this->Nrollouts);
        
        // Parameters:
        this->Nomegas = this->phi_samples_all_dim.cols();

        // Others:
        this->clean_format = Eigen::IOFormat(4, 0, ", ", "\n", "[", "]");

        // Pre-allocate to avoid dynamic allocation:
        for(int ii=0;ii<this->dim_out;++ii){
            this->phi_mat.push_back(Eigen::MatrixXd::Zero(1,this->Nomegas));
        }

        Eigen::MatrixXd sampled_beta_mat_aux_dd;
        sampled_beta_mat_aux_dd = Eigen::MatrixXd::Zero(this->Nrollouts,this->Nomegas);
        for(int dd=0;dd<this->dim_out;++dd){
            for(int rr=0;rr<this->Nrollouts;++rr){
                sampled_beta_mat_aux_dd.row(rr) = this->mean_beta_pred_all_dim.row(dd) + this->noise_mat.row(rr) * this->cov_beta_pred_chol_all_dim[dd].transpose(); // [1,Nomegas] + [1,Nomegas]*[Nomegas,Nomegas]
            }
            this->sampled_beta_mat.push_back(sampled_beta_mat_aux_dd);
        }


        for(int rr=0;rr<this->Nrollouts;++rr){
            this->xtraj_sampled_all_rollouts.push_back(Eigen::MatrixXd::Zero(this->Nhor,this->dim_out)); // [Nrollouts,Nhorizon_rec,dim_x]
        }




    }

    ~Predictions(){std::cout << "Destroying Predictions class ...\n";}


    std::vector<std::vector<Eigen::MatrixXd>> run_all_rollouts_for_entire_trajectory( const Eigen::Ref<const Eigen::MatrixXd>& x_traj_real,
                                            const Eigen::Ref<const Eigen::MatrixXd>& u_traj_real,
                                            size_t Nhor);

    std::vector<Eigen::MatrixXd> run_all_rollouts_from_current_state(const Eigen::Ref<const Eigen::MatrixXd>& x0,
                                        const Eigen::Ref<const Eigen::MatrixXd>& u_traj);

    Eigen::MatrixXd run_one_rollout_from_current_state( const Eigen::Ref<const Eigen::MatrixXd>& x0,
                                        const Eigen::Ref<const Eigen::MatrixXd>& u_traj,
                                        const Eigen::Ref<const Eigen::MatrixXd>& vec_noise,
                                        int ind_rollout,
                                        bool verbo);
    Eigen::MatrixXd get_ft_sampled_all_dims(const Eigen::Ref<const Eigen::MatrixXd>& X, const Eigen::Ref<const Eigen::MatrixXd>& vec_noise, int ind_rollout, bool verbo);
    void update_features_mat(const Eigen::Ref<const Eigen::MatrixXd>& X);
    Eigen::MatrixXd get_features_one_channel(const Eigen::Ref<const Eigen::MatrixXd>& X, size_t ind_dim);

    size_t dim_x;
    size_t dim_u;
    size_t dim_in;
    size_t dim_out;
    size_t Nomegas;


    Eigen::MatrixXd phi_samples_all_dim; 		// [dim_out,Nomegas] || phi_samples_all_dim = np.zeros((self.dim_out,Nomegas))
    std::vector<Eigen::MatrixXd> W_samples_all_dim; // [dim_out,Nomegas,dim_in] || W_samples_all_dim = np.zeros((self.dim_out,Nomegas,self.dim_in))
    Eigen::MatrixXd mean_beta_pred_all_dim; 	// [dim_out,Nomegas] || mean_beta_pred_all_dim = np.zeros((self.dim_out,Nomegas))
    std::vector<Eigen::MatrixXd> cov_beta_pred_chol_all_dim; // [dim_out,Nomegas,Nomegas] || cov_beta_pred_chol_all_dim = np.zeros((self.dim_out,Nomegas,Nomegas))

    Eigen::MatrixXd noise_mat;

    // Eigen::MatrixXd S_samples_all_dim; 			// [dim_out,Nomegas]
    // Eigen::MatrixXd dw_samples_all_dim; 		// [dim_out,Nomegas]
    // Eigen::MatrixXd chol_corr_noise_mat; 		// [dim_out,dim_out]


    std::vector<Eigen::MatrixXd> xtraj_sampled_all_rollouts; // [Nrollouts,Nhorizon_rec,dim_x]
    // std::array<std::array<Eigen::MatrixXd>> x_traj_pred_all_vec;

    std::vector<Eigen::MatrixXd> phi_mat; // One per output channel [dim_out,1,Nomegas]

    // x_traj_pred_all_vec = np.zeros((Nsteps_tot,Nrollouts,Nhorizon_rec,dim_x)) # For plotting

    std::vector<Eigen::MatrixXd> sampled_beta_mat;


    // Efficiency:
    size_t Nrollouts;
    size_t Nhor;
    // std::array<Eigen::MatrixXd, 20> xtraj_sampled_all_rollouts; // [Nrollouts,Nhorizon_rec,dim_x]

    // Flags:

    Eigen::IOFormat clean_format;
protected:

};


#endif  // _PREDICTIONS_H_
