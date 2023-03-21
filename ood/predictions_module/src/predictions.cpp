/* --------------------------------------------------------------------------------------------------------------- */
/* ------------------------------------------ Position Holder ---------------------------------------------------- */
/* --------------------------------------------------------------------------------------------------------------- */


#include <predictions.hpp>

#include <chrono>

std::vector<std::vector<Eigen::MatrixXd>> Predictions::run_all_rollouts_for_entire_trajectory(  const Eigen::Ref<const Eigen::MatrixXd>& x_traj_real,
                                                                                                const Eigen::Ref<const Eigen::MatrixXd>& u_traj_real,
                                                                                                size_t Nhor){



    // std::vector<std::vector<Eigen::MatrixXd>> xtraj_sampled_all_rollouts_for_entire_trajectory;
    std::vector<std::vector<Eigen::MatrixXd>> xtraj_sampled_all_rollouts_for_entire_trajectory;
    size_t Nsteps = u_traj_real.rows() - Nhor;


    for(int tt=0; tt<Nsteps; tt++){

        std::chrono::steady_clock::time_point time_begin = std::chrono::steady_clock::now();

        std::cout << "tt: " << tt << "\n";
        Eigen::MatrixXd xt = x_traj_real.row(tt);
        // Eigen::MatrixXd u_traj_real_sliced = u_traj_real(Eigen::ArithmeticSequence::seqN(tt,Nhor),Eigen::placeholders::all);
        Eigen::MatrixXd u_traj_real_sliced = u_traj_real(Eigen::seqN(tt,Nhor),Eigen::indexing::all);
        xtraj_sampled_all_rollouts_for_entire_trajectory.push_back(this->run_all_rollouts_from_current_state(xt,u_traj_real_sliced));

        std::chrono::steady_clock::time_point time_end = std::chrono::steady_clock::now();

        std::cout << "Time difference = " << float(std::chrono::duration_cast<std::chrono::microseconds>(time_end - time_begin).count()/1000.0) << " [ms]" << std::endl;

    }

    return xtraj_sampled_all_rollouts_for_entire_trajectory; // [Nsteps, Nrollouts, Nhor, dim_out]
}



std::vector<Eigen::MatrixXd> Predictions::run_all_rollouts_from_current_state(   const Eigen::Ref<const Eigen::MatrixXd>& x0,
                                                        const Eigen::Ref<const Eigen::MatrixXd>& u_traj){

    // size_t Nrollouts = this->noise_mat.rows();
    
    // std::vector<Eigen::MatrixXd> xtraj_sampled_all_rollouts;
    // std::array<Eigen::MatrixXd, Nrollouts> xtraj_sampled_all_rollouts;

    // noise_mat: [Nrollouts,Nomegas]
    for(int rr=0; rr<Nrollouts; rr++){
        // std::cout << "this->noise_mat.row(rr):" << this->noise_mat.row(rr) << "\n";
        // xtraj_sampled_all_rollouts.push_back(this->run_one_rollout_from_current_state(x0,u_traj,this->noise_mat.row(rr)));
        this->xtraj_sampled_all_rollouts[rr] = this->run_one_rollout_from_current_state(x0,u_traj,this->noise_mat.row(rr),rr,false);
    }

    return this->xtraj_sampled_all_rollouts;
}



Eigen::MatrixXd Predictions::run_one_rollout_from_current_state(   const Eigen::Ref<const Eigen::MatrixXd>& x0,
                                                        const Eigen::Ref<const Eigen::MatrixXd>& u_traj,
                                                        const Eigen::Ref<const Eigen::MatrixXd>& vec_noise,
                                                        int ind_rollout,
                                                        bool verbo = false){

    Eigen::MatrixXd xt_next_delta = Eigen::MatrixXd::Zero(1,this->dim_out); // [1,dim_out]
    Eigen::MatrixXd xt_next = Eigen::MatrixXd::Zero(1,this->dim_out); // [1,dim_out]
    // vec_noise: [1,Nomegas]


    size_t Nsteps_u = u_traj.rows();
    Eigen::MatrixXd X_in(1, x0.cols()+u_traj.cols());



    Eigen::MatrixXd xtraj_sampled = Eigen::MatrixXd::Zero(Nsteps_u+1,this->dim_out); // [Nsteps_u+1,dim_out]
    
    xtraj_sampled.row(0) = x0;
    xt_next = x0;
    for(int tt=0; tt<Nsteps_u; tt++){
        
        // Prepare input:
        X_in << xt_next, u_traj.row(tt);

        // Predict:
        xt_next_delta = this->get_ft_sampled_all_dims(X_in,vec_noise,ind_rollout,tt==0 && verbo==true);

        // Actual next state:
        xt_next += xt_next_delta;

        // Store:
        xtraj_sampled.row(tt+1) = xt_next;

    }


    return xtraj_sampled;
}



Eigen::MatrixXd Predictions::get_ft_sampled_all_dims(const Eigen::Ref<const Eigen::MatrixXd>& X, const Eigen::Ref<const Eigen::MatrixXd>& vec_noise, int ind_rollout, bool verbo = false){

    Eigen::MatrixXd ft_sampled = Eigen::MatrixXd::Zero(1,this->dim_out);
    Eigen::MatrixXd sampled_beta = Eigen::MatrixXd::Zero(1,this->Nomegas);

    // vec_noise: [1,Nomegas]



    this->update_features_mat(X);
    for(int dd=0; dd < this->dim_out; dd++){
        
        // sampled_beta = tf.reshape(mean_beta,(-1,1)) + cov_beta_chol @ sample_mv0 # [Nfeat,1] + [Nfeat,Nfeat] @ [Nfeat,Nsamples]
        // sampled_beta = this->mean_beta_pred_all_dim.row(dd) + this->cov_beta_pred_chol_all_dim[dd] * vec_noise.transpose(); // [1,Nomegas] + [1,Nomegas]*[Nomegas,Nomegas]
        // sampled_beta = this->mean_beta_pred_all_dim.row(dd) + vec_noise * this->cov_beta_pred_chol_all_dim[dd].transpose(); // [1,Nomegas] + [1,Nomegas]*[Nomegas,Nomegas]

        
        
        // ft_sampled.col(dd) = this->phi_mat[dd] * sampled_beta.transpose(); // Dot product, should return a scalar
        ft_sampled.col(dd) = this->phi_mat[dd] * this->sampled_beta_mat[dd].row(ind_rollout).transpose(); // Dot product, should return a scalar

        if(verbo){
            std::cout << "mean_beta_pred_all_dim(dd,Eigen::seqN(0,10)): " << this->mean_beta_pred_all_dim(dd,Eigen::seqN(0,10)).format(this->clean_format) << "\n";
            std::cout << "cov_beta_pred_chol_all_dim(dd,Eigen::seqN(0,5),Eigen::seqN(0,5)): " << this->cov_beta_pred_chol_all_dim[dd](Eigen::seqN(0,5),Eigen::seqN(0,5)).format(this->clean_format) << "\n";
            std::cout << "sampled_beta: " << sampled_beta(0,Eigen::seqN(0,10)).format(this->clean_format) << "\n";
            std::cout << "this->phi_mat[dd]: " << this->phi_mat[dd](0,Eigen::seqN(0,5)).format(this->clean_format) << "\n";
        }

    }


    if(verbo){
        std::cout << "ft_sampled: " << ft_sampled.format(this->clean_format) << "\n";
    }

    

    return ft_sampled; // [1,dim_out]
}


void Predictions::update_features_mat(const Eigen::Ref<const Eigen::MatrixXd>& X){

    
    // Eigen::MatrixXd phi_mat = Eigen::MatrixXd::Zero(X.rows(),this->Nomegas);

    for(int dd=0; dd < this->dim_out; dd++){
        this->phi_mat[dd] = this->get_features_one_channel(X,dd); // [dim_out, Nxpoints=1, Nomegas]
    }

    return;
}

Eigen::MatrixXd Predictions::get_features_one_channel(const Eigen::Ref<const Eigen::MatrixXd>& X, size_t ind_dim){


    // WX = X @ tf.transpose(self.W_samples) # [Nxpoints=1, Nomegas]
    // harmonics_vec = tf.math.cos(WX + tf.transpose(self.phi_samples_vec) + self.dbg_phase_added_to_features) # [Nxpoints=1, Nomegas]



    Eigen::MatrixXd WX_plit_phi = X * this->W_samples_all_dim[ind_dim].transpose() + this->phi_samples_all_dim.row(ind_dim);

    Eigen::MatrixXd harmonics_vec = WX_plit_phi.array().cos(); // .array() compues the cos() element-wise

    // std::cout << "WX_plit_phi: " << WX_plit_phi.format(this->clean_format) << "\n";
    // std::cout << "harmonics_vec: " << harmonics_vec.format(this->clean_format) << "\n";

    return harmonics_vec; // [Nxpoints=1, Nomegas]
}


// void Predictions::update_sampled_betas(void){



//     for(int dd=0; dd < this->dim_out; dd++){
//         sampled_beta = mean_beta_pred_all_dim.row(dd) + vec_noise * cov_beta_pred_chol_all_dim[dd]; // [1,Nomegas] + [1,Nomegas]*[Nomegas,Nomegas]
//         ft_sampled.col(dd) = this->phi_mat[dd] * sampled_beta.transpose(); // Dot product, should return a scalar
//     }



// }


