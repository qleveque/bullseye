def prior_triplet(prior_std):
        #if prior_std.shape.as_list()[0] == 0 :
        if tf.shape(prior_std)[0] == 0:
            return tf.zeros([]), tf.zeros([p]), tf.zeros([p,p])
        assert prior_std.shape.as_list()[1] == p
        
        #TODO test
        #multipliers = tf.where(tf.cast(prior_std,bool), prior_std/tf.ones_like(prior_std), tf.zeros_like(prior_std))
        
        multipliers = prior_std
        
        if G.compute_prior_kernel:
            hess_multipliers = tf.multiply(tf.expand_dims(multipliers, 2), tf.expand_dims(multipliers, 1))
        local_mu_prior = tf.einsum('ij,j->i', multipliers, new_mu)
        
        #TODO add prior_std_trick
        if G.compute_prior_kernel:
            local_std_prior = tf.sqrt(tf.einsum('ijk,jk->i', hess_multipliers, cov)) # [p]
        else:
            local_std_prior = tf.sqrt(tf.einsum('ij,jk,ik->i', multipliers, cov, multipliers))
        
        # activations : [quadrature_deg, p]
        activations_prior = tf.expand_dims(local_mu_prior,0) +  tf.expand_dims(local_std_prior,0) * tf.expand_dims(z_prior, 1)
        phi_prior = tf.square(activations_prior) / 2 # [quadrature_deg, p]
        phi_grad_prior = activations_prior # [quadrature_deg, p]
        phi_hessian_prior = tf.ones_like(activations_prior)    # [quadrature_deg, p]

        mean_phi_prior = tf.einsum('i,ij->j', z_weights_prior, phi_prior)
        mean_phi_grad_prior = tf.einsum('i,ij->j', z_weights_prior, phi_grad_prior)
        mean_phi_hessian_prior= tf.einsum('i,ij->j', z_weights_prior, phi_hessian_prior)

        computed_e_prior = tf.reduce_sum(mean_phi_prior)
        computed_rho_prior  = tf.einsum('i,ij->j',   mean_phi_grad_prior,    multipliers)
        
        if G.compute_prior_kernel:
            computed_beta_prior = tf.einsum('i,ijk->jk', mean_phi_hessian_prior, hess_multipliers)
        else:
            computed_beta_prior = tf.einsum('ij,i,ik->jk', multipliers, mean_phi_hessian_prior, multipliers)
            
        return computed_e_prior, computed_rho_prior, computed_beta_prior
        

