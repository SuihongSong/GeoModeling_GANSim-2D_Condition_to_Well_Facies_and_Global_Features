import numpy as np
import tensorflow as tf
import tfutil

#----------------------------------------------------------------------------
# Convenience func that casts all of its arguments to tf.float32.

def fp32(*values):
    if len(values) == 1 and isinstance(values[0], tuple):
        values = values[0]
    values = tuple(tf.cast(v, tf.float32) for v in values)
    return values if len(values) >= 2 else values[0]

#----------------------------------------------------------------------------
# Generator loss function used in the paper.

def G_wgan_acgan(G, D, opt, lod, training_set, labels, minibatch_size, labeltypes = None, MudProp_weight = 10.0, Width_weight = 10.0, Sinuosity_weight = 10.0):
    #labeltypes, e.g., labeltypes = [1]  # can include: 0 for 'channelorientation', 1 for 'mudproportion', 2 for 'channelwidth', 3 for 'channelsinuosity', set in config file
    # loss for channel orientation is not designed below, so do not include "0" in labeltypes.
    
    latents = tf.random_normal([minibatch_size] + G.input_shapes[0][1:])
    fake_images_out = G.get_output_for(latents, labels, is_training=True) 
    fake_scores_out, fake_labels_out = fp32(D.get_output_for(fake_images_out, is_training=True))
    loss = -fake_scores_out     

    with tf.name_scope('LabelPenalty'):
        def addMudPropPenalty(index):
            MudPropPenalty = tf.nn.l2_loss(labels[:, index] - fake_labels_out[:, index]) 
            MudPropPenalty = tfutil.autosummary('Loss_G/MudPropPenalty', MudPropPenalty)        
            MudPropPenalty = MudPropPenalty * MudProp_weight  
            return loss+MudPropPenalty
        if 1 in labeltypes:
            ind = labeltypes.index(1)  # because labeltypes might be like [2,1,3]
            loss = addMudPropPenalty(ind)
        
        def addWidthPenalty(index):
            WidthPenalty = tf.nn.l2_loss(labels[:, index] - fake_labels_out[:, index]) 
            WidthPenalty = tfutil.autosummary('Loss_G/WidthPenalty', WidthPenalty)             
            WidthPenalty = WidthPenalty * Width_weight            
            return loss+WidthPenalty
        if 2 in labeltypes:
            ind = labeltypes.index(2)
            loss = addWidthPenalty(ind) #tf.cond(tf.math.less(lod, tf.fill([], 5.)), lambda: addWidthPenalty(ind), lambda: loss)
        
        def addSinuosityPenalty(index):
            SinuosityPenalty = tf.nn.l2_loss(labels[:, index] - fake_labels_out[:, index]) 
            SinuosityPenalty = tfutil.autosummary('Loss_G/SinuosityPenalty', SinuosityPenalty)            
            SinuosityPenalty = SinuosityPenalty * Sinuosity_weight              
            return loss+SinuosityPenalty
        if 3 in labeltypes:
            ind = labeltypes.index(3)
            loss = addSinuosityPenalty(ind) #tf.cond(tf.math.less(lod, tf.fill([], 5.)), lambda: addSinuosityPenalty(ind), lambda: loss)    
            
    loss = tfutil.autosummary('Loss_G/Total_loss', loss)
    return loss

#----------------------------------------------------------------------------
# Discriminator loss function used in the paper.

def D_wgangp_acgan(G, D, opt, training_set, minibatch_size, reals, labels,
    wgan_lambda     = 10.0,     # Weight for the gradient penalty term.
    wgan_epsilon    = 0.001,    # Weight for the epsilon term, \epsilon_{drift}.
    wgan_target     = 1.0,      # Target value for gradient magnitudes.
    cond_weight     = 10.0):     # Weight of the conditioning terms.

    latents = tf.random_normal([minibatch_size] + G.input_shapes[0][1:])
    fake_images_out = G.get_output_for(latents, labels, is_training=True)
    real_scores_out, real_labels_out = fp32(D.get_output_for(reals, is_training=True))
    fake_scores_out, fake_labels_out = fp32(D.get_output_for(fake_images_out, is_training=True))
    real_scores_out = tfutil.autosummary('Loss_D/real_scores', real_scores_out)
    fake_scores_out = tfutil.autosummary('Loss_D/fake_scores', fake_scores_out)
    loss = fake_scores_out - real_scores_out

    with tf.name_scope('GradientPenalty'):
        mixing_factors = tf.random_uniform([minibatch_size, 1, 1, 1], 0.0, 1.0, dtype=fake_images_out.dtype)
        mixed_images_out = tfutil.lerp(tf.cast(reals, fake_images_out.dtype), fake_images_out, mixing_factors)
        mixed_scores_out, mixed_labels_out = fp32(D.get_output_for(mixed_images_out, is_training=True))
        #mixed_scores_out = tfutil.autosummary('Loss_D/mixed_scores', mixed_scores_out)
        mixed_loss = opt.apply_loss_scaling(tf.reduce_sum(mixed_scores_out))
        mixed_grads = opt.undo_loss_scaling(fp32(tf.gradients(mixed_loss, [mixed_images_out])[0]))
        mixed_norms = tf.sqrt(tf.reduce_sum(tf.square(mixed_grads), axis=[1,2,3]))
        mixed_norms = tfutil.autosummary('Loss_D/mixed_norms', mixed_norms)
        gradient_penalty = tf.square(mixed_norms - wgan_target)
    loss += gradient_penalty * (wgan_lambda / (wgan_target**2))
    loss = tfutil.autosummary('Loss_D/WGAN_GP_loss', loss)

    with tf.name_scope('EpsilonPenalty'):
        epsilon_penalty = tfutil.autosummary('Loss_D/epsilon_penalty', tf.square(real_scores_out))
    loss += epsilon_penalty * wgan_epsilon

    if D.output_shapes[1][1] > 0:
        with tf.name_scope('LabelPenalty'):                   
            label_penalty_reals = tf.nn.l2_loss(labels - real_labels_out) 
            label_penalty_fakes = tf.nn.l2_loss(labels - fake_labels_out) 
            label_penalty_reals = tfutil.autosummary('Loss_D/label_penalty_reals', label_penalty_reals)
            label_penalty_fakes = tfutil.autosummary('Loss_D/label_penalty_fakes', label_penalty_fakes)
        loss += (label_penalty_reals + label_penalty_fakes) * cond_weight
        loss = tfutil.autosummary('Loss_D/Total_loss', loss)
    return loss

#----------------------------------------------------------------------------
