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
# Generator loss function.

def G_wgan_acgan(G, D, opt, lod, labels, well_facies, minibatch_size,
    Wellfaciesloss_weight = 500, MudProp_weight = 10, Width_weight = 20, Sinuosity_weight = 5, orig_weight = 1, labeltypes = None, lossnorm = False): 
#labeltypes, e.g., labeltypes = [1]  # can include: 0 for 'channelorientation', 1 for 'mudproportion', 2 for 'channelwidth', 3 for 'channelsinuosity', set in config file
    # loss for channel orientation is not designed below, so do not include "0" in labeltypes.
    # lossnorm: True to normalize loss into standard Gaussian before multiplying with weights.
    
    label_size = len(labeltypes)
    if label_size == 0: 
        labels_in = labels
    else:     
        labels_list = []
        for k in range(label_size):
            labels_list.append(tf.random.uniform(([minibatch_size]), minval=-1, maxval=1))
        if 1 in labeltypes:   # mud proportion
            ind = labeltypes.index(1)
            labels_list[ind] = tf.clip_by_value(labels[:, ind] + tf.random.uniform([minibatch_size], minval=-0.2, maxval=0.2), -1, 1)    
        labels_in = tf.stack(labels_list, axis = 1)          
    
    latents = tf.random_normal([minibatch_size] + G.input_shapes[0][1:])
    well_facies = tf.cast(well_facies, tf.float32)
    fake_images_out = G.get_output_for(latents, labels_in, well_facies, is_training=True)  
    fake_scores_out, fake_labels_out = fp32(D.get_output_for(fake_images_out, is_training=True))
    loss = -fake_scores_out
    if lossnorm: loss = (loss -14.6829250772099) / 4.83122039859412   #To Normalize
    loss = tfutil.autosummary('Loss_G/GANloss', loss)
    loss = loss * orig_weight     

    with tf.name_scope('LabelPenalty'):
        def addMudPropPenalty(index):
            MudPropPenalty = tf.nn.l2_loss(labels_in[:, index] - fake_labels_out[:, index]) # [:,0] is the inter-channel mud facies ratio 
            if lossnorm: MudPropPenalty = (MudPropPenalty -0.36079434843794) / 0.11613414177144  # To normalize this loss 
            MudPropPenalty = tfutil.autosummary('Loss_G/MudPropPenalty', MudPropPenalty)        
            MudPropPenalty = MudPropPenalty * MudProp_weight  
            return loss+MudPropPenalty
        if 1 in labeltypes:
            ind = labeltypes.index(1)
            loss = addMudPropPenalty(ind)
        
        def addWidthPenalty(index):
            WidthPenalty = tf.nn.l2_loss(labels_in[:, index] - fake_labels_out[:, index]) # [:,0] is the inter-channel mud facies ratio 
            if lossnorm: WidthPenalty = (WidthPenalty -0.600282781464712) / 0.270670509379704  # To normalize this loss 
            WidthPenalty = tfutil.autosummary('Loss_G/WidthPenalty', WidthPenalty)             
            WidthPenalty = WidthPenalty * Width_weight            
            return loss+WidthPenalty
        if 2 in labeltypes:
            ind = labeltypes.index(2)
            loss = tf.cond(tf.math.less(lod, tf.fill([], 5.)), lambda: addWidthPenalty(ind), lambda: loss)
        
        def addSinuosityPenalty(index):
            SinuosityPenalty = tf.nn.l2_loss(labels_in[:, index] - fake_labels_out[:, index]) # [:,0] is the inter-channel mud facies ratio 
            if lossnorm: SinuosityPenalty = (SinuosityPenalty -0.451279248935835) / 0.145642580091667  # To normalize this loss 
            SinuosityPenalty = tfutil.autosummary('Loss_G/SinuosityPenalty', SinuosityPenalty)            
            SinuosityPenalty = SinuosityPenalty * Sinuosity_weight              
            return loss+SinuosityPenalty
        if 3 in labeltypes:
            ind = labeltypes.index(3)
            loss = tf.cond(tf.math.less(lod, tf.fill([], 5.)), lambda: addSinuosityPenalty(ind), lambda: loss)
                
    def Wellpoints_L2loss(well_facies, fake_images):
        loss = tf.nn.l2_loss(well_facies[:,0:1]* (well_facies[:,1:2] - tf.where((fake_images+1)/2>0.4, tf.fill(tf.shape(fake_images), 1.), (fake_images+1)/2)))
        loss = loss / tf.reduce_sum(well_facies[:, 0:1])
        return loss
    def addwellfaciespenalty(well_facies, fake_images_out, loss, Wellfaciesloss_weight):
        with tf.name_scope('WellfaciesPenalty'):
            WellfaciesPenalty =  Wellpoints_L2loss(well_facies, fake_images_out)   # as levee is 0.5, in well_facies data, levee and channels' codes are all 1        
            if lossnorm: WellfaciesPenalty = (WellfaciesPenalty -0.00887323171768953) / 0.00517647244943928
            WellfaciesPenalty = tfutil.autosummary('Loss_G/WellfaciesPenalty', WellfaciesPenalty)
            loss += WellfaciesPenalty * Wellfaciesloss_weight   
        return loss   
    loss = tf.cond(tf.math.less_equal(lod, tf.fill([], 4.)), lambda: addwellfaciespenalty(well_facies, fake_images_out, loss, Wellfaciesloss_weight), lambda: loss)
       
    loss = tfutil.autosummary('Loss_G/Total_loss', loss)
    
    return loss

#----------------------------------------------------------------------------
# Discriminator loss function.
def D_wgangp_acgan(G, D, opt, minibatch_size, reals, labels, well_facies,
    wgan_lambda     = 10.0,     # Weight for the gradient penalty term.
    wgan_epsilon    = 0.001,    # Weight for the epsilon term, \epsilon_{drift}.
    wgan_target     = 1.0,      # Target value for gradient magnitudes.
    label_weight     = 10):     # Weight of the conditioning terms.

    latents = tf.random_normal([minibatch_size] + G.input_shapes[0][1:])
    fake_images_out = G.get_output_for(latents, labels, well_facies, is_training=True)
    real_scores_out, real_labels_out = fp32(D.get_output_for(reals, is_training=True))
    fake_scores_out, fake_labels_out = fp32(D.get_output_for(fake_images_out, is_training=True))
    real_scores_out = tfutil.autosummary('Loss_D/real_scores', real_scores_out)
    fake_scores_out = tfutil.autosummary('Loss_D/fake_scores', fake_scores_out)
    loss = fake_scores_out - real_scores_out 

    with tf.name_scope('GradientPenalty'):
        mixing_factors = tf.random_uniform([minibatch_size, 1, 1, 1], 0.0, 1.0, dtype=fake_images_out.dtype)
        mixed_images_out = tfutil.lerp(tf.cast(reals, fake_images_out.dtype), fake_images_out, mixing_factors)
        mixed_scores_out, mixed_labels_out = fp32(D.get_output_for(mixed_images_out, is_training=True))
        mixed_scores_out = tfutil.autosummary('Loss_D/mixed_scores', mixed_scores_out)
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

    with tf.name_scope('LabelPenalty'):
        label_penalty_reals = tf.nn.l2_loss(labels - real_labels_out)                            
        label_penalty_fakes = tf.nn.l2_loss(labels - fake_labels_out)
        label_penalty_reals = tfutil.autosummary('Loss_D/label_penalty_reals', label_penalty_reals)
        label_penalty_fakes = tfutil.autosummary('Loss_D/label_penalty_fakes', label_penalty_fakes)
        loss += (label_penalty_reals + label_penalty_fakes) * label_weight
        loss = tfutil.autosummary('Loss_D/Total_loss', loss)
    return loss